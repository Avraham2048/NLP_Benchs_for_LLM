import os
import re
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# --- API Clients & Setup ---
from dotenv import load_dotenv
import anthropic
from termcolor import colored

try:
    from google import genai
    GOOGLE_SDK_AVAILABLE = True
except ImportError:
    GOOGLE_SDK_AVAILABLE = False

load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

print(f"Google API Key loaded: {GOOGLE_API_KEY is not None}")
print(f"Anthropic API Key loaded: {ANTHROPIC_API_KEY is not None}")

# Initialize clients
gemini_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_SDK_AVAILABLE else None
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# --- Synchronous API Call Functions ---

def call_gemini(prompt: str, model_name: str = "gemini-3-pro") -> str:
    global gemini_client
    try:
        response = gemini_client.models.generate_content(
            model=f"models/{model_name}",
            contents=prompt
        )
        if response.text:
            return response.text
        else:
            return "Error: Empty response from Gemini (possibly blocked by safety filters)."
    except Exception as e:
        print(f"Error calling {model_name}: {e}")
        return f"Error: {e}"

def call_claude(prompt: str, model_name: str = "claude-haiku-4-5") -> str:
    try:
        message = claude_client.messages.create(
            model=model_name,
            max_tokens=20000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Error calling {model_name}: {e}")
        return f"Error: {e}"

def call_llm(prompt, model_name):
    if model_name.startswith("gemini"):
        return call_gemini(prompt, model_name)
    elif model_name.startswith("claude"):
        return call_claude(prompt, model_name)
    else:
        msg = f"Error: model name '{model_name}' not recognized"
        print(msg)
        return msg

# --- File Utility Functions ---

def load_text(path: Path) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

# --- Data Structures ---

@dataclass
class Sub:
    """Holds all data for a single origin file's analysis."""
    # for reco
    origin_path: Path = None
    reco_dir: Path = None 
    reco_gt_path: Path = None
    reco_date: Optional[datetime] = None
    
    # for eval
    eval_dir: Path = None
    eval_scores: Dict[str, List[float]] = field(default_factory=dict)
    eval_date: Optional[datetime] = None
    
    # for reports
    reports_dir: Path = None

    def reco(self, reco_prompt: str, models_names: List[str]) -> None:
        """Runs all reco models on the origin text."""
        origin_text = load_text(self.origin_path)
        for model_name in models_names:
            path = self.reco_dir / f"{model_name}.txt"
            if path.exists():
                print(f"file in {path} already exists")
                continue
            prompt = reco_prompt + origin_text
            print(f"reco now  {self.origin_path.stem} with {model_name}")
            reco_text = call_llm(prompt, model_name)
            if 'error' not in reco_text.lower():
                save_text(path, reco_text)
            else:
                print(f"  [SKIPPED] {model_name} output contains error keyword.")
    
    def get_model_nickname(self, full_model_name: str) -> str:
        """Convert full model name to short nickname for display/saving."""
        mapping = {
            "claude-sonnet-4-5": "sonnet-4.5",
            "claude-haiku-4-5": "haiku-4.5",
            "claude-opus-4-6": "opus-4.6",
            "gemini-3-flash-preview": "flash-3",
            "gemini-3-pro-preview": "pro-3",
        }
        return mapping.get(full_model_name, full_model_name)

    def eval(self, eval_prompt: str, eval_model_name: str, reco_models_names: List[str]) -> None:
        """Runs evals on all recos"""
        reco_gt = load_text(self.reco_gt_path)
        for model_name in reco_models_names:
            path = self.eval_dir / f"{self.get_model_nickname(eval_model_name)}_evals_the_reco_of_{self.get_model_nickname(model_name)}.txt"
            if path.exists():
                print(f"file in {path} already exists")
                eval_text = load_text(path)
                self.eval_scores[model_name] = self.extract_scores_from_eval_text(eval_text)
                continue
            
            reco_file_path = self.reco_dir / f"{model_name}.txt"
            if not reco_file_path.exists():
                print(f"  [SKIPPED EVAL] Reco file missing for {model_name}. Skipping evaluation.")
                self.eval_scores[model_name] = [0.0] * 5 
                continue

            reco_text = load_text(reco_file_path)
            prompt = eval_prompt + reco_gt + reco_text
            print(f"evals now  {path.stem}")
            eval_text = call_llm(prompt, eval_model_name)

            if 'error' not in eval_text.lower():
                save_text(path, eval_text)
                self.eval_scores[model_name] = self.extract_scores_from_eval_text(eval_text)
            else:
                print(f"  [SKIPPED] Eval for {model_name} contains error keyword.")
                self.eval_scores[model_name] = [0.0] * 5

            self.eval_date = datetime.now()

    def extract_scores_from_eval_text(self, eval_text: str) -> List[float]:
        '''Extracts exactly 5 scores (0-100) from an evaluation text.'''
        scores = []
        last_50_words = ' '.join(eval_text.split()[-50:])
        match = re.search(r'\[([^\]]+)\](?!.*\[)', last_50_words)

        search_text = eval_text
        if match:
            search_text = match.group(1)
            print(f"  Info: Found score bracket: [{search_text}]")
        else:
            print("  Warning: No score bracket [] found. Searching whole text.")

        numbers = re.findall(r'\b\d+\.?\d*\b', search_text)
        scores = [float(num) for num in numbers if 0 <= float(num) <= 100]

        if len(scores) < 5:
            print(f"  Warning: Found only {len(scores)} scores. Padding with 0.0.")
            scores.extend([0.0] * (5 - len(scores)))
        elif len(scores) > 5:
            print(f"  Warning: Found {len(scores)} scores inside search area. Taking the last 5.")
            scores = scores[-5:]

        return scores


@dataclass
class SubCollection:
    """Manages a collection of Sub objects."""
    subs: List[Sub] = field(default_factory=list)
    reports_dir: Path = None

    def add(self, sub: Sub) -> None:
        self.subs.append(sub)

    def col_reco(self, reco_prompt: str, models_names: List[str]) -> None:
        for sub in self.subs:
            sub.reco(reco_prompt, models_names)

    def col_eval(self, eval_prompt: str, eval_model_name: str, reco_models_names: List[str]) -> None:
        for sub in self.subs:
            sub.eval(eval_prompt, eval_model_name, reco_models_names)


# --- Main Pipeline ---

class LLMPipeline:
    """Main class to run the reco and eval pipeline."""

    def __init__(self, base_dir: Path, reco_prompt: str, eval_prompt: str):
        self.base_dir = Path(base_dir)
        # for reco
        self.origins_dir = self.base_dir / "pp_origins"
        self.recos_dir = self.base_dir / "pp_recos"
        self.recogt_dir = self.base_dir / "reco_gt"
        self.reco_prompt = reco_prompt
        # for eval
        self.eval_dir = self.base_dir / "pp_evals"
        self.reports_dir = self.base_dir / "pp_reports"
        self.eval_prompt = eval_prompt

    def reco(self, models_names: List[str]) -> SubCollection:
        """Runs all reco models on the origin text."""
        sub_col = SubCollection(reports_dir=self.reports_dir)
        for origin_path in self.origins_dir.iterdir():
            if not origin_path.is_file():
                continue
            file_name = origin_path.name
            sub = Sub(origin_path=origin_path,
                      reco_dir=(self.recos_dir / origin_path.stem),
                      reco_gt_path=self.recogt_dir / file_name,
                      eval_dir=(self.eval_dir / origin_path.stem))
            sub_col.add(sub)
            
        sub_col.col_reco(self.reco_prompt, models_names)
        return sub_col

    def eval(self, sub_col: SubCollection, eval_model_name: str, reco_models_names: List[str]) -> None:
        sub_col.col_eval(self.eval_prompt, eval_model_name, reco_models_names)

# =======================
# VISUALIZATION & EXPORT
# =======================

ROOT_FOLDER = "pp_evals"
REPORT_FOLDER = "pp_reports"

if not os.path.exists(REPORT_FOLDER):
    os.makedirs(REPORT_FOLDER)

LLM_MAPPING = {
    "flash-3": "3 Flash",
    "pro-3": "3 Pro",
    "haiku-4.5": "Haiku 4.5",
    "sonnet-4.5": "Sonnet 4.5",
    "opus-4.6": "opus 4.6",
}

CRITERIA_NAMES = {
    1: "Precisions of categories",
    2: "Recall of categories",
    3: "Precisions of opinions",
    4: "Recall of opinions"
}

CUSTOM_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

def get_file_info(filename):
    filename_lower = filename.lower()
    found_llm = "Unknown"
    
    matches = []
    for pattern, display_name in LLM_MAPPING.items():
        for match in re.finditer(re.escape(pattern), filename_lower):
            matches.append({
                'start': match.start(),
                'len': len(pattern),
                'name': display_name
            })
    
    if matches:
        matches.sort(key=lambda x: (x['start'], x['len']), reverse=True)
        found_llm = matches[0]['name']

    weight = 1
    match_weight = re.search(r'\[(\d+)\]', filename)
    if match_weight:
        weight = int(match_weight.group(1))
        
    return found_llm, weight

def get_valid_scores(text):
    pattern = r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]"
    matches = re.findall(pattern, text)
    
    for match in reversed(matches):
        try:
            full_scores = [float(n) for n in match]
            if sum(full_scores) == 0:
                continue 
            if all(1 <= s <= 100 for s in full_scores):
                return full_scores[1:] 
        except ValueError:
            continue
    return []

def calculate_weighted_avg(df):
    if df.empty or df['Weight'].sum() == 0:
        return 0
    weighted_sum = (df['Score'] * df['Weight']).sum()
    total_weights = df['Weight'].sum()
    return weighted_sum / total_weights

def plot_global_results(df):
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    df_global = df.groupby('LLM', group_keys=False).apply(calculate_weighted_avg).reset_index(name='Final_Score')
    df_global = df_global.sort_values('Final_Score', ascending=False)
    
    file_counts = df.groupby('LLM')['Source_File'].nunique()
    print("\n--- Files successfully included per Model ---")
    print(file_counts)
    
    ax = sns.barplot(
        data=df_global, x='LLM', y='Final_Score', hue='LLM',
        palette=CUSTOM_PALETTE, edgecolor='black', legend=False
    )
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=5, fontsize=12, fontweight='bold')

    plt.title('AVERAGE GLOBAL SCORE (Out of 100)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=13)
    plt.ylabel('Average Score (/100)', fontsize=13)
    plt.ylim(0, 105)
    plt.tight_layout()
    
    save_path = os.path.join(REPORT_FOLDER, "global_scores.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 Saved global chart to: {save_path}")
    plt.show()

def plot_criteria_results(df):
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    df_crit = df.groupby(['Criteria_Index', 'LLM'], group_keys=False).apply(calculate_weighted_avg).reset_index(name='Final_Score')
    
    sns.barplot(
        data=df_crit, x='Criteria_Index', y='Final_Score', hue='LLM',
        palette=CUSTOM_PALETTE, edgecolor='black'
    )
    
    plt.title('Performance by Category', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Evaluation Categories', fontsize=13)
    plt.ylabel('Average Model Score (/100)', fontsize=13)
    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Models")
    plt.tight_layout()
    
    save_path = os.path.join(REPORT_FOLDER, "category_scores.png")
    plt.savefig(save_path, dpi=300)
    print(f"📊 Saved category chart to: {save_path}")
    plt.show()

# =======================
# EXECUTION SCRIPT
# =======================

def generate_reports():
    data = []
    
    if not os.path.exists(ROOT_FOLDER):
        print(f"ERROR: Folder '{ROOT_FOLDER}' not found.")
        print(f"Please create '{ROOT_FOLDER}' and add your subject folders inside.")
        return

    print(f"--- Analyzing folder: {ROOT_FOLDER} ---")

    for root, dirs, files in os.walk(ROOT_FOLDER):
        category = os.path.basename(root)
        if root == ROOT_FOLDER: continue
            
        for file in files:
            if not file.endswith(".txt"): continue
                
            path = os.path.join(root, file)
            llm, weight = get_file_info(file)
            
            if llm == "Unknown":
                continue
                
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                valid_scores = get_valid_scores(content)
                
                if len(valid_scores) == 4:
                    for i, score in enumerate(valid_scores, start=1):
                        data.append({
                            'Source_File': file,
                            'Category': category,
                            'Criteria_Index': CRITERIA_NAMES.get(i, f"Criterion {i}"),
                            'LLM': llm,
                            'Score': score,
                            'Weight': weight
                        })
                
            except Exception as e:
                print(f"  [ERROR] {file}: {e}")

    df = pd.DataFrame(data)
    
    if df.empty:
        print("\n❌ No valid data found.")
        print("Ensure files have a bracket like [80, 90, 85, 90, 95] (1-100 range).")
        return

    print(f"\n✅ Success: {len(df)} individual scores extracted.")
    
    plot_global_results(df)
    plot_criteria_results(df)


if __name__ == "__main__":
    # --- 1. Run Pipeline ---
    base_dir = Path(r"C:\Users\julia\OneDrive - Technion\Documents\nlp-project\code")
    reco_prompt_path = base_dir / "prompts" / "reco_prompts" / "pp_reco_prompt.txt"
    eval_prompt_path = base_dir / "prompts" / "eval_prompt" / "eval_prompt.txt"
    
    try:
        reco_prompt = load_text(reco_prompt_path)
        eval_prompt = load_text(eval_prompt_path)
        
        pipe = LLMPipeline(base_dir=base_dir, reco_prompt=reco_prompt, eval_prompt=eval_prompt)
        models_names = ["claude-haiku-4-5", "gemini-3-flash-preview", "gemini-3-pro-preview", "claude-sonnet-4-5", "claude-opus-4-6"]
        
        pipe.eval(pipe.reco(models_names), "gemini-3-pro-preview", models_names)
    except FileNotFoundError as e:
        print(f"File loading error during pipeline execution: {e}")
        print("Skipping LLM calls. Proceeding to report generation if data exists.")

    # --- 2. Generate Reports ---
    generate_reports()
