# 📜 NLP Benchmark: Evaluating LLMs on Complex Jewish Legal Texts

Welcome to the **NLP Benchmark for Jewish Legal Texts**! This project provides a rigorous, automated framework to evaluate the performance of leading Large Language Models (LLMs) in understanding, analyzing, and synthesizing complex traditional Jewish texts (such as Halachic rulings, Talmudic debates, and responsa). 

Understanding these texts requires more than just translation; it demands deep contextual awareness, logical tracking of multi-tier debates, and familiarity with unique terminology. This benchmark tests whether modern AI can rise to the challenge. 🧠💡

## ✨ Project Strengths & Academic Value

While LLMs excel at standard NLP tasks, ancient and medieval legal texts present a unique frontier:
* **High Density & Implicit Context:** Rabbinic literature often relies on deeply implicit knowledge and dense logical structures.
* **Complex Debates:** Texts frequently weave between multiple distinct opinions, hypothetical scenarios, and conditional rulings.
* **A Rigorous Testing Ground:** By evaluating how well an LLM extracts precise *categories* and *chains of transmission* from these texts, we push the boundaries of current AI reasoning, recall, and precision capabilities.

---

## 🎯 The Task: What We Ask the LLMs to Do

We task the LLMs with reading raw text from classical Jewish legal codes (specifically the *Tur* and its commentary, the *Beit Yosef*) and breaking them down into structured, logical components. The models must identify the core question being debated, the different Halachic categories (opinions/rulings), and the exact chain of transmission (who said what, in whose name) alongside the relevant quotation.

### 📝 Example: Rejoicing on a Festival (שמחת יום טוב)

To understand the complexity, let's look at a translated example from our dataset.

**The Raw Input (Origin Text):**
> **Tur:** Rambam wrote: A person is obligated to be happy and of good heart on a festival... and the men eat meat and drink wine...
> **Beit Yosef:** ...Regarding what our Rabbi wrote that men eat meat and drink wine, the Rambam concludes that there is no joy except with meat, and no joy except with wine. But in the Talmud... it is taught that nowadays when the Temple does not exist, there is no joy except with wine... Thus, one must wonder about the Rambam why he required both eating meat and drinking wine, since the Baraita implies that wine is sufficient without meat.

**The Expected Output (Ground Truth):**
The LLM is expected to parse this narrative flow into the following strict hierarchy:

* **# Question:** How do men rejoice on a festival nowadays?
    * **## Category 1:** By eating meat and drinking wine.
        * `- Tur >> Rambam:` "and the men eat meat and drink wine"
        * `- Beit Yosef >> Rambam:` "Rambam concludes that there is no joy except with meat, and no joy except with wine"
    * **## Category 2:** Drinking wine without meat is sufficient.
        * `- Beit Yosef:` "which implies that wine is sufficient without meat"

As you can see, the model must differentiate between what the *Tur* quotes from the *Rambam*, and the *Beit Yosef's* independent analytical conclusion based on the Talmud.

---

## ⚙️ How It Works: The Methodology

Our highly automated **"LLM-as-a-Judge" pipeline** executes a three-stage process to ensure fair and accurate evaluation:

### 1. Generation (Inference) 🤖
We feed the original Hebrew/Aramaic source texts into several state-of-the-art models (including Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3.7 Sonnet, Gemini 3 Pro, and Gemini 3 Flash). Using a standardized prompt, the models are tasked with extracting the core legal opinions, chains of transmission, and relevant quotes.

### 2. Evaluation (LLM-as-a-Judge) ⚖️
Manually grading complex textual analysis is highly subjective and time-consuming. Instead, we use an advanced reasoning model (e.g., `Gemini-3-Pro-Preview`) as an objective judge. The evaluator compares the generated response against a human-verified **Ground Truth (GT)**. 

The judge uses a principle of **Semantic Equivalence** (e.g., treating "forbidden" and "must not do" as identical) but demands **Strict Structural Accuracy** for the chains of transmission. 

The judge outputs a standardized array of 5 numeric scores `[Title, Cat_Precision, Cat_Recall, Op_Precision, Op_Recall]` based on the following rubric:

1.  🏷️ **Title Match (0-100):** A semantic similarity score comparing the model's generated question to the GT question.
2.  🎯 **Category Precision:** Did the model invent any halachic rulings? (True Positives / Total Categories Generated).
3.  🔄 **Category Recall:** Did the model find all the halachic rulings present in the text? (True Positives / Total Categories in GT).
4.  🎯 **Opinion (Chain) Precision:** Within the correct categories, did the model build the exact, accurate chain of rabbis (e.g., `Tur >> Beit Yosef`, not just `Tur`) without adding false links?
5.  🔄 **Opinion (Chain) Recall:** Did the model successfully identify every individual chain of transmission mapped in the GT?

### 3. Analysis & Reporting 📊
The pipeline automatically parses the evaluation files, extracts the 5-number arrays via regex, applies appropriate weighting based on the complexity of the text, and generates comprehensive data visualizations using `pandas` and `seaborn`.

---

## 📂 Directory Structure

Here is a breakdown of the main folders in this repository and their roles in the pipeline:

* **`/origins`** 📜
  Contains the raw, original source texts (Halachic texts from the Tur and Beit Yosef) that serve as the input for the LLMs.
  
* **`/reco_gt` (Ground Truth)** ✅
  Contains the meticulously crafted, human-verified "correct" parsing for each origin text. This is the gold standard used by the evaluator to grade the models.

* **`/prompts`** 📝
  Houses the instructional text templates used to guide the LLMs.
  * `reco_prompts/`: The few-shot instructions telling the models exactly how to read and break down the origins.
  * `eval_prompt/`: The strict rubric and mathematical instructions used by the "Judge LLM" to score the outputs.

* **`/recos` (Recommendations/Outputs)** 💬
  The raw generated outputs produced by each LLM (Claude, Gemini, etc.) after analyzing the origin files. 

* **`/evals` (Evaluations)** 💯
  The grading reports generated by the Judge LLM. Each file contains a detailed critique and the final `[5-number]` score array comparing a specific model's output to the ground truth.

* **`/report`** 📈
  The final output destination for our data visualization. Contains generated `.png` charts (like `global_scores.png` and `category_scores.png`) that provide a beautiful, at-a-glance understanding of which model reigns supreme.

---

## 🚀 Getting Started

### Prerequisites
To run this benchmark, you will need API keys for the respective models and a standard Python data science environment:
* `google-genai` (for Gemini models)
* `anthropic` (for Claude models)
* `pandas`, `seaborn`, `matplotlib` (for data processing and visualization)
* `python-dotenv` (for managing API keys)

### Running the Pipeline
Simply execute the Jupyter Notebook (`new.ipynb`). The script will automatically:
1. Load your API keys from a `.env` file.
2. Iterate through the `/origins`.
3. Call the target LLMs to generate the parsed analyses.
4. Call the Judge LLM to evaluate the results using the stringent scoring array.
5. Extract the scores and generate visual reports in the `/report` folder.

---
*Created with a passion for connecting deep architectural software structures with the timeless depth of Jewish texts.*