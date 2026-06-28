# **📜 NLP Benchmark: Evaluating LLMs on Complex Jewish Legal Texts**

## **Project Context**

This project was developed by **Tsuriel Vizel** and **Avraham Guez**, under the supervision of **Oren Mishali**. It serves as an academic exploration into the capabilities of modern Artificial Intelligence when tasked with processing highly complex, dense, and historically significant literature.

## **📖 Introduction & Academic Motivation**

This project evaluates the ability of Large Language Models (LLMs) to process complex Rabbinic legal texts. We focus on the Beit Yosef, a foundational 16th-century halachic commentary, using a structured benchmark to measure model accuracy in extracting legal debates.

Rationale: To improve LLM performance on non-standard, dense structures, a Ground Truth (GT) dataset is required to quantify errors and guide iterative refinement of extraction prompts.

Specifically, this benchmark focuses on the **Beit Yosef** (authored by Rabbi Yosef Karo), a foundational 16th-century commentary on the Tur. The Beit Yosef is characterized by its high textual density, heavily implicit context, and intricate mapping of multi-tier halachic (Jewish legal) debates. It traces the development of practical law from Talmudic origins through medieval commentators (Rishonim), constantly weaving between distinct opinions, conditional rulings, and hypothetical scenarios.

### **Why extract debates (מחלוקות)?**

Automated extraction of halachic debates is a core challenge in Digital Humanities. Unlike standard summarization, it requires identifying multi-tier legal arguments, mapping conflicting opinions, and tracking conditional rulings within dense, unpunctuated text. Successfully automating the extraction of these debates provides substantial value for the field of Digital Humanities: it facilitates the digitization and structural mapping of centuries of complex legal reasoning.

From an AI and Natural Language Processing perspective, it provides a rigorous evaluation environment to push the boundaries of an LLM's reasoning, context-retention, and information-extraction capabilities across specialized, non-standard text structures.

## **🏗️ Project Overview**

The project is built upon three main pillars:

1. **The Benchmark Dataset:** A curated collection of original textual excerpts alongside an expert-annotated "Ground Truth" (GT) parsing for each.  
2. **LLM Execution Pipeline:** A systematic process to prompt various state-of-the-art models (such as Claude Sonnet 3.7/4.5 , Haiku 4.5, Opus 4.6 , Gemini 3 Flash/Pro, etc.) to extract and format the debates from the raw texts.  
3. **LLM-as-a-Judge Evaluation:** An automated grading system where an advanced LLM compares the generated outputs against our Ground Truth using a strict, multi-dimensional rubric.

## **📊 The Benchmark: Structure & Methodology**

To effectively measure LLM performance, we constructed a dedicated benchmark dataset comprising **31 distinct halachic debates** (extracted from 18 original source files).

* **The Source Files:** Located in the [./origins](./origins) directory. These are raw .txt files containing the original Hebrew texts of the Tur and Beit Yosef. (Note: Many files contain multiple distinct debates, indicated by the bracketed number in the filename, e.g., \[4\]).  
* **The Ground Truth (GT):** Located in the [./reco_gt](./reco_gt) directory. These .txt files contain the expected output, structured in a hierarchical Markdown format to allow for standardized parsing and evaluation.

### **Annotation Methodology**

The Ground Truth was established by the researchers utilizing domain expertise in Rabbinic literature. For each source text, we analyzed the halachic discourse, identified the core subject of the debate, categorized the different approaches (e.g., "מחמירים", "מקילין"), and formally mapped each specific Rabbinic authority (e.g., "הרמב"ם", "רש"י") to their respective category. This process creates an objective, expert-annotated gold standard against which the LLMs are measured.

## **🔍 Illustrative Example**

To understand the complexity of the task, let us examine a brief example of how a text is processed into a structured debate format.

### **1\. The Source Text (Origin)**

Consider a text discussing the laws of Yom Tov (Festivals):

**"ומשמע מדברי הרמב"ם דאפילו ביום טוב שני אסור... אבל רש"י התיר אפילו ביום טוב ראשון... והרא"ש פסק כרש"י אבל רק ליום טוב שני."**

### **2\. Explanation of the Debate**

In this short excerpt, there is a clear disagreement regarding a specific action on Yom Tov.

* **Maimonides (הרמב"ם)** is stringent, forbidding the action entirely on both the first and second days of the festival.  
* **Rashi (רש"י)** is lenient, permitting it even on the first day.  
* **The Rosh (הרא"ש)** takes a middle ground, agreeing with Rashi's leniency, but restricting it only to the second day.

### **3\. The Expected Output (Ground Truth Format)**

The LLM is tasked with reading the raw text and outputting a structured mapping identical to the format below. This specific format enables our evaluation script to parse and grade the results programmatically.

# Title
מחלוקת לגבי איסור הפעולה ביום טוב

## Categories
### מחמירים לגמרי
* הרמב"ם

### מקילין לגמרי
* רש"י

### מקילין חלקית (רק ביום טוב שני)
* הרא"ש

## **⚖️ The Grading Rubric (LLM-as-a-Judge)**

Evaluating generated text against a Ground Truth is notoriously difficult. Traditional metrics like BLEU or ROUGE fail to capture logical accuracy. Therefore, we utilize an **LLM-as-a-Judge** approach.

A high-tier model acts as the evaluator, comparing the tested model's output to the Ground Truth across 5 distinct mathematical criteria (scored from 0.0 to 1.0). **Note:** These metric names correspond exactly to the evaluation logic and the generated visual reports:

1. **Title**: Accuracy of identifying the core topic/subject of the debate. (This part was really welled identifyed by almost all models so we didn't take it into account for the final grades.) 
2. **Precisions of categories**: Did the model invent categories that do not exist in the Ground Truth? (Few hallucinations \= High Precision).  
3. **Recall of categories**: Did the model successfully find all the categories present in the Ground Truth? (Few misses \= High Recall).  
4. **Precisions of opinions**: Did the model assign Rabbis/Opinions to the wrong categories or invent assignments?  
5. **Recall of opinions**: Did the model correctly place all the Rabbis/Opinions mentioned in the Ground Truth into their appropriate categories?

## **📈 Interpretation of Findings**

Through our visual reports (available in the [./report](./report) directory), we observe distinct variations in model capabilities:

* Models demonstrate strong capabilities in identifying the overall Title of the debate.  
* Extracting specific opinions yields mixed results depending on the model's architectural generation style and context window.  
* **Text Preparation Impact:** We found that modern models perform significantly better when dealing with "raw, unedited text chunks" rather than heavily pre-processed or segmented texts. Providing the LLM with the organic flow of the Beit Yosef allows its internal attention mechanisms to naturally track the evolving context and implicit references (e.g., "And he said..."), which are often lost if the text is artificially divided beforehand.
Further evaluation demonstrated that segmenting the essays into **smaller, thematic subsections** significantly **enhances** the detection performance of Large Language Models (LLMs). Consequently, this preprocessing technique yielded the superior results among the various methods tested for improving model comprehension.

## **📂 Directory Structure**

* [./origins](./origins): The raw, original source texts (Tur and Beit Yosef).
* [./sep_origins](./sep_origins): The separated in small parts source texts (Tur and Beit Yosef).
* [./pp_origins](./pp_origins): The pre-processed source texts (Tur and Beit Yosef).
* [./reco_gt](./reco_gt): The human-verified, expert-annotated correct parsing for each origin text.
* [./prompts](./prompts): The instructional text templates used to guide the LLMs.
* [./recos](./recos): The raw generated outputs produced by each LLM for original source texts.
* [./sep_recos](./sep_recos): The raw generated outputs produced by each LLM for separated source texts.
* [./pp_recos](./pp_recos): The raw generated outputs produced by each LLM for pre-processed source texts.
* [./evals](./evals): Grading reports generated by the Judge LLM for the original source texts.
* [./sep_evals](./sep_evals): Grading reports generated by the Judge LLM for the separated source texts.
* [./pp_evals](./pp_evals): Grading reports generated by the Judge LLM for the pre-processed source texts.
* [./report](./report): Generated data visualizations (.png charts).
* [./result](./result): Final comparative analysis of results categorized by text processing methodology.

## **🚀 Getting Started**

### **Prerequisites**

To run this benchmark, you will need API keys for the respective models and a standard Python data science environment:

* google-genai (for Gemini models)  
* anthropic (for Claude models)  
* pandas, seaborn, matplotlib (for data processing and visualization)  
* python-dotenv (for managing API keys)

### **Running the Pipeline**

Execute the provided Python Script (nlp_project.py). The script will automatically:

1. Load API keys from your .env file.  
2. Iterate through the /origins dataset.  
3. Prompt the target LLMs to generate parsed analyses.  
4. Call the Judge LLM to strictly evaluate the results against the GT.  
5. Parse the scores and output visual charts to the /report folder.
