## Research Experience

**Research Assistant, Social Intervention Group (SIG), Columbia University**  
*2026 – Present*

- Contributed to the ENVIO project, which evaluates the effectiveness of large language models (LLMs) in coding de-identified qualitative interview transcripts compared to human coders  
- Explored LLM-based qualitative coding, topic modeling, and workflow efficiency for HIV-related social science research  
- Developed a background-aware RAG (Retrieval-Augmented Generation) pipeline integrating structured user profiles, vector-based memory, and dual-context retrieval  
- Designed a query orchestration framework to dynamically combine user background, document context, and external knowledge for grounded and personalized generation  
- Improved model reliability by reducing hallucination and enforcing evidence-based responses through document grounding  
- Investigated evaluation strategies including response consistency, grounding quality, and alignment with human coding behavior

📊 ENVIO LLM Coding Assistant

A document-grounded system for automated qualitative coding, topic modeling, and LLM–human comparison in social science research.

🧠 Overview

This project implements an end-to-end pipeline for analyzing qualitative transcripts using Large Language Models (LLMs), designed for the ENVIO / TechMPower research context.

The goal is to explore whether LLMs can support or partially automate traditional qualitative coding workflows, which are often time-consuming and subject to human variability.

The system focuses on identifying themes such as:

Healthcare access
Social support
Stigma
Mental health
Environmental barriers

⚙️ Key Features

1. Multi-Source Input
Upload multiple files (PDF, DOCX, TXT)
Optional URL ingestion
Automatic merging of related documents into a single “participant”

2. LLM-Based Coding
Splits transcripts into meaningful segments
Assigns codes using a predefined codebook
Produces structured JSON and CSV outputs

3. Quantitative Analysis
Code frequency distributions
Grouped analysis (e.g., interview vs costing vs policy)
Participant-level comparison
Time/date-based trends

4. Visualization
Heatmaps (overall, grouped, participant-level)
Easy-to-interpret summary tables

5. Topic Modeling
LDA-based topic extraction
Optional BERTopic integration
Designed to complement qualitative coding

6. Report Generation
Automatically generates a PDF report using ReportLab
Includes:
Summary statistics
Code distributions
Topic modeling results
Sample coded segments

7. LLM vs Human Comparison (Optional)
Upload human-coded CSV
Compute:
F1 Score
Cohen’s Kappa
Visualize agreement with heatmaps

8. RAG Integration
Retrieval-Augmented Generation (RAG)
Incorporates background documents into coding context
Improves domain-specific interpretation

⚙️ Outputs
Each run generates:

*_coding.csv → segment-level coding results

batch_summary_coding.csv → merged dataset

lda_topics.csv → topic modeling results

llm_vs_human_comparison.csv → agreement metrics (if provided)

envio_coding_report.pdf → final report

.zip → all outputs bundled

⚙️ How to Run
1. Install dependencies
pip install -r requirements.txt

Make sure you include:

streamlit
openai
pandas
scikit-learn
matplotlib
seaborn
reportlab
beautifulsoup4
requests

2. Set OpenAI API key

export OPENAI_API_KEY=your_key_here

3. Run the app

- streamlit run app.py

- Example Workflow

Upload one or more transcripts

- Click Run

- View:

Coding outputs

Heatmaps

Topic modeling

- Download:

PDF report (recommended)

ZIP file (full reproducibility)



⚙️ Research Motivation

Traditional qualitative coding:

- Requires extensive manual effort

- Can vary across annotators

- Is difficult to scale

This system aims to:

- Reduce researcher workload

- Provide consistent coding structure

- Enable rapid exploratory analysis

- Support (not replace) human interpretation

⚙️ Limitations

- LLM outputs may vary depending on prompt and context

- Coding quality depends on codebook clarity

- Topic modeling is unsupervised and may require interpretation

- Human-coded ground truth is needed for rigorous evaluation

⚙️ Future Work
- Improved human–LLM agreement benchmarking

- Better domain adaptation via fine-tuning

- Enhanced multi-document reasoning

- Integration with qualitative research tools (e.g., NVivo-like workflows)

Author

Chi (Charlie) Zhang
M.A. in Statistics (Advanced Machine Learning)
Columbia University

Focus:

- Statistical learning

- Applied machine learning

- LLMs for qualitative research

📌 Summary

This project demonstrates how LLMs can be integrated into qualitative research pipelines to:

Automate coding
Generate structured insights
Support scalable analysis

while maintaining interpretability and alignment with traditional research practices.
