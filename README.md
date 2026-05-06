# ENVIO LLM Coding Assistant

A context-aware, human-aligned qualitative coding framework using Large Language Models (LLMs) for transcript analysis, retrieval-augmented reasoning, and adaptive expression.

---

# Overview

The ENVIO LLM Coding Assistant is a research-oriented qualitative coding system developed as part of the ENVIO / AI4SGS initiative at Columbia University.

The project investigates how Large Language Models (LLMs) can support qualitative social science workflows through:

- human-aligned transcript coding
- contextual retrieval augmentation (RAG)
- adaptive qualitative reasoning
- topic modeling
- structured evaluation
- background-aware expression generation

Unlike traditional GPT prompting pipelines, this system provides:

- reproducible qualitative coding
- controlled RAG vs No-RAG evaluation
- contextual reasoning windows
- batch-consistent coding
- human-comparison metrics
- adaptive explanation layers

---

# Research Context

This project was developed within the Social Intervention Group (SIG) and AI for Social Good and Society (AI4SGS) research environment at Columbia University.

The system is designed to support:

- qualitative transcript analysis
- HIV/social intervention research
- coding consistency studies
- LLM-human alignment research
- workflow efficiency exploration
- adaptive AI communication systems

---

# Main Contributions

## Human-Aligned Qualitative Coding

The system aligns LLM-generated coding outputs with human-coded transcript segments for structured evaluation.

Supported metrics include:

- Macro F1
- Cohen’s Kappa
- category-level agreement
- RAG vs No-RAG comparison

---

## Contextual Retrieval-Augmented Reasoning

Instead of coding isolated sentences independently, the system performs:

```text
previous context
+ current segment
+ next context
```

retrieval-aware qualitative reasoning.

This substantially improves semantic grounding for ambiguous qualitative categories.

---

## Batch-Consistent Coding

The latest production pipeline uses:

```python
BATCH_SIZE = 5
```

to improve:

- coding consistency
- semantic continuity
- transcript coherence
- retrieval efficiency

---

## Background-Aware Expression Layer

The system includes an adaptive expression layer capable of generating different explanation styles depending on user background.

Supported perspectives include:

- General
- Engineering
- Business
- PM / Workflow-oriented

This transforms the project from a simple coding assistant into a context-aware qualitative reasoning framework.

---

# System Architecture

```text
Transcript / PDF
↓
Human-Aligned Segmentation
↓
Batch Coding
↓
Contextual RAG Retrieval
↓
RAG Refinement
↓
Structured Output Generation
↓
Human Comparison
↓
Evaluation Metrics + Heatmaps + Reports
```

---

# Core Features

- Human CSV alignment
- Batch qualitative coding
- Contextual retrieval windows
- RAG vs No-RAG comparison
- Topic modeling (LDA)
- Adaptive expression generation
- PDF / CSV / ZIP export
- Human agreement heatmaps
- Streamlit interactive UI
- Profile-aware explanation styles

---

# Experimental Results

## Macro-Level Evaluation

| Setting | Macro F1 | Macro Cohen’s Kappa |
|---|---|---|
| No-RAG | 0.5496 | 0.4620 |
| Contextual RAG | 0.6204 | 0.5408 |

## Improvements from Contextual RAG

| Metric | Improvement |
|---|---|
| Macro F1 | +0.0708 |
| Macro Cohen’s Kappa | +0.0788 |

---

# Per-Code Agreement Results

| Code | Cohen’s Kappa | F1 |
|---|---|---|
| environmental_barrier | 0.3517 | 0.4167 |
| healthcare_access | 0.3925 | 0.5714 |
| mental_health | 0.7070 | 0.7273 |
| social_support | 0.4093 | 0.5294 |
| stigma | 0.8433 | 0.8571 |

---

# Key Findings

The experimental results suggest:

- contextual retrieval substantially improves qualitative reasoning
- transcript-local semantic context is highly important
- human-aligned segmentation improves evaluation validity
- batch-consistent coding stabilizes qualitative outputs
- explicit categories achieve higher agreement
- semantically distributed categories benefit most from RAG

The strongest contextual improvements were observed for:

- healthcare_access
- social_support
- environmental_barrier

The highest human agreement was observed for:

- stigma
- mental_health

---

# Why This Project Is Different from Plain GPT

A standard GPT prompt may generate labels for transcript segments, but it does not provide:

- stable segmentation
- structured evaluation
- reproducibility
- contextual retrieval refinement
- batch consistency
- human comparison
- adaptive communication
- systematic experimentation

This project operationalizes LLM-assisted qualitative coding into a reproducible research framework.

---

# Repository Structure

```text
app.py
Main Streamlit application.

rag_system.py
Retrieval-Augmented Generation pipeline.

llm_batch_coding.py
Batch coding + contextual refinement logic.

background_memory.py
User profile and memory handling.

expression_layer.py
Adaptive expression generation.

query_orchestrator.py
Routing and orchestration layer.

requirements.txt
Dependencies.

outputs/
Generated CSVs, heatmaps, JSON outputs, ZIP files.
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Chi123Zhang/ENVIO-LLM-Coding-Assistant

cd ENVIO-LLM-Coding-Assistant
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

## Mac / Linux

```bash
export OPENAI_API_KEY=your_api_key

export HF_TOKEN=your_huggingface_token
```

## Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your_api_key"

$env:HF_TOKEN="your_huggingface_token"
```

---

# Run Locally

```bash
streamlit run app.py
```

---

# Streamlit Cloud Deployment

1. Connect GitHub repository
2. Set secrets:
   - OPENAI_API_KEY
   - HF_TOKEN
3. Deploy app.py

---

# Recommended Workflow

## Step 1 — Basic Transcript Coding

Upload transcript and run qualitative coding.

Outputs:
- coded CSV
- topic modeling
- frequency tables

---

## Step 2 — Instruction-Aware Coding

Add coding instructions or research questions.

Example:

```text
Focus on barriers to healthcare access and social support.
```

This enables guided contextual reasoning.

---

## Step 3 — Background-Aware Expression

Select different explanation styles:

- General
- Engineering
- Business
- PM

The same underlying reasoning is expressed differently depending on audience context.

---

## Step 4 — Human Comparison Evaluation

Upload human-coded CSV.

The system computes:

- Macro F1
- Cohen’s Kappa
- code-level agreement
- heatmaps
- RAG vs No-RAG comparison

---

# Example Outputs

The system automatically generates:

- coded transcript CSVs
- evaluation CSVs
- heatmaps
- JSON outputs
- PDF reports
- ZIP export packages

---

# Topic Modeling

The LDA topic modeling pipeline identifies recurring semantic structures including:

- healthcare access
- institutional barriers
- educational resources
- stigma reduction
- support systems

Representative topic keywords include:

| Topic | Representative Terms |
|---|---|
| Topic 1 | access, telehealth, help |
| Topic 2 | staffing, barriers, supports |
| Topic 3 | grants, reduction, help |
| Topic 4 | educational, resources |
| Topic 5 | program, services, access |

---

# Current Limitations

Current known limitations include:

- long transcript latency
- Streamlit timeout risks
- retrieval sensitivity
- stochastic LLM outputs
- qualitative ambiguity
- limited fine-tuning data

---

# Future Directions

Potential future work includes:

- adaptive retrieval routing
- hierarchical coding
- semantic uncertainty estimation
- selective RAG refinement
- LoRA fine-tuning
- multi-document reasoning
- richer profile inference
- reinforcement-guided refinement
- longitudinal transcript analysis

---

# Research Interpretation

The main contribution of this project is not simply that retrieval improves coding.

Instead, the project demonstrates that:

- human-aligned segmentation
- contextual retrieval windows
- batch-consistent coding
- structured evaluation
- adaptive expression framing

collectively improve the reproducibility and reliability of LLM-assisted qualitative coding workflows.

The system therefore functions as a context-aware qualitative reasoning framework rather than a simple GPT wrapper.

---

# Live Demo

Streamlit App:

```text
https://expression-4wfrtnmolvi6ksdsst2h3c.streamlit.app/
```

---

# Author

## Chi (Charlie) Zhang

M.A. Statistics (Advanced Machine Learning Track)  
Columbia University

Graduate Research Assistant  
AI for Social Good and Society (AI4SGS)  
Social Intervention Group (SIG)

---

# Acknowledgements

This project was developed as part of the ENVIO / AI4SGS research initiative focused on AI-assisted qualitative reasoning and social science workflow support.
