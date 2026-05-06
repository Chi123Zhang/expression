# ENVIO LLM Coding Assistant

A retrieval-grounded qualitative analysis framework for transcript coding, implementation-science synthesis, and stakeholder-adaptive evidence organization using Large Language Models (LLMs).

---

# Overview

The ENVIO LLM Coding Assistant is a research-oriented qualitative analysis system developed within the AI for Social Good and Society (AI4SGS) and ENVIO research environment at Columbia University.

The project investigates how retrieval-grounded LLM systems can support qualitative social science workflows through:

- human-aligned transcript coding
- contextual retrieval augmentation (RAG)
- structured evaluation
- cross-interview thematic synthesis
- stakeholder-adaptive explanation generation

Rather than functioning as a simple GPT wrapper or transcript summarizer, the framework operationalizes qualitative coding into a reproducible and context-aware workflow.

The system was designed for implementation-science and public health transcript analysis involving themes such as:

- healthcare access
- social support
- stigma
- staffing barriers
- organizational coordination
- correctional health workflows
- substance use intervention

The broader goal of the project is not to replace qualitative researchers, but to support evidence organization, coding consistency, and scalable thematic synthesis while preserving transcript grounding.

---

# Research Context

This project was developed within the Social Intervention Group (SIG) and AI for Social Good and Society (AI4SGS) research environment at Columbia University.

The interview materials used in this project were derived from de-identified ENVIO qualitative interview data associated with implementation-science and public health workflow analysis.

The framework is intended to support:

- qualitative transcript analysis
- implementation-science research
- HIV and public health intervention studies
- coding reproducibility analysis
- LLM-human alignment research
- adaptive evidence synthesis

The system is designed for research support only and is not intended for participant-level decision-making or automated intervention assignment.

---

# Main Contributions

## Human-Referenced Qualitative Coding

The system aligns LLM-generated coding outputs with manually coded transcript segments for structured evaluation.

Supported evaluation metrics include:

- Macro F1
- Cohen’s Kappa
- category-level agreement
- RAG vs No-RAG comparison

---

## Retrieval-Grounded Coding Refinement

Instead of coding isolated transcript segments independently, the framework performs retrieval-grounded refinement using contextual transcript windows.

The latest production pipeline uses:

```text
previous batch
+ current batch
+ next batch
```

to improve semantic grounding for contextually ambiguous categories.

---

## Batch-Consistent Coding

The production pipeline performs coding in transcript batches rather than isolated sentence-level prompting.

Current configuration:

```python
BATCH_SIZE = 5
```

This improves:

- coding consistency
- semantic continuity
- transcript coherence
- contextual stability

---

## Stakeholder-Adaptive Evidence Synthesis

The framework includes a stakeholder-adaptive synthesis layer capable of generating different implementation-oriented interpretations using the same retrieved evidence.

Supported perspectives currently include:

- General / Researcher
- Program Manager
- Engineer
- Business-oriented

The synthesis layer modifies:

- explanation structure
- implementation framing
- workflow emphasis
- organizational abstraction

while preserving the same underlying transcript evidence.

This transforms the system from a simple coding pipeline into a retrieval-grounded qualitative reasoning framework.

---

# System Architecture

```text
Transcript Files
        ↓
Chunking / Human Alignment
        ↓
Batch LLM Coding
        ↓
BM25 + Dense Retrieval
        ↓
Cross-Encoder Reranking
        ↓
Contextual RAG Refinement
        ↓
Structured Outputs
   ├── Human Comparison
   ├── Thematic Synthesis
   └── Stakeholder-Adaptive Outputs
           ├── General
           ├── Program Manager
           └── Engineer
```

The framework separates:

- retrieval,
- coding,
- and explanation synthesis

into distinct stages.

Retrieval preserves transcript grounding, while the synthesis layer adapts organization and interpretation to stakeholder needs.

---

# Core Features

- Human-aligned transcript segmentation
- Batch qualitative coding
- Retrieval-grounded refinement
- BM25 + dense retrieval
- Cross-encoder reranking
- RAG vs No-RAG comparison
- Human agreement evaluation
- Multi-interview thematic synthesis
- Stakeholder-adaptive explanation generation
- Topic modeling (LDA)
- CSV / JSON / ZIP export
- Heatmap generation
- Streamlit interactive interface

---

# Retrieval Architecture

The retrieval pipeline combines:

- BM25 lexical retrieval
- dense semantic retrieval
- cross-encoder reranking

to support transcript-local contextual reasoning.

---

## Dense Retrieval

Semantic retrieval uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

with FAISS indexing.

This allows the framework to identify semantically related transcript regions even when participants use different wording.

---

## Cross-Encoder Reranking

Candidate retrieval windows are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

before refinement.

This improves retrieval precision and contextual relevance.

---

## Retrieval Parameters

| Parameter | Value |
|---|---|
| Chunk size | 220 words |
| Chunk overlap | 50 words |
| BM25 top-k | 12 |
| Dense retrieval top-k | 12 |
| Final reranked top-k | 8 |

---

# Experimental Results

## Benchmark Evaluation

Formal quantitative evaluation was conducted on a manually aligned benchmark interview containing 79 human-coded transcript segments.

The evaluation compared:

- baseline No-RAG coding
- retrieval-grounded refinement
- human annotation agreement

---

## Macro-Level Results

| Setting | Macro F1 | Macro Cohen’s Kappa |
|---|---|---|
| No-RAG | 0.5446 | 0.4585 |
| Retrieval-Grounded RAG | 0.5961 | 0.5090 |

---

## Improvements from Retrieval Grounding

| Metric | Improvement |
|---|---|
| Macro F1 | +0.0515 |
| Macro Cohen’s Kappa | +0.0505 |

The improvements were modest but directionally consistent across both metrics.

---

## Per-Category Agreement Results

| Code | Cohen’s Kappa | F1 | Human Positive | LLM Positive |
|---|---|---|---|---|
| environmental_barrier | 0.3517 | 0.4167 | 19 | 5 |
| healthcare_access | 0.3724 | 0.5660 | 19 | 34 |
| mental_health | 0.5741 | 0.6000 | 6 | 4 |
| social_support | 0.4034 | 0.5405 | 21 | 16 |
| stigma | 0.8433 | 0.8571 | 7 | 7 |

---

# Interpretation of Results

The results suggest that retrieval grounding is particularly useful for semantically distributed qualitative categories.

The strongest retrieval-related improvements were observed for:

- healthcare_access
- social_support
- environmental_barrier

These categories often depended on contextual interpretation across neighboring transcript segments rather than isolated sentence-level semantics.

The highest agreement was observed for:

- stigma
- mental_health

suggesting that explicit semantic categories are easier for both human coders and LLM-based systems to identify consistently.

Importantly, the project does not claim that retrieval universally improves qualitative coding.

Instead, the findings suggest that:

- human-aligned segmentation,
- contextual retrieval,
- batch-consistent coding,
- and structured evaluation

collectively improve the reproducibility and interpretability of LLM-assisted qualitative workflows.

---

# Multi-Interview Thematic Synthesis

Beyond benchmark evaluation, the framework was extended to support retrieval-grounded thematic synthesis across multiple implementation-science interviews.

The synthesis corpus included:

- `002S Qual Interview Including Costing 2025-06-16.pdf`
- `003S Officer and Progr Dir Qual Interview 2025-05-14 Part 1.pdf`
- `003S Qual Interview 2025-05-14 Part 2.pdf`
- `005S Qual Interview 2025-05-14 Part 1.pdf`
- `005S Qual Interview 2025-05-14 Part 2.pdf`

These files were used for:

- cross-interview retrieval
- thematic aggregation
- implementation-science synthesis
- stakeholder-adaptive explanation generation

They were not included in quantitative benchmark metric computation.

---

# Representative Themes Identified

The retrieval-grounded synthesis pipeline identified recurring themes involving:

- healthcare access
- staffing limitations
- organizational barriers
- telehealth
- transportation constraints
- support systems
- educational resources
- stigma reduction

The multi-interview corpus substantially improved:

- thematic diversity
- retrieval realism
- implementation-science coverage
- workflow complexity

relative to single-transcript evaluation.

---

# Recommended Workflow

## Step 1 — Upload Transcripts

Upload one or more qualitative interview transcripts in:

- PDF
- DOCX
- TXT

format.

Optional:

- upload human-coded CSV for benchmark evaluation.

---

## Step 2 — Run Initial Coding

The system performs:

- transcript segmentation
- batch coding
- initial qualitative labeling

Outputs include:

- coded CSVs
- coding summaries
- JSON outputs

---

## Step 3 — Retrieval-Grounded Refinement

Enable contextual retrieval refinement.

The system retrieves neighboring transcript evidence and performs contextual code revision.

This stage improves semantic grounding and transcript continuity.

---

## Step 4 — Human Comparison Evaluation

If a human-coded CSV is uploaded, the framework computes:

- Macro F1
- Cohen’s Kappa
- category-level agreement
- RAG vs No-RAG comparison

Outputs include:

- evaluation CSVs
- heatmaps
- ZIP exports

---

## Step 5 — Multi-Interview Synthesis

Merge multiple transcript files to perform:

- thematic aggregation
- implementation-science synthesis
- cross-interview retrieval

This stage is designed for exploratory qualitative analysis rather than benchmark evaluation.

---

## Step 6 — Stakeholder-Adaptive Synthesis

Generate implementation-oriented summaries for different stakeholder perspectives:

- General
- Program Manager
- Engineer
- Business-oriented

The same evidence is reorganized differently depending on operational context and interpretive needs.

---

# Repository Structure

```text
app.py
Main Streamlit orchestration layer.

rag_system.py
Retrieval pipeline and contextual refinement logic.

llm_batch_coding.py
Batch coding and evaluation pipeline.

expression_layer.py
Stakeholder-adaptive synthesis generation.

query_orchestrator.py
Workflow routing and orchestration.

background_memory.py
Profile and adaptive reasoning support.

outputs/
Generated CSVs, JSON outputs, heatmaps, ZIP exports.

data/
Transcript and benchmark storage.
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

---

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
2. Add Streamlit Secrets:
   - OPENAI_API_KEY
   - HF_TOKEN
3. Deploy `app.py`

---

# Topic Modeling

The framework includes an optional LDA topic modeling pipeline for exploratory thematic analysis.

Representative themes identified include:

| Topic | Representative Terms |
|---|---|
| Topic 1 | access, telehealth, help |
| Topic 2 | staffing, barriers, supports |
| Topic 3 | grants, reduction, help |
| Topic 4 | educational, resources |
| Topic 5 | program, services, access |

These topics aligned closely with the manually defined qualitative coding categories.

---

# Current Limitations

Current limitations include:

- relatively small benchmark evaluation size
- absence of multi-annotator human agreement analysis
- retrieval sensitivity to transcript quality
- stochastic LLM behavior
- latency for large transcript collections
- Streamlit timeout risk during large runs
- qualitative ambiguity that remains difficult for both humans and models

The stakeholder-adaptive synthesis layer was evaluated qualitatively rather than through controlled user studies.

---

# Future Directions

Potential future work includes:

- larger multi-annotator benchmarks
- adaptive retrieval routing
- uncertainty-aware coding
- hierarchical codebooks
- selective refinement
- longitudinal transcript analysis
- LoRA fine-tuning
- richer implementation-science datasets
- formal usability evaluation
- stakeholder-specific workflow optimization

---

# Live Demo

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

This project was developed within the ENVIO / AI4SGS research environment focused on AI-assisted qualitative analysis, implementation-science workflows, and human-centered retrieval-grounded reasoning systems.
