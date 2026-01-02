# Clinical RAG System with Safety & Bias Controls

A **production-oriented Retrieval-Augmented Generation (RAG) system** designed for **clinical guideline analysis**, with strong emphasis on **hallucination prevention, evidence grounding, age-bias detection, and CI/CD safety testing**.

Built using **WHO clinical guidelines** as the knowledge source.

---

## Why This Project?

Most RAG demos:
- blindly trust LLM outputs
- hallucinate medical facts
- lack safety & bias checks
- have no automated tests

This project **solves those problems** by design.

---
## Key Features

### Hallucination-Resistant Design
- LLM **never sees the full document**
- Answers generated **only from extracted evidence**
- Safe fallback: *"Insufficient evidence"* when needed

###  Responsible AI & Bias Detection
- Detects **age-deterministic or stigmatizing language**
- **Negation-aware bias logic** (e.g. “not inevitable” is allowed)
- Transparent bias flags instead of silent filtering

###  Offline & Production-Friendly
- Local embeddings (CPU)
- Local LLM (no API rate limits)
- Privacy-preserving (no external calls)

###  CI/CD & Testing
Automated pytest suite verifies:
- retrieval relevance
- evidence-only answers
- bias safety
- graceful failure on unrelated queries

---

##  Example Query

**Question:**  
> What are the risk factors for cognitive decline?

**Output Includes:**
- Evidence-backed answer
- Chunk-level citations
- Confidence score
- Bias safety check result

---

## Tech Stack

- SentenceTransformers (MiniLM)
- Ollama (local LLM – Phi-3)
- PyPDF
- NumPy
- Pytest
- LangChain text splitters

---
##  What This Demonstrates

- Real-world RAG system design
- Safety-first AI thinking
- Clinical/healthcare AI awareness
- CI/CD & test-driven development
- Responsible AI practices

---

## Build by 

**Kriti Kashyap**  
Computer Science Student   

