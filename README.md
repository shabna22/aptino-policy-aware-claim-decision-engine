# Policy-Aware Multi-Agent RAG Claim Decision Engine

An AI-powered health-insurance claim analysis system that uses hybrid RAG and a multi-agent workflow to make policy-grounded claim decisions.

## Overview

The system analyzes synthetic insurance claims against the supplied insurance policy PDF.

It retrieves relevant policy evidence using hybrid retrieval, reranks the retrieved evidence, and passes structured information through multiple specialized agents.

If the available evidence is insufficient, the system can return `NEEDS_REVIEW` instead of making an unsupported decision.

## Architecture

```text
Claim
  ↓
Case Analysis Agent
  ↓
Hybrid Policy Retrieval
(BM25 + Dense Retrieval)
  ↓
Cross-Encoder Reranking
  ↓
Coverage & Exclusion Agent
  ↓
Decision Agent
  ↓
Validation Agent
  ↓
Final Decision + Evidence

Agents

1. Case Analysis Agent

Extracts relevant claim facts
Identifies decision dimensions
Identifies missing evidence
Creates an investigation plan

2. Policy Evidence Agent

Generates targeted retrieval queries
Performs BM25 and dense retrieval
Combines retrieval results
Applies cross-encoder reranking
Returns policy evidence with page and chunk metadata

3. Coverage & Exclusion Agent

Analyzes coverage
Waiting periods
Exclusions
Definitions
Policy limits
Evidence sufficiency

4. Decision Agent
Produces one of:

ADMISSIBLE
ADMISSIBLE_WITH_LIMITS
PARTIALLY_ADMISSIBLE
NOT_ADMISSIBLE
NEEDS_REVIEW

Decision statements contain provenance such as claim facts, policy rules, calculations, and inferences.

5. Validation Agent
Checks:

Claim facts
Policy support
Citations
Calculations
Unsupported claims

RAG Pipeline
Policy PDF
   ↓
PDF Parsing
   ↓
Meaningful Policy Chunking
   ↓
Dense Retrieval + BM25
   ↓
Hybrid Fusion
   ↓
Cross-Encoder Reranking
   ↓
Policy Evidence

Each policy chunk contains:

Page
Section
Chunk ID
Text
Metadata

This allows decisions to reference the source policy evidence.

API
Health
GET /health

Analyze Claim
POST /analyze

The API returns:

Case ID
Decision
Confidence
Key findings
Applicable limits
Missing evidence
Policy citations
Validation result
Agent trace
Frontend

The project includes a Streamlit interface that allows users to:

Enter or paste claim JSON
Analyze claims
View the decision and confidence
View key findings
View policy limits
View missing evidence
Inspect policy citations
View validation results
Inspect the agent trace

NEEDS_REVIEW decisions are clearly displayed when evidence is insufficient.

Evaluation
Public Cases

All 12 supplied public cases were executed.

PUB-001  ADMISSIBLE_WITH_LIMITS
PUB-002  NOT_ADMISSIBLE
PUB-003  ADMISSIBLE
PUB-004  ADMISSIBLE_WITH_LIMITS
PUB-005  ADMISSIBLE
PUB-006  NEEDS_REVIEW
PUB-007  PARTIALLY_ADMISSIBLE
PUB-008  NOT_ADMISSIBLE
PUB-009  ADMISSIBLE
PUB-010  ADMISSIBLE
PUB-011  NEEDS_REVIEW
PUB-012  NEEDS_REVIEW

All 12 completed with Validation: PASS.

Results:

public_evaluation_results.json

Additional Cases

Five additional cases were created to test:

Initial waiting period
Domiciliary treatment and limits
Pre/post-hospitalization expenses
Missing hospital-definition evidence
Cosmetic treatment exclusion
ADD-001  ADMISSIBLE
ADD-002  ADMISSIBLE_WITH_LIMITS
ADD-003  ADMISSIBLE_WITH_LIMITS
ADD-004  NEEDS_REVIEW
ADD-005  NOT_ADMISSIBLE

All 5 completed with Validation: PASS.

Results:

additional_evaluation_results.json

Technology Stack
Python
FastAPI
Streamlit
Google Gemini
Sentence Transformers
BM25
Cross-Encoder Reranking
PyMuPDF
Pydantic
LangGraph

Project Structure
Aptino_Candidate_Package_FINAL/
│
├── app/
│   ├── agents/
│   ├── api/
│   ├── frontend/
│   ├── ingestion/
│   └── retrieval/
│
├── candidate_data/
├── policy/
├── schema/
│
├── public_evaluation_results.json
├── additional_evaluation_results.json
├── requirements.txt
├── .gitignore
└── README.md

Setup

Install dependencies:

pip install -r requirements.txt

Create .env:

GEMINI_API_KEY=your_api_key_here

Run the backend:

uvicorn app.api.main:app --reload

Run the frontend in another terminal:

streamlit run app/frontend/streamlit_app.py
Design Decisions
Hybrid Retrieval

BM25 handles exact policy terminology while dense retrieval captures semantic similarity.

Reranking

A cross-encoder reranks retrieved policy chunks before they are passed to reasoning agents.

Evidence Grounding

Policy citations contain page, section, and chunk identifiers.

Abstention

The system can return NEEDS_REVIEW when required evidence is unavailable instead of assuming missing facts.

Limitations

Designed around the supplied policy and synthetic claims.
LLM outputs can have some variability.
Retrieval quality depends on chunking and query formulation.
Production deployment would require authentication, monitoring, rate limiting, and additional testing.

Security

API keys are stored in environment variables and .env is excluded through .gitignore.

Never commit API keys to the repository.


**This is the version I'd use.** It gives the evaluator everything they need without wasting your time.

Now save `README.md`. After that, **we should move immediately to GitHub and deployment**, because those are more important for today's submission than making the README longer.