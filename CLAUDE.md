# 🚀 Iterative LLM-as-Data-Generator + Self-Instruct + Continuous Fine-tuning Workflow

This workflow combines your **active learning loop** with the **Self-Instruct pipeline** ([paper 2212.10560](https://arxiv.org/pdf/2212.10560)), adapted for **testcase generation** and **continuous fine-tuning**.

---

## 🔁 End-to-End Workflow

### 1. Initialization
- **Seed dataset**: small set of human-labeled examples (or none if bootstrapping).
- **Coverage grid**: define intents/entities/task categories for synthetic coverage.
- **Model checkpoint**: pretrained tiny model (student) to fine-tune.

> 🔗 Parallel to Self-Instruct: instead of "seed tasks," you provide **task categories**, and the LLM generates instructions/testcases.

---

### 2. Synthetic Data Generation (Self-Instruct style)
- Use a large LLM (teacher) to **generate testcases** based on:
  - Coverage grid
  - Error taxonomy (from prior iterations)
  - Low-confidence predictions from the student
  - Random long-tail cases for diversity
- Apply **in-prompt constraints** (deduplication, typos, formats, locale noise).
- **Two-pass labeling**: LLM A generates, LLM B verifies.

---

### 3. Data Curation
- **Deduplication & filtering**
  - N-gram uniqueness
  - Semantic dedup (cosine sim / LSH)
  - Balance across intents/entities
- **Verifier pass** for schema/labels/offsets.
- Store clean data into `data_pool`.

---

### 4. Continuous Fine-tuning
- Train tiny model starting from **last checkpoint** (not from scratch).
- Each loop incorporates newly curated synthetic examples.
- **Curriculum scheduling**: begin with easier cases, progress to harder negatives.
- **Stopping criteria**:
  - Dev/test F1 improvement < ε for K loops, or
  - Budget constraints reached.

---

### 5. Evaluation & Monitoring
- **Frozen human-curated dev/test set** (never touched by LLM).
- Track metrics:
  - Per-intent and per-entity performance
  - Error taxonomy buckets
  - Calibration (ECE)
- Detect drift between synthetic and real data.

---

### 6. Human-in-the-Loop Checkpoints
Although mostly autonomous, humans are needed for:
1. **Frozen dev/test creation** (initial seed).
2. **Periodic sanity checks** (review 1–5% of production queries).
3. **Schema drift detection** (synthetic data may drift in format).
4. **Long-run monitoring** (multi-hour/day runs can accumulate subtle errors).

---

## 📥 Inputs Required from User

To run the system **with minimal human intervention**, users must provide the following:

### A. Data & Task Definition
1. **Task specification** (classification, span extraction, intent/entity schema).
2. **Coverage grid / taxonomy** of categories & entities.
   - Example: `{ intent: "payment", entities: [currency, recipient, amount] }`
3. **Seed dev/test set** (small, human-labeled, frozen).
4. **Optional**: seed training data (human examples).

### B. Model Setup
5. **Base checkpoint** of tiny model (student).
6. **Teacher LLM** (large model) for data generation.
7. **Verifier LLM** (can reuse teacher with different prompts or a cheaper model).

### C. Generation & Training Config
8. **Generation hyperparameters**:
   - temperature, top_p, penalties, #examples per bucket/loop.
9. **Deduplication thresholds**:
   - cosine similarity cutoff, n-gram uniqueness rules.
10. **Budget constraints**:
    - max examples, compute, loops, early stop criteria.
11. **Curriculum strategy**:
    - Mix ratios (e.g., 40% errors, 40% low-confidence, 20% random).

### D. Monitoring & Human Intervention Points
12. **Human review budget/schedule** (e.g., review 50 samples per 10k synthetic examples).
13. **Alert thresholds** (e.g., dev F1 drop, schema errors > 5%).
14. **Logging/output destination** (datasets, checkpoints, metrics).

---

## 📝 Bottom Line
- **Self-Instruct** provides the **pattern**: use a huge LLM to bootstrap new examples iteratively.
- **Your loop** adds **guardrails, active learning signals, and continuous fine-tuning**.
- To make it run hands-free, the **user must supply upfront**:
  - Schema (task definition + taxonomy)
  - A frozen human test/dev set
  - Initial student checkpoint
  - Generator/verifier LLM access
  - Config knobs (hyperparams, budget, curriculum mix, thresholds)

After setup, the pipeline can run for **hours/days autonomously**, with only **periodic human sanity checks** to prevent drift.

---

## 🏗️ Project Structure

### `src/`
Shared package containing common utilities, models, and core logic used across different components (app, training, data generation, etc.). This promotes code reuse and maintains consistency across the workflow components.

### `app/`
Main application - a simple web service with REST API endpoints. Instead of implementing the workflow through direct coding, the app exposes HTTP endpoints that the workflow can call into. This design provides:
- **Decoupling**: Separates workflow orchestration from implementation logic
- **Monitoring**: Easier to track and observe workflow execution through API calls
- **Scalability**: Can be deployed and scaled independently
- **Flexibility**: Different workflow runners can interact with the same core functionality