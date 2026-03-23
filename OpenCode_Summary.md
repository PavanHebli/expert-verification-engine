# Expert Verification Engine - Project Summary

## Project Overview

**What it does:** RL-based system that detects "knowledge fraud" in job candidates — people who use ChatGPT to fabricate ML expertise they don't actually have.

**What it doesn't do:** This is NOT identity fraud detection. The candidate is a real, verified person — they are overstating what they know and what they have built.

---

## Core Architecture

### The RL Agent
- **Framework:** DQN via Stable-Baselines3
- **Policy:** MlpPolicy (neural network)
- **Actions:** 3 options (PASS, FLAG, PROBE)
- **Learning:** Through rewards from candidate evaluations

### The Five Signals (Features)

| # | Signal Name | Weight | What It Detects |
|---|-------------|--------|-----------------|
| 0 | `omniscience_score` | 0.30 | Skill breadth implausible for years of experience |
| 1 | `scar_tissue_score` | 0.28 | Absence of failure/recovery language |
| 2 | `adjacent_ignorance_score` | 0.22 | Unnaturally uniform knowledge depth |
| 3 | `opinion_fingerprint` | 0.15 | LLM-style hedging vs. practitioner conviction |
| 4 | `recency_gap_score` | 0.05 | Claimed experience vs. tool release date math |

---

## Data Generation

### Feature Distributions (Beta Sampling)
```
Genuine candidates:  Beta(2,7)  → scores LOW  (0.1-0.3 range)
Fraud candidates:   Beta(7,2)  → scores HIGH (0.7-0.9 range)
```

### Ground Truth
```
fraud_probability = Σ(features[i] × weights[i]) + small_noise
```

---

## RL Mechanics

### State Space
```
5-dimensional vector of floats in [0.0, 1.0]
Each value represents a fraud signal strength
```

### Action Space
| Action | Description | When to Use |
|--------|-------------|-------------|
| PASS (0) | Mark as genuine | Low scores across features |
| FLAG (1) | Mark as fraud | High scores across features |
| PROBE (2) | Ask follow-up | Uncertain, mixed signals |

### Reward System
| Outcome | Reward | Explanation |
|---------|--------|-------------|
| Correct FLAG | +1.0 | Fraud caught |
| Correct PASS | +0.5 | Genuine passed |
| False Positive | -1.5 | Genuine wrongly flagged (worst outcome) |
| Missed Fraud | -0.5 | Fraud slipped through |
| PROBE (budget left) | +0.1 | Good - seeking more info |
| PROBE (exhausted) | -0.2 | Bad - over-probing |

### PROBE Mechanic
```
PROBE = "I'm not sure, let me ask one more question"
→ Adds Gaussian noise (σ=0.05) to features
→ Represents new information from follow-up
→ Max 2 probes per candidate
→ After 2 probes, agent MUST decide (PASS or FLAG)
```

---

## File Structure

```
Fraud_detect_RL/
├── requirements.txt          # Dependencies
├── data_generator.py         # Generate mock candidates
├── utils.py                  # Helper functions
├── environment.py            # Gymnasium RL environment
├── train.py                  # Train DQN agent
├── evaluate.py               # Test & visualize agent
├── candidates.json           # Generated dataset
└── expert_verifier_dqn.zip   # Saved trained model
```

---

## Build Order

1. **`requirements.txt`** - List dependencies
2. **`data_generator.py`** - Create synthetic candidate data
3. **`utils.py`** - Helper functions (set_seed, moving_average, etc.)
4. **`environment.py`** - Gymnasium environment class
5. **`train.py`** - Train the DQN agent
6. **`evaluate.py`** - Evaluate and visualize results

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Labels | Binary (0=genuine, 1=fraud) | Simpler, assessment says "flag or pass" |
| Actions | 3 (PASS, FLAG, PROBE) | Full RL experience, realistic probing |
| Feature generation | Beta distributions | Creates realistic overlap for ambiguous cases |
| Primary metric | Cost-weighted score | Reflects business reality (false positive worse than miss) |

---

## Evaluation Metrics

| Metric | Formula | Priority |
|--------|---------|----------|
| **Cost-weighted score** | `1.0 - (FP×1.5 + FN×0.5) / n` | PRIMARY |
| F1 Score | 2 × (Precision × Recall) / (Precision + Recall) | Secondary |
| False Positive Rate | FP / (FP + TN) | Report explicitly |
| AUC-PR | Area under precision-recall curve | Handles imbalance |
| Recall | TP / (TP + FN) | Frauds caught |

---

## Assessment Deliverables

| Part | Description | Output |
|------|-------------|--------|
| Part 1 | Strategy Memo (thesis) | 1-page, <500 words |
| Part 2 | Technical Implementation | GitHub repo with code |
| Bonus | Multi-Modal Live Evaluator | Description document |

---

## Key Insight: How to Detect ChatGPT Experts

Genuine expertise is built through friction. Bugs hit, tools abandoned, tradeoffs regretted, knowledge that is deep in some areas and thin in others. AI-generated expertise has none of this. It is smooth, complete, and uniform. The system detects that smoothness.

### Signal 1: Omniscience (Weight 0.30)
- LLMs mirror job descriptions, not real career histories
- Real engineer: 3 years in NLP = gaps in MLOps, thin Spark knowledge
- Fraud: Claims full modern ML stack with no gaps

### Signal 2: Scar Tissue (Weight 0.28)
- LLMs produce success narratives by default
- Real engineers produce war stories: "Our first model failed in production..."
- Fraud: Every project delivered value, no regret, no rollback

### Signal 3: Adjacent Ignorance (Weight 0.22)
- A real expert's ignorance is as specific as their knowledge
- Real: "I have never had to scale to Spark — it was never in my stack"
- Fraud: Uniform medium-depth answers on every subfield

### Signal 4: Opinion Fingerprint (Weight 0.15)
- LLMs present balanced views by design — non-controversial
- Real: "I would not use AutoML in a regulated industry — black box problem"
- Fraud: "Both approaches have merits depending on use case"

### Signal 5: Recency Gap (Weight 0.05)
- Mathematically impossible timelines
- Example: claiming 10 years of Python 3.10 (released 2020)

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Agent always FLAGs | Cost-weighted score penalizes this |
| Agent always PASSes | Rewards are asymmetric (flag correct = +1.0) |
| PROBE overuse | Budget limit (2) + penalty for exhausted probe |
| Class imbalance | Fraud rate parameter in data generation |

---

## Implementation Status

| File | Status |
|------|--------|
| requirements.txt | Pending |
| data_generator.py | Pending |
| utils.py | Pending |
| environment.py | Pending |
| train.py | Pending |
| evaluate.py | Pending |
| Strategy Memo | Pending |
