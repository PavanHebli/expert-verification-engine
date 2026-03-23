# Knowledge Integrity & Expert Verification Engine

A reinforcement learning system that learns to detect expertise fraud — candidates who use AI to fabricate screening answers and curate LinkedIn profiles for roles they never held.

---

## The Problem

Identity fraud is solved. The new frontier is **knowledge fraud**: a candidate who is real, verified, and entirely fabricated in terms of expertise. A sophisticated fraudster does not make obvious profile mistakes. They use LLMs to generate technically correct answers, mirror senior-level job descriptions, and produce polished narratives that pass a surface-level screen.

The fraud lives in the answers — in the absence of lived specificity that no language model can reliably generate.

---

## Approach

The system frames candidate evaluation as a **sequential decision problem**. An agent interviews one candidate per episode, choosing which signals to probe before committing to a final FLAG or PASS decision. Rather than applying a fixed scoring rubric to every candidate, the agent learns a state-dependent policy: which question to ask first given the profile shape, when accumulated evidence is sufficient to decide, and how to sequence follow-up questions when early signals are ambiguous.

Full signal metric calculations: [`calculations.md`](calculations.md)

---

## Signal Design

Signals are organized into three groups, each targeting a distinct layer of the detection problem.

### Group 1 — Career Physics (Static)
Structural signals derived from the candidate's profile. Always visible at episode start.

| Signal | What it captures |
|---|---|
| `complexity_match` | Does scope of work align with stated years of experience? |
| `horizontal_inflation` | Is the candidate claiming equal depth across too many unrelated domains? |
| `domain_transition_credibility` | When switching domains, is there a traceable technical bridge? |

### Group 2 — Web Corroboration (Static)
Treated as **positive signal only**. Absence of public presence is not penalized.

| Signal | What it captures |
|---|---|
| `project_complexity_match` | Do public repos reflect the architectural depth expected at stated seniority? |
| `endorsement_quality` | Are endorsers technically relevant? Bulk or reciprocal endorsement patterns are penalized. |

### Group 3 — Answer Quality (Dynamic)
The primary discriminators. These are only revealed when the agent explicitly asks. LLMs produce breadth without specificity — correct frameworks, generic challenges, no friction. These five signals target what lived experience produces and language models cannot reliably replicate.

| Signal | What it captures |
|---|---|
| `failure_recall_depth` | Can they name what failed, why, and how it was specifically resolved? |
| `stance_conviction` | Do they take defensible positions grounded in experience, or hedge uniformly? |
| `knowledge_boundary_awareness` | Do they show honest ignorance in areas adjacent to claimed expertise? |
| `org_context_specificity` | Do they reference real org constraints, or give textbook recommendations? |
| `narrative_friction` | Does the answer have a real situation and obstacle, or a polished numbered framework? |

---

## RL Framework

**State** — 10-dimensional vector: 5 static signals (always visible) + 5 dynamic signals (initialized to `-1`, filled in as the agent asks questions).

**Actions** — 7 discrete choices:
- Actions `0–4`: ask one of the 5 dynamic signal questions
- Action `5`: FLAG (fraud)
- Action `6`: PASS (legitimate)

**Reward**
- Per question: information gain — `|fraud_probability_after - fraud_probability_before|`
- Terminal: `+1.0` correct decision, `-1.0` false positive, `-1.5` missed fraud
- The asymmetry reflects real-world cost — a missed fraud is worse than an unnecessary flag

**Episode** — Begins with one candidate loaded and static features visible. Ends when the agent makes a FLAG/PASS decision or exhausts the question budget of 5.

**Model** — DQN with MlpPolicy via `stable-baselines3`. Trained for 50,000 timesteps on 800 synthetic candidates (seed=42). Evaluated on 200 unseen candidates (seed=99).

---

## Project Structure

```
├── candidate.py        # Candidate dataclass + synthetic data generation
├── environment.py      # Gymnasium environment (state, actions, rewards)
├── question_bank.py    # Maps action indices to signal names and interview questions
├── train.py            # Training loop — generates dataset, trains DQN, saves model
├── evaluate.py         # Evaluation loop — loads model, tests on unseen candidates
├── calculations.md     # Full metric calculations for all 10 signals
└── requirements.txt    # Dependencies
```

---

## Setup

```bash
# create and activate environment
conda create -n fraud_rl python=3.12
conda activate fraud_rl

# install dependencies
pip install -r requirements.txt
```

---

## Running

**Train the model:**
```bash
python train.py
```
Generates 800 synthetic candidates, trains a DQN for 50,000 timesteps, saves `fraud_detector.zip`.

**Evaluate the model:**
```bash
python evaluate.py
```
Loads the saved model, tests on 200 unseen candidates (seed=99), prints accuracy, false positive rate, missed fraud rate, and average questions asked per episode.

**Sanity check data generation:**
```bash
python candidate.py
```
Prints a sample of 10 candidates with static and dynamic score averages to verify signal distributions.

---

## Design Decisions

**Repeat penalty vs. action masking**

The environment penalizes repeat questions (`-0.1`) rather than masking them out entirely. The cleaner production approach would be action masking — maintaining a set of unused actions and restricting the model to pick only from that set at each step. This eliminates the possibility of the agent getting stuck in a repeat loop and removes the need for the `max_steps` safety guard in `evaluate.py`.

Action masking was not implemented here because it requires either a custom policy or `MaskableDQN` from `sb3-contrib`, adding complexity beyond the scope of this proof-of-concept. The repeat penalty achieves the same outcome during training at the cost of a small defensive guard at evaluation time.

**Synthetic data distributions**

Fraud candidates: static signals sampled from `Beta(3,2)` (mean ~0.60) — profile looks believable, not obviously wrong. Dynamic signals from `Beta(2,8)` (mean ~0.20) — answer quality consistently low.

Real candidates: static signals from `Beta(5,2)` (mean ~0.71). Dynamic signals from `Beta(7,2)` (mean ~0.78). `project_complexity_match` intentionally uses `Beta(4,3)` (mean ~0.57) — not every real expert has a polished public GitHub, and absence of web presence is not a fraud signal.

**Train/test split via seeds**

Rather than splitting one dataset, training and evaluation generate separate datasets with different seeds (`seed=42` for 800 training candidates, `seed=99` for 200 test candidates). This avoids any risk of the model having seen test candidates during training.

---

## Bonus: Multi-Modal Live Evaluator

In a live video interview setting, the RL framework extends naturally to real-time signal streams.

**Signals to extract:**

| Modality | Signal | What it captures |
|---|---|---|
| Audio | Response latency | Delay before answering — genuine recall takes time; retrieval from an LLM is fast and uniform |
| Audio | Filler and self-correction patterns | Real experts self-correct and backtrack; LLM output is linearly structured |
| Visual | Gaze deviation | Sustained off-screen gaze during technical questions suggests reading from another source |
| Visual | Micro-expression consistency | Confidence displayed vs. linguistic hedging — mismatches are a signal |
| Screen | Tab switching | Switching away during a technical question and returning before answering |
| Screen | Typing patterns | Unusually fast structured input during open-ended questions |

**Training approach:**

The state vector would be extended to include streaming features alongside the existing profile and answer quality signals. The agent would observe a rolling window of behavioral signals per question and learn to weigh them alongside the linguistic content of the answer. Reward structure remains the same — the agent is still optimizing for correct FLAG/PASS decisions, now with richer real-time evidence. Initial training would require a labeled dataset of live interviews with known outcomes, with the synthetic data approach used here serving as a warm-start prior.
