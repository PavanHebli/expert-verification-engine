# Signal Metric Calculations

Reference document for how each signal in the fraud detection system is measured and vectorized into a float [0, 1].

**Score interpretation (consistent across all signals):**
- Score near `1.0` → legitimate / real expert
- Score near `0.0` → suspicious / likely fraud

---

## GROUP 1: Career Physics

---

### 1. `complexity_match`

**What it measures:** Does the scope of work described in the candidate's profile match what is realistically achievable at their stated years of experience (YOE)?

---

**Complexity Scale (1-5):**

| Level | Description | Example |
|---|---|---|
| 1 | Task execution | "Implemented feature X, fixed bugs" |
| 2 | Module ownership | "Owned component Y, designed small system" |
| 3 | System design | "Designed service Z, made architectural decisions" |
| 4 | Platform leadership | "Led team, architected platform serving X users" |
| 5 | Org-level influence | "Cross-team principal decisions, defined roadmap" |

---

**YOE → Expected Complexity Range:**

| YOE | Expected Level |
|---|---|
| 0-2 years | 1 – 2 |
| 3-5 years | 2 – 3 |
| 6-9 years | 3 – 4 |
| 10+ years | 4 – 5 |

---

**Formula:**
```
score = 1 - (gap / max_possible_gap)

gap               = claimed_complexity - expected_midpoint
max_possible_gap  = 4
```

---

**Examples:**

```
Fraud candidate:
  YOE = 2
  Job description: "Architected and led a real-time ML inference
                    platform serving 50M daily users across 3 regions"
  Extracted complexity level  = 4 (platform leadership)
  Expected midpoint for 2 YOE = 1.5
  Gap                         = 4 - 1.5 = 2.5
  Score                       = 1 - (2.5 / 4) = 0.375  ← suspicious

Real candidate:
  YOE = 8
  Job description: "Led ML platform team, designed inference
                    infrastructure, made architectural decisions"
  Extracted complexity level  = 4
  Expected midpoint for 8 YOE = 3.5
  Gap                         = 4 - 3.5 = 0.5
  Score                       = 1 - (0.5 / 4) = 0.875  ← legitimate
```

---

**Production method:** NLP / LLM classifier on job description text → extract scope indicators (team size, scale, ownership language) → map to 1-5 scale

**PoC:** Synthetic float sampled from distributions calibrated to reflect separation between fraudulent and legitimate profiles

---

### 2. `horizontal_inflation`

**What it measures:** Real expertise is spiky — deep in 1-2 domains, shallow in others. A fraud profile claims equal depth across many unrelated domains simultaneously.

---

**Step 1 — Map skills to domain buckets:**

```
PyTorch, TensorFlow, sklearn     →  ML domain
Kubernetes, MLflow, Docker       →  MLOps domain
Spark, Kafka, dbt                →  Data Engineering domain
RAG, LangChain, fine-tuning      →  LLM domain
React, TypeScript, Vue           →  Frontend domain
OpenCV, YOLO                     →  Computer Vision domain
```

---

**Step 2 — Score depth per domain using proficiency signals from profile text:**

| Signal in text | Example keywords | Points |
|---|---|---|
| Leadership / architecture | "led", "architected", "designed", "owned" | 3 |
| Builder language | "built", "developed", "implemented", "proficient" | 2 |
| Exposure language | "familiar", "basic", "worked with", "used", "exposure to" | 1 |

Sum points per domain → map raw score to depth (1-5):

```
Raw score   →   Depth
0 - 1 pts   →   1
2 - 3 pts   →   2
4 - 5 pts   →   3
6 - 8 pts   →   4
9+ pts      →   5
```

---

**Step 3 — Measure spikiness via standard deviation:**

```
score = min(std_dev(domain_depth_vector) / threshold, 1.0)

threshold = 1.5  (calibrated empirically)

high std dev  →  spiky profile  →  score near 1.0  →  legitimate
low std dev   →  flat profile   →  score near 0.0  →  suspicious
```

---

**Full Example:**

```
Candidate profile text:
  "Architected and led ML training pipelines using PyTorch and TensorFlow.
   Proficient in model evaluation with sklearn.
   Basic knowledge of Kubernetes for deployment.
   Familiar with Spark for data processing.
   Built RAG systems using LangChain. Expert in LLM fine-tuning."

Extraction:
  ML domain:              PyTorch (3) + TensorFlow (3) + sklearn (2)  →  raw 8  →  depth 4
  MLOps domain:           Kubernetes (1)                              →  raw 1  →  depth 1
  Data Engineering:       Spark (1)                                   →  raw 1  →  depth 1
  LLM domain:             LangChain (2) + fine-tuning (3)             →  raw 5  →  depth 3

Domain depth vector = [4, 1, 1, 3]
std_dev = 1.26
score   = min(1.26 / 1.5, 1.0) = 0.84  ← legitimate (spiky profile)
```

```
Fraud candidate:
  Claims equal expertise across ML, MLOps, Backend, Data Engineering, CV, LLM
  Domain depth vector = [5, 5, 5, 5, 5, 5]
  std_dev = 0.0
  score   = 0.0  ← highly suspicious (flat profile)
```

---

**Production method:** NLP extraction of proficiency signals from profile text and job descriptions using keyword/regex matching or LLM-based extraction

**PoC:** Synthetic float sampled from distributions calibrated to reflect separation between fraudulent and legitimate profiles

---

### 3. `domain_transition_credibility`

**What it measures:** When a candidate switches domains, is there a traceable technical bridge that makes the transition believable?

---

**Two sub-scores:**

```
1. domain_overlap_score  — how much technical foundation is shared between past and present domains?
2. bridge_bonus          — does a specific project exist that spans both domains?
```

---

**Step 1 — Identify past and present domain** from job titles, job descriptions, and claimed skills.

---

**Step 2 — Score domain overlap using a predefined matrix:**

| Transition | Overlap Score | Shared Foundation |
|---|---|---|
| NLP → LLM | 0.9 | Almost directly related |
| ML → LLM | 0.8 | PyTorch, model training, evaluation |
| CV → LLM | 0.7 | Deep learning, transformers, embeddings |
| Backend → MLOps | 0.6 | Infrastructure, APIs, deployment |
| Data Eng → ML | 0.5 | Python, data pipelines, feature engineering |
| CV → Backend | 0.2 | Little technical overlap |
| Frontend → LLM | 0.2 | Little technical overlap |

---

**Step 3 — Check for bridge project:**

A bridge project spans both past and present domains (e.g., CV engineer has a "Visual QA system using CLIP + LLM" project).

```
bridge_bonus = 0.3 if bridge project exists, else 0.0
```

---

**Formula:**

```
score = min(domain_overlap_score + bridge_bonus, 1.0)
```

---

**Examples:**

```
Case 1 — CV engineer → LLM role, has bridge project:
  domain_overlap  = 0.7
  bridge_bonus    = 0.3
  score           = min(0.7 + 0.3, 1.0) = 1.0  ← strong positive

Case 2 — CV engineer → LLM role, no bridge project:
  domain_overlap  = 0.7
  bridge_bonus    = 0.0
  score           = 0.7  ← still reasonable, domain overlap carries it

Case 3 — Frontend engineer → LLM expert, no bridge:
  domain_overlap  = 0.2
  bridge_bonus    = 0.0
  score           = 0.2  ← suspicious, answer quality becomes critical tiebreaker

Case 4 — Frontend engineer → LLM, built LLM-powered UI:
  domain_overlap  = 0.2
  bridge_bonus    = 0.3
  score           = 0.5  ← borderline, answer quality decides
```

---

**Production method:** Skill taxonomy graph for dynamic domain overlap scoring + NLP scan of GitHub repos and project descriptions for bridge project detection

**PoC:** Predefined overlap matrix for common transitions + synthetic bridge boolean

---

## GROUP 2: Web Corroboration

---

### 4. `project_complexity_match`

**What it measures:** Do the candidate's public GitHub repositories in the claimed domain reflect engineering depth expected at their seniority level?

---

**Step 1 — Filter repos by claimed domain** using keyword matching on repo names, descriptions, and dependencies.

---

**Step 2 — Score each repo's complexity (1-5):**

| Level | Description | Example |
|---|---|---|
| 1 | Tutorial / beginner | Single script, follows a tutorial, bare API calls |
| 2 | Basic portfolio | Slightly structured, minimal custom logic |
| 3 | Intermediate | Multiple components, some custom logic, demonstrates understanding |
| 4 | Advanced | Architectural decisions, custom implementations, handles edge cases |
| 5 | Production-grade | Testing, CI/CD, performance optimization, scalable design |

---

**Step 3 — Map claimed seniority to expected complexity:**

| Seniority | YOE | Expected Project Complexity |
|---|---|---|
| Junior | 0-2 | 1 – 2 |
| Mid | 3-5 | 2 – 3 |
| Senior | 6-9 | 3 – 4 |
| Staff / Principal | 10+ | 4 – 5 |

---

**Formula:**

```
avg_complexity = mean(top_3_repo_complexity_scores)

gap   = max(0, expected_midpoint - avg_complexity)
        # only penalize if projects are BELOW expected
        # projects at or above expected → gap = 0 → no penalty

score = 1 - (gap / max_possible_gap)
max_possible_gap = 4
```

---

**Examples:**

```
Fraud — Senior ML (8 YOE) claiming LLM expertise:
  "LLM chatbot"   → OpenAI API wrapper, single script       → complexity 1
  "RAG system"    → LangChain tutorial, minimal custom logic → complexity 2
  "ML classifier" → sklearn notebook                        → complexity 1

  avg_complexity    = (1 + 2 + 1) / 3 = 1.33
  expected midpoint = 3.5
  gap               = max(0, 3.5 - 1.33) = 2.17
  score             = 1 - (2.17 / 4) = 0.46  ← suspicious

Real — Senior ML (8 YOE) claiming LLM expertise:
  "Custom RAG eval framework" → custom metrics, multiple retrievers   → complexity 4
  "Fine-tuning pipeline"      → custom dataset prep, training loop    → complexity 4
  "LLM inference optimizer"   → batching, caching, latency benchmarks → complexity 4

  avg_complexity    = 4.0
  expected midpoint = 3.5
  gap               = max(0, 3.5 - 4.0) = 0
  score             = 1.0  ← legitimate
```

---

**Production method:** GitHub API → analyze repo structure (file count, module depth, dependency sophistication, presence of tests/CI config) → map to 1-5 scale using heuristics or LLM-based repo assessment

**PoC:** Synthetic float sampled from distributions calibrated to reflect separation between fraudulent and legitimate profiles

---

### 5. `endorsement_quality`

**What it measures:** Are endorsers technically credible and relevant to the skill endorsed, and do endorsement patterns look organic or manufactured?

---

**Two components:**

```
1. endorser_quality_score  — how credible and relevant are the endorsers?
2. pattern_penalties       — do endorsement patterns look manufactured?
```

---

**Step 1 — Score each endorser on two dimensions:**

Relevance — does the endorser's domain match the skill being endorsed?

| Endorser domain vs. skill | Points |
|---|---|
| Same domain ("Senior ML Engineer" endorsing PyTorch) | 3 |
| Adjacent domain ("Data Engineer" endorsing PyTorch) | 2 |
| Unrelated ("Marketing Manager" endorsing PyTorch) | 1 |

Seniority — how senior is the endorser?

| Endorser title | Points |
|---|---|
| Senior / Staff / Principal / Lead / Manager / Director | 3 |
| Mid-level Engineer / Analyst | 2 |
| Junior / Entry / Student | 1 |

```
per_endorser_score = (relevance_points × seniority_points) / 9
                     # 9 = max possible (3×3), normalizes to 0-1

base_score = mean(per_endorser_score across all endorsers)
```

---

**Step 2 — Apply pattern penalties:**

```
Burst endorsements (>50% arriving in same month):
  if (endorsements_in_peak_month / total_endorsements) > 0.5 → penalty += 0.3

Reciprocal endorsements (candidate endorsed back >70% of endorsers):
  if (reciprocal_count / total_endorsements) > 0.7 → penalty += 0.2
```

---

**Formula:**

```
score = max(0, base_score - penalties)
```

---

**Examples:**

```
Fraud — 10 endorsements for PyTorch:
  8 × "Software Developer" (unrelated, junior)   → each (1×1)/9 = 0.11
  2 × "Data Analyst" (adjacent, mid)             → each (2×2)/9 = 0.44

  base_score      = (8×0.11 + 2×0.44) / 10 = 0.176
  burst penalty   = 8/10 in same 2-week window → +0.3
  reciprocal pen. = 9/10 reciprocal            → +0.2
  score           = max(0, 0.176 - 0.5) = 0.0  ← highly suspicious

Real — 6 endorsements for PyTorch:
  3 × "Senior ML Engineer" (same domain, senior) → each (3×3)/9 = 1.0
  2 × "ML Lead" (same domain, senior)            → each (3×3)/9 = 1.0
  1 × "Data Engineer" (adjacent, mid)            → each (2×2)/9 = 0.44

  base_score = (3×1.0 + 2×1.0 + 1×0.44) / 6 = 0.91
  penalties  = 0  (spread over 2 years, only 2/6 reciprocal)
  score      = 0.91  ← strong positive
```

---

**Production method:** LinkedIn graph API → extract endorser titles, seniority, endorsement timestamps, cross-reference reciprocal endorsements

**PoC:** Synthetic float sampled from distributions calibrated to reflect separation between fraudulent and legitimate profiles

---

## GROUP 3: Answer Quality

---

### 6. `failure_recall_depth`

**What it measures:** Can the candidate recall a specific failure with named cause, concrete consequence, and specific resolution?

---

**Four core dimensions (0 or 1 each):**

| Dimension | Generic = 0 | Specific = 1 |
|---|---|---|
| `what` | "the model", "the system" | "our BERT classifier for ticket routing" |
| `why` | "data drift", "infrastructure issues" | "annotation team changed guidelines without notifying us" |
| `impact` | "it affected production" | "800 mis-routed tickets over 2 weeks" |
| `resolution` | "we added monitoring" | "per-category accuracy alerts + monthly retraining with sliding window" |

**Two bonus dimensions (0 or 0.5 each):**

```
timeline_bonus = 0.5 if any time reference present
                 ("after 3 months", "in Q2", "two weeks before launch")

team_bonus     = 0.5 if team/person reference present
                 ("we", "our data team", "the ML lead")
```

---

**Formula:**

```
raw_score = what + why + impact + resolution + timeline_bonus + team_bonus
max_raw   = 4 + 0.5 + 0.5 = 5

score = raw_score / 5
```

---

**Examples:**

```
LLM answer:
  "It's important to monitor models in production for data drift.
   Best practices include alerting systems and rollback mechanisms."

  what = 0, why = 0, impact = 0, resolution = 0, timeline = 0, team = 0
  score = 0 / 5 = 0.0  ← highly suspicious

Real expert answer:
  "Our BERT classifier for customer support ticket routing started
   misclassifying after 3 months. The support team had started using a
   new ticket template with different vocabulary from our training data.
   We didn't catch it for 2 weeks — cost us 800 mis-routed tickets.
   Fixed it with per-category accuracy alerts and monthly retraining
   using a sliding window of recent tickets."

  what       = 1  (BERT classifier for customer support ticket routing)
  why        = 1  (new ticket template, vocabulary mismatch)
  impact     = 1  (2 weeks undetected, 800 mis-routed tickets)
  resolution = 1  (per-category alerts + sliding window retraining)
  timeline   = 0.5  ("after 3 months", "2 weeks")
  team       = 0.5  ("support team", "we")
  score = 5 / 5 = 1.0  ← legitimate
```

---

**Production method:** NLP analysis of answer text → keyword detection for specificity markers per dimension, or LLM-based scoring with structured rubric

**PoC:** Synthetic float sampled from distributions calibrated to reflect separation between fraudulent and legitimate profiles

---

### 7. `stance_conviction`

**What it measures:** Does the candidate take a defensible position grounded in personal experience, or hedge across all sides equally?

---

**Two dimensions:**

```
1. stance_strength       — did they actually take a position?
2. experience_grounding  — is the position backed by personal experience?
```

---

**Dimension 1 — Stance Strength (0 to 1)**

Count opinion markers and hedge markers via keyword/phrase matching:

Opinion markers (positive):
```
Strong stance:      "I prefer", "I would choose", "I'd choose", "I strongly recommend",
                    "I recommend", "I would avoid", "I'd avoid", "honestly", "frankly",
                    "overengineered", "underrated", "never use", "always use"
Comparative:        "X is better than", "X beats Y", "I'd pick X over Y"
Contrarian:         "despite what", "contrary to", "everyone says X but"
Direct:             "use X", "don't use X", "avoid X", "stick with X"
```

Hedge markers (negative):
```
Dependency:         "it depends", "it really depends", "depends on your"
Both-sides:         "both have", "both approaches", "merits and drawbacks",
                    "advantages and disadvantages", "trade-offs to consider"
Context deferral:   "your specific use case", "your specific needs",
                    "your requirements", "your team size"
Best practice:      "it's important to", "best practice is", "it's recommended to",
                    "you should consider", "generally speaking"
```

```
opinion_marker_count = number of opinion phrase matches in answer text
hedge_marker_count   = number of hedge phrase matches in answer text

if opinion_markers == 0 and hedge_markers == 0:
    stance_strength = 0.3        # no opinion expressed at all

elif hedge_markers == 0:
    stance_strength = 1.0        # pure opinion, no hedging

else:
    hedge_ratio     = hedge_markers / (opinion_markers + hedge_markers)
    stance_strength = 1 - hedge_ratio
```

---

**Dimension 2 — Experience Grounding (0 or 1)**

```
Experience markers (score = 1):
  "I used", "we tried", "we chose", "we switched", "in my experience",
  "when I worked at", "when we built", "we found that"

General knowledge markers (score = 0):
  "teams typically", "organizations often", "best practice is",
  "according to the docs", "if you were to use X"
```

---

**Formula:**

```
score = 0.6 × stance_strength + 0.4 × experience_grounding
```

---

**Examples:**

```
LLM answer:
  "Both approaches have their merits. It really depends on your specific
   use case and infrastructure requirements. It's important to consider
   the trade-offs carefully."

  Opinion markers:  0
  Hedge markers:    "both approaches have their merits" → 1
                    "it really depends" → 1
                    "your specific use case" → 1
                    "infrastructure requirements" → 1
                    "it's important to consider" → 1
                    "trade-offs" → 1  →  total = 6

  hedge_ratio       = 6 / (0 + 6) = 1.0
  stance_strength   = 1 - 1.0 = 0.0
  experience_grounding = 0  (no "I/we used/tried")
  score = 0.6 × 0.0 + 0.4 × 0.0 = 0.0  ← highly suspicious

Real expert answer:
  "Honestly, feature stores are overengineered for most teams. We tried
   Feast at my last company and spent 3 months on infrastructure before
   scrapping it. I'd only recommend one if you have multiple teams sharing
   features across dozens of models."

  Opinion markers:  "honestly" → 1, "overengineered" → 1,
                    "I'd only recommend" → 1  →  total = 3
  Hedge markers:    0

  stance_strength      = 1 - (0 / (3 + 0)) = 1.0
  experience_grounding = 1  ("we tried Feast", "at my last company")
  score = 0.6 × 1.0 + 0.4 × 1.0 = 1.0  ← legitimate
```

---

**Production method:** Regex phrase matching against curated keyword dictionary per marker type, or fine-tuned text classifier trained on labeled interview answers

**PoC:** Synthetic float sampled from distributions calibrated to reflect separation between fraudulent and legitimate profiles

---

### 8. `knowledge_boundary_awareness`

**What it measures:** Real experts have honest gaps in adjacent areas. LLMs answer adjacent questions with the same generic confidence as core questions — no awareness of where their knowledge ends.

---

**Two sub-scores:**

```
1. admission_score  — did they show honest uncertainty on adjacent questions?
2. contrast_score   — is there meaningful depth difference between core and adjacent answers?
```

---

**Sub-Score 1 — Admission Score (0 or 1)**

Detect admission language in the adjacent domain answer:

```
Admission markers (score = 1):
  "I haven't worked with that specifically"
  "I've only used X at a surface level"
  "that's outside my core area"
  "I know the concept but haven't implemented it"
  "our infra team handled that, I wasn't involved"
  "I'd have to look into that more"

Fake confidence markers (score = 0):
  Complete structured answer to adjacent topic with no hedging
  Same level of detail as core domain answers
  No acknowledgment of limited experience
```

---

**Sub-Score 2 — Contrast Score (0 to 1)**

Measure specificity of core vs. adjacent answer using simplified `failure_recall_depth` scoring:

```
core_specificity     = specificity score of core domain answer     (0-1)
adjacent_specificity = specificity score of adjacent domain answer (0-1)

contrast_score = max(0, core_specificity - adjacent_specificity)
```

---

**Formula:**

```
score = 0.5 × admission_score + 0.5 × contrast_score
```

---

**Examples:**

```
Candidate claims: Senior ML Engineer, PyTorch / model training expertise
Core question:     "Walk me through debugging a model overfitting in production"
Adjacent question: "How do you configure PersistentVolumeClaims in Kubernetes?"

Fraud candidate:
  Core answer:     generic regularization advice, no specific model/failure
                   core_specificity = 0.2
  Adjacent answer: structured YAML/storage class explanation, no admission
                   admission_score      = 0
                   adjacent_specificity = 0.2

  contrast_score = max(0, 0.2 - 0.2) = 0.0
  score = 0.5 × 0 + 0.5 × 0.0 = 0.0  ← highly suspicious

Real ML engineer:
  Core answer:     "Our ResNet-50 was overfitting after epoch 8 — only 2000
                    images per class. Added elastic transforms, dropped LR to
                    1e-5. Validation accuracy went from 67% to 84%."
                   core_specificity = 0.9
  Adjacent answer: "Honestly our infra team handled the K8s config. I wrote
                    job specs but didn't deal with PVC configs directly."
                   admission_score      = 1
                   adjacent_specificity = 0.1

  contrast_score = max(0, 0.9 - 0.1) = 0.8
  score = 0.5 × 1 + 0.5 × 0.8 = 0.9  ← legitimate
```

---

**Production method:** Two-question probe strategy (one core, one adjacent) → NLP scoring of both answers for specificity + admission language detection

**PoC:** Synthetic float sampled from distributions calibrated to reflect separation between fraudulent and legitimate profiles

---

### 9. `org_context_specificity`

**What it measures:** Does the candidate explain decisions within real organizational constraints — team dynamics, resource limits, existing infrastructure — or give textbook recommendations as if operating in a vacuum?

---

**Two dimensions:**

```
1. org_language_score       — do they use "we", team references, constraint language?
2. constraint_decision_link — is there a specific constraint that explains a specific decision?
```

---

**Dimension 1 — Org Language Score (0 to 1)**

Count org-specific markers vs. textbook markers:

```
Org language markers (positive):
  First person plural:    "we", "our team", "our company"
  Team references:        "the data team", "the ML platform team",
                          "the infra team", "our manager"
  Constraint language:    "we couldn't because", "we had to", "we weren't allowed to",
                          "budget constraints", "compliance required", "policy against"
  Historical context:     "we had already", "at the time we", "given our existing"
  Org decision context:   "the team decided", "we agreed that", "the decision was made"

Textbook markers (negative):
  "you should", "one should", "teams should"
  "best practice is", "it is recommended", "it's important to"
  "the standard approach", "organizations typically"

if positive == 0 and negative == 0:
    org_language_score = 0.3    # no org language at all — slightly suspicious
elif negative == 0:
    org_language_score = 1.0
else:
    org_language_score = positive / (positive + negative)
```

---

**Dimension 2 — Constraint-Decision Link (0 or 1)**

Is there a specific org constraint that directly explains a specific decision?

```
Score = 1:
  "We chose Airflow over Prefect because our platform team had a hard rule
   against Python-based schedulers after a bad Celery incident"
  → constraint: policy against Python schedulers → decision: chose Airflow

  "We batched inference instead of real-time because our SLA was 24h —
   real-time infra wasn't worth the cost"
  → constraint: 24h SLA + cost pressure → decision: batch inference

Score = 0:
  "We should use Airflow because it's the industry standard"
  → no org constraint, just a general recommendation
```

---

**Formula:**

```
score = 0.5 × org_language_score + 0.5 × constraint_decision_link
```

---

**Examples:**

```
LLM answer:
  "When choosing an ML infrastructure stack, it's important to consider
   scalability, maintainability, and team expertise. You should evaluate
   tools like MLflow, Airflow, and Kubernetes."

  Org markers:      "your team" → 1  (weak, no "we", no specific team)
  Textbook markers: "it's important to", "you should" → 2
  org_language_score       = 1 / (1 + 2) = 0.33
  constraint_decision_link = 0
  score = 0.5 × 0.33 + 0.5 × 0 = 0.165  ← suspicious

Real expert answer:
  "We were already on Airflow for data pipelines so MLflow was a natural
   extension — the data team knew the DAG paradigm. We evaluated Prefect
   but our platform team had a hard rule against Python-based schedulers
   after a bad Celery incident. Kubernetes was off the table because our
   infra team had standardized on AKS with a networking setup that made
   custom deployments painful."

  Org markers:      "we", "our data team", "our platform team",
                    "our infra team", "we evaluated", "we were already" → 6
  Textbook markers: 0
  org_language_score       = 6 / (6 + 0) = 1.0
  constraint_decision_link = 1  ("policy against Python schedulers" → no Prefect,
                                  "AKS networking setup" → K8s off the table)
  score = 0.5 × 1.0 + 0.5 × 1.0 = 1.0  ← legitimate
```

---

**Production method:** NLP → org marker frequency counting + pattern detection for constraint-decision links using dependency parsing or LLM-based rubric

**PoC:** Synthetic float sampled from distributions calibrated to reflect separation between fraudulent and legitimate profiles

---

### 10. `narrative_friction`

**What it measures:** Does the answer tell a story with a situation, friction, and resolution — or deliver a polished numbered framework? Real experience produces narrative with obstacles. LLMs produce structured frameworks with no friction.

---

**Two dimensions:**

```
1. narrative_structure_score  — does the answer have a story arc: situation → friction → resolution?
2. framework_penalty          — does the answer use framework language that replaces the story?
```

---

**Dimension 1 — Narrative Structure Score (0 to 1)**

Check for presence of all three story arc components:

```
Situation markers (sets up context in past tense):
  "we were", "at the time", "we had", "the situation was"
  past tense setup with specific role/timeframe

Friction markers (something went wrong or was difficult):
  "we ran into", "the problem was", "we struggled with", "it wasn't working"
  "we hit a wall", "turned out", "we realized", "didn't work as expected"
  "unexpectedly", "kept failing", "we discovered", "was a disaster"

Resolution markers (how it ended):
  "eventually", "finally", "we ended up", "what worked was"
  "the fix was", "we solved it by", "in the end", "what we learned"

situation  = 1 if situation markers present, else 0
friction   = 1 if friction markers present, else 0
resolution = 1 if resolution markers present, else 0

narrative_structure_score = (situation + friction + resolution) / 3
```

---

**Dimension 2 — Framework Penalty**

```
Framework markers:
  "there are N approaches / steps / considerations"
  "the key factors are", "key considerations include"
  "first... second... third..."  (used as steps, not story progression)
  "on one hand... on the other hand"
  "it's important to note", "it's worth considering"
  "in summary", "to summarize"

framework_penalty = min(framework_marker_count × 0.15, 0.5)
                    # capped at 0.5
```

---

**Formula:**

```
score = max(0, narrative_structure_score - framework_penalty)
```

---

**Examples:**

```
LLM answer:
  "There are several key considerations when deploying ML models. First,
   ensure model versioning. Second, implement monitoring for data drift.
   Third, set up rollback procedures. It's important to consider both
   technical and business requirements."

  situation = 0, friction = 0, resolution = 0
  narrative_structure_score = 0 / 3 = 0.0

  Framework markers: "there are several", "first", "second", "third",
                     "it's important to" → 5
  framework_penalty = min(5 × 0.15, 0.5) = 0.5

  score = max(0, 0.0 - 0.5) = 0.0  ← highly suspicious

Real expert answer:
  "Our first production deployment was a disaster honestly. We had tested
   the model locally but hadn't accounted for our pipeline running on a
   6-hour delay. The model was predicting on stale features for 3 days
   before a business analyst noticed the churn numbers looked off. We ended
   up adding a data freshness check as a pre-prediction gate — if features
   are older than 2 hours, we throw an alert instead of running inference."

  situation  = 1  ("we had tested", "our first production deployment")
  friction   = 1  ("was a disaster", "hadn't accounted for", "stale features", "3 days")
  resolution = 1  ("we ended up adding", "data freshness check", "pre-prediction gate")
  narrative_structure_score = 3 / 3 = 1.0

  Framework markers: 0  →  framework_penalty = 0

  score = max(0, 1.0 - 0) = 1.0  ← legitimate
```

---

**Production method:** NLP → marker detection per story arc component + framework pattern detection using regex or LLM-based rubric

**PoC:** Synthetic float sampled from distributions calibrated to reflect separation between fraudulent and legitimate profiles
