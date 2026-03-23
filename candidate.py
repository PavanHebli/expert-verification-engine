import numpy as np
from dataclasses import dataclass


# Signal names in fixed order — imported by environment, agent, and training
# to ensure state vector indexing is consistent across the codebase
STATIC_SIGNALS = [
    "complexity_match",
    "horizontal_inflation",
    "domain_transition_credibility",
    "project_complexity_match",
    "endorsement_quality",
]

DYNAMIC_SIGNALS = [
    "failure_recall_depth",
    "stance_conviction",
    "knowledge_boundary_awareness",
    "org_context_specificity",
    "narrative_friction",
]

ALL_SIGNALS = STATIC_SIGNALS + DYNAMIC_SIGNALS


@dataclass
class Candidate:
    candidate_id: str
    is_fraud: bool

    # Group 1: Career Physics
    complexity_match: float               # scope of work matches YOE
    horizontal_inflation: float           # spiky expertise vs. uniform spread
    domain_transition_credibility: float  # traceable bridge between past and present

    # Group 2: Web Corroboration
    project_complexity_match: float       # GitHub repo depth vs. claimed seniority
    endorsement_quality: float            # endorser relevance + seniority + pattern

    # Group 3: Answer Quality (revealed only when agent asks)
    failure_recall_depth: float           # specific failures vs. generic best practices
    stance_conviction: float              # takes positions vs. uniform hedging
    knowledge_boundary_awareness: float   # honest gaps in adjacent areas
    org_context_specificity: float        # real org constraints vs. textbook answers
    narrative_friction: float             # story with obstacles vs. polished framework

    def static_features(self) -> np.ndarray:
        """
        Returns the 5 static signal scores as a numpy array.
        Called once at the start of each episode to initialize the state vector.
        """
        return np.array([
            self.complexity_match,
            self.horizontal_inflation,
            self.domain_transition_credibility,
            self.project_complexity_match,
            self.endorsement_quality,
        ], dtype=np.float32)

    def get_dynamic_score(self, signal_idx: int) -> float:
        """
        Returns the ground truth score for a dynamic signal when the agent asks for it.
        signal_idx maps directly to the agent's action index (0-4).
        """
        dynamic = [
            self.failure_recall_depth,
            self.stance_conviction,
            self.knowledge_boundary_awareness,
            self.org_context_specificity,
            self.narrative_friction,
        ]
        return dynamic[signal_idx]


def _generate_fraud_candidate(candidate_id: str, rng: np.random.Generator) -> Candidate:
    """
    Sophisticated fraudster: profile looks believable (moderate static scores)
    but answer quality is consistently low (low dynamic scores).

    Static scores are moderate — a sophisticated fraudster won't make obvious
    profile mistakes. The signal lives in the answers.
    """
    return Candidate(
        candidate_id=candidate_id,
        is_fraud=True,

        # Profile looks believable — moderate scores, not obviously wrong
        complexity_match=float(rng.beta(3, 2)),              # mean ~0.60
        horizontal_inflation=float(rng.beta(2, 5)),          # mean ~0.29
        domain_transition_credibility=float(rng.beta(4, 2)), # mean ~0.67
        project_complexity_match=float(rng.beta(2, 5)),      # mean ~0.29
        endorsement_quality=float(rng.beta(2, 3)),           # mean ~0.40

        # Answer quality consistently low — where the fraud is exposed
        failure_recall_depth=float(rng.beta(2, 8)),          # mean ~0.20
        stance_conviction=float(rng.beta(2, 8)),             # mean ~0.20
        knowledge_boundary_awareness=float(rng.beta(2, 8)),  # mean ~0.20
        org_context_specificity=float(rng.beta(2, 8)),       # mean ~0.20
        narrative_friction=float(rng.beta(2, 8)),            # mean ~0.20
    )


def _generate_real_candidate(candidate_id: str, rng: np.random.Generator) -> Candidate:
    """
    Real expert: strong answer quality, generally solid profile.
    Natural variation — real experts aren't perfect on everything.

    project_complexity_match uses Beta(4,3) intentionally — not every real
    expert has a polished public GitHub. Absence of web presence is not fraud.
    """
    return Candidate(
        candidate_id=candidate_id,
        is_fraud=False,

        # Profile generally solid, natural variation expected
        complexity_match=float(rng.beta(5, 2)),              # mean ~0.71
        horizontal_inflation=float(rng.beta(5, 2)),          # mean ~0.71
        domain_transition_credibility=float(rng.beta(4, 2)), # mean ~0.67
        project_complexity_match=float(rng.beta(4, 3)),      # mean ~0.57, intentional variance
        endorsement_quality=float(rng.beta(4, 2)),           # mean ~0.67

        # Answer quality consistently high — lived specificity
        failure_recall_depth=float(rng.beta(7, 2)),          # mean ~0.78
        stance_conviction=float(rng.beta(7, 2)),             # mean ~0.78
        knowledge_boundary_awareness=float(rng.beta(7, 2)),  # mean ~0.78
        org_context_specificity=float(rng.beta(7, 2)),       # mean ~0.78
        narrative_friction=float(rng.beta(7, 2)),            # mean ~0.78
    )


def generate_candidate(is_fraud: bool, candidate_id: str = None, rng: np.random.Generator = None) -> Candidate:
    """
    Public wrapper for quick one-off candidate generation.
    Used during testing and debugging — not during training.

    For training, use generate_dataset() which controls the rng properly.
    """
    if rng is None:
        rng = np.random.default_rng()
    if candidate_id is None:
        candidate_id = f"{'fraud' if is_fraud else 'real'}_{rng.integers(10000)}"

    if is_fraud:
        return _generate_fraud_candidate(candidate_id, rng)
    return _generate_real_candidate(candidate_id, rng)


def generate_dataset(n_candidates: int, fraud_ratio: float = 0.4, seed: int = 42):
    """
    Generates a shuffled dataset of synthetic candidates.

    Args:
        n_candidates: total number of candidates
        fraud_ratio:  fraction that are fraudulent (default 40%)
        seed:         fixed seed — same seed produces same dataset every run

    Returns:
        shuffled list of Candidate objects
    """
    rng = np.random.default_rng(seed)

    n_fraud = int(n_candidates * fraud_ratio)
    n_real  = n_candidates - n_fraud

    candidates = []
    for i in range(n_fraud):
        candidates.append(_generate_fraud_candidate(f"fraud_{i}", rng))
    for i in range(n_real):
        candidates.append(_generate_real_candidate(f"real_{i}", rng))

    indices = rng.permutation(len(candidates))
    return [candidates[i] for i in indices]


if __name__ == "__main__":
    dataset = generate_dataset(n_candidates=10, fraud_ratio=0.4, seed=42)

    print(f"{'ID':<12} {'Fraud':<8} {'Static Avg':>12} {'Dynamic Avg':>13}")
    print("-" * 50)
    for c in dataset:
        static_avg  = c.static_features().mean()
        dynamic_avg = np.mean([
            c.failure_recall_depth,
            c.stance_conviction,
            c.knowledge_boundary_awareness,
            c.org_context_specificity,
            c.narrative_friction,
        ])
        print(f"{c.candidate_id:<12} {str(c.is_fraud):<8} {static_avg:>12.3f} {dynamic_avg:>13.3f}")
