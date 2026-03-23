import numpy as np
import gymnasium as gym
from gymnasium import spaces
from candidate import Candidate, DYNAMIC_SIGNALS


NUM_DYNAMIC = len(DYNAMIC_SIGNALS)   # 5
NUM_STATIC  = 5
STATE_SIZE  = NUM_STATIC + NUM_DYNAMIC  # 10

ACTION_ASK_START  = 0   # actions 0-4: ask a dynamic signal question
ACTION_FLAG       = 5   # action 5: final decision — fraud
ACTION_PASS       = 6   # action 6: final decision — legitimate

MAX_QUESTIONS     = 5   # max dynamic questions per episode
REPEAT_PENALTY    = -0.1
CORRECT_REWARD    = 1.0
WRONG_REWARD      = -1.5
MISSED_FRAUD_PENALTY = -1.5  # missing a fraud is worse than a false positive


class CandidateEnv(gym.Env):

    def __init__(self, dataset: list):
        super().__init__()

        self.dataset     = dataset
        self.current_idx = 0       # cycles through dataset during training

        # tells stable-baselines3 the shape and range of the state vector
        # low=-1 because unasked dynamic signals use -1 as placeholder
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(STATE_SIZE,), dtype=np.float32
        )

        # 5 dynamic questions + FLAG + PASS
        self.action_space = spaces.Discrete(7)

        # set on every reset()
        self.candidate     = None
        self.state         = None
        self.asked_actions = set()
        self.questions_asked = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # pick next candidate, cycle back when dataset exhausted
        self.candidate    = self.dataset[self.current_idx % len(self.dataset)]
        self.current_idx += 1

        # static features loaded, dynamic slots set to -1
        static            = self.candidate.static_features()
        dynamic           = np.full(NUM_DYNAMIC, -1.0, dtype=np.float32)
        self.state        = np.concatenate([static, dynamic])

        # reset episode tracking
        self.asked_actions   = set()
        self.questions_asked = 0

        return self.state, {}

    def _compute_fraud_probability(self) -> float:
        # weight per signal — reflects group importance
        static_weights = [
            0.083, 0.083, 0.083,   # Group 1: 25% across 3 signals
            0.075, 0.075,           # Group 2: 15% across 2 signals
        ]
        dynamic_weight = 0.12       # Group 3: 60% across 5 signals (12% each)

        weighted_sum = 0.0
        total_weight = 0.0

        # static signals always contribute
        for i, w in enumerate(static_weights):
            weighted_sum += self.state[i] * w
            total_weight += w

        # dynamic signals only contribute when observed (not -1)
        for i in range(NUM_DYNAMIC):
            score = self.state[NUM_STATIC + i]
            if score != -1.0:
                weighted_sum += score * dynamic_weight
                total_weight += dynamic_weight

        legitimacy_score  = weighted_sum / total_weight
        fraud_probability = 1.0 - legitimacy_score

        return float(np.clip(fraud_probability, 0.0, 1.0))

    def step(self, action: int):
        terminated = False
        truncated  = False
        reward     = 0.0

        # record fraud probability before action (for information gain calculation)
        prob_before = self._compute_fraud_probability()

        # --- CASE 1 & 2: agent asked a question (action 0-4) ---
        if action < ACTION_FLAG:

            if action in self.asked_actions:
                # Case 1 — repeat question, penalize and do nothing
                reward = REPEAT_PENALTY

            else:
                # Case 2 — new question, reveal score and update state
                score = self.candidate.get_dynamic_score(action)
                self.state[NUM_STATIC + action] = score
                self.asked_actions.add(action)
                self.questions_asked += 1

                # reward = how much this question shifted the fraud belief
                prob_after = self._compute_fraud_probability()
                reward     = abs(prob_after - prob_before)

                # question budget exhausted — force episode end
                if self.questions_asked >= MAX_QUESTIONS:
                    terminated = True

        # --- CASE 3: agent made a final decision (action 5 or 6) ---
        else:
            terminated  = True
            agent_flags = (action == ACTION_FLAG)

            if agent_flags == self.candidate.is_fraud:
                reward = CORRECT_REWARD                  # correct decision
            elif agent_flags and not self.candidate.is_fraud:
                reward = WRONG_REWARD                    # false positive
            else:
                reward = -MISSED_FRAUD_PENALTY           # missed fraud — heaviest penalty

        return self.state, reward, terminated, truncated, self._get_info()

    def _get_info(self) -> dict:
        return {
            "candidate_id":      self.candidate.candidate_id,
            "is_fraud":          self.candidate.is_fraud,
            "fraud_probability": self._compute_fraud_probability(),
            "questions_asked":   self.questions_asked,
            "asked_actions":     list(self.asked_actions),
        }
