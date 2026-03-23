from stable_baselines3 import DQN

from candidate import generate_dataset
from environment import CandidateEnv, ACTION_FLAG, MAX_QUESTIONS
from question_bank import get_signal_name


def evaluate(model_path: str = "fraud_detector", n_candidates: int = 200):

    # 1. load trained model
    model = DQN.load(model_path)
    print(f"Model loaded from {model_path}.zip\n")

    # 2. generate test dataset — seed=99, never seen during training
    test_dataset = generate_dataset(
        n_candidates=n_candidates,
        fraud_ratio=0.4,
        seed=99
    )

    # 3. create test environment
    env = CandidateEnv(test_dataset)

    results = []

    # 4. run one episode per candidate
    for candidate in test_dataset:
        obs, _ = env.reset()
        done    = False
        questions_asked = []
        decision        = None
        agent_flagged   = None

        max_steps  = MAX_QUESTIONS * 3   # safety limit — prevents infinite loop on repeat actions
        step_count = 0

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            action    = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

            if action < ACTION_FLAG:
                # agent asked a question — record signal name
                signal = get_signal_name(action)
                if signal not in questions_asked:   # avoid logging repeats
                    questions_asked.append(signal)
            else:
                # agent made an explicit decision
                decision      = "FLAG" if action == ACTION_FLAG else "PASS"
                agent_flagged = (action == ACTION_FLAG)

        # if budget exhausted without explicit FLAG/PASS
        # use fraud probability as implicit decision
        if decision is None:
            fraud_prob    = info["fraud_probability"]
            agent_flagged = fraud_prob > 0.5
            decision      = "FLAG" if agent_flagged else "PASS"

        correct = (agent_flagged == candidate.is_fraud)

        results.append({
            "candidate_id":   candidate.candidate_id,
            "is_fraud":       candidate.is_fraud,
            "decision":       decision,
            "correct":        correct,
            "questions_asked": questions_asked,
            "n_questions":    len(questions_asked),
        })

    _print_summary(results)
    return results


def _print_summary(results: list):
    total          = len(results)
    correct        = sum(r["correct"] for r in results)
    false_positives = sum(1 for r in results if r["decision"] == "FLAG" and not r["is_fraud"])
    missed_frauds  = sum(1 for r in results if r["decision"] == "PASS" and r["is_fraud"])
    avg_questions  = sum(r["n_questions"] for r in results) / total

    print("=" * 55)
    print(f"  EVALUATION RESULTS  ({total} candidates)")
    print("=" * 55)
    print(f"  Accuracy:          {correct / total * 100:.1f}%  ({correct}/{total})")
    print(f"  False positives:   {false_positives}  ({false_positives / total * 100:.1f}%)")
    print(f"  Missed frauds:     {missed_frauds}  ({missed_frauds / total * 100:.1f}%)")
    print(f"  Avg questions:     {avg_questions:.1f} / {MAX_QUESTIONS}")
    print("=" * 55)

    # sample: show first 10 results
    print("\n  SAMPLE RESULTS (first 10)")
    print(f"  {'ID':<12} {'Fraud':<8} {'Decision':<8} {'Correct':<8} {'Questions Asked'}")
    print("  " + "-" * 70)
    for r in results[:10]:
        questions = ", ".join(r["questions_asked"]) if r["questions_asked"] else "none"
        correct   = "✓" if r["correct"] else "✗"
        print(f"  {r['candidate_id']:<12} {str(r['is_fraud']):<8} {r['decision']:<8} {correct:<8} {questions}")


if __name__ == "__main__":
    evaluate()
