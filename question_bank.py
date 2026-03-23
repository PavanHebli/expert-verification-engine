import random

# Maps action index (0-4) to signal name and interview questions.
# Action index matches DYNAMIC_SIGNALS order in candidate.py:
#   0 → failure_recall_depth
#   1 → stance_conviction
#   2 → knowledge_boundary_awareness
#   3 → org_context_specificity
#   4 → narrative_friction

QUESTION_BANK = {
    0: {
        "signal": "failure_recall_depth",
        "questions": [
            "Tell me about a time a model you deployed failed in production. What happened?",
            "Describe a project that went wrong. What was the root cause and how did you resolve it?",
            "What is the most painful technical failure you have experienced? Walk me through it.",
        ]
    },
    1: {
        "signal": "stance_conviction",
        "questions": [
            "Do you prefer feature stores or computing features on the fly? Strong opinion either way?",
            "MLflow vs Weights and Biases — which do you prefer and why?",
            "What is your take on real-time versus batch inference for production ML systems?",
        ]
    },
    2: {
        "signal": "knowledge_boundary_awareness",
        "questions": [
            "You mentioned Kubernetes — how do you handle persistent storage for long training jobs?",
            "You have worked with PyTorch — how do you approach distributed training across multiple nodes?",
            "How would you optimize database queries for a high-throughput feature serving layer?",
        ]
    },
    3: {
        "signal": "org_context_specificity",
        "questions": [
            "How did your team decide on your ML infrastructure stack?",
            "Tell me about a technical decision at your company that was shaped by org constraints.",
            "Describe a time you had to push back on a technical direction at your company.",
        ]
    },
    4: {
        "signal": "narrative_friction",
        "questions": [
            "Walk me through your most challenging deployment end to end.",
            "Tell me about a project where things did not go as planned.",
            "Describe a time you had to change your technical approach mid-project and why.",
        ]
    }
}


def get_question(action_idx: int) -> str:
    """
    Returns a random question for the given action index.
    Used during evaluation to display what the agent would ask a candidate.
    """
    return random.choice(QUESTION_BANK[action_idx]["questions"])


def get_signal_name(action_idx: int) -> str:
    """
    Returns the signal name for a given action index.
    Useful for logging which signal the agent chose to probe.
    """
    return QUESTION_BANK[action_idx]["signal"]
