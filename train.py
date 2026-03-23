from stable_baselines3 import DQN

from candidate import generate_dataset
from environment import CandidateEnv


def train():
    # 1. generate synthetic training dataset
    # seed=42 → training candidates
    # seed=99 → test candidates (used in evaluate.py, never seen during training)
    dataset = generate_dataset(
        n_candidates=800,
        fraud_ratio=0.4,
        seed=42
    )

    # 2. create environment
    env = CandidateEnv(dataset)

    # 3. create DQN model
    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-3,
        buffer_size=10000,
        batch_size=32,
        exploration_fraction=0.3,    # first 30% of training = mostly random
        exploration_final_eps=0.05,  # settle at 5% random after exploration
        verbose=1
    )

    # 4. train
    print("Starting training...")
    model.learn(total_timesteps=50000)
    print("Training complete.")

    # 5. save
    model.save("fraud_detector")
    print("Model saved to fraud_detector.zip")


if __name__ == "__main__":
    train()
