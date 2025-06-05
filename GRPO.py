# fine_tune_grpo_cartpole_full.py

import torch
import gym
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig, AutoModelForCausalLMWithValueHead

# --- CONFIG ---
MODEL_NAME = "sshleifer/tiny-gpt2"
SAVE_DIR = "./saved_model_grpo"
TOTAL_EPISODES = 300
EVAL_INTERVAL = 25
MAX_STEPS = 200

# --- ENV ---
env = gym.make("CartPole-v1")

# --- TOKENIZER & MODEL ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)

# --- GRPO CONFIG ---
config = GRPOConfig(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    max_length=32,
    beta=0.1,
    gamma=0.99,
    kl_penalty=0.1,
    seed=42,
    log_with=None,
    total_episodes=TOTAL_EPISODES,
    target_kl=0.1,
    mini_batch_size=4,
)


# --- REWARD FUNCTION ---
def reward_fn(state, done):
    return 1.0 if not done else -1.0


# --- GENERA ESPERIENZA ---
def generate_experience(model, env, tokenizer, max_steps=MAX_STEPS, render=False):
    state = env.reset()[0]
    done = False
    episode_reward = 0
    states, actions, rewards = [], [], []

    for _ in range(max_steps):
        if done:
            break

        prompt = str(state)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(input_ids=input_ids, max_new_tokens=1)

        action = int(output[0][-1]) % env.action_space.n

        if render:
            env.render()

        next_state, _, done, _, _ = env.step(action)
        reward = reward_fn(next_state, done)

        states.append(prompt)
        actions.append(str(action))
        rewards.append(reward)

        episode_reward += reward
        state = next_state

    return states, actions, rewards, episode_reward


# --- VALUTAZIONE ---
def evaluate_model(model, env, tokenizer, num_episodes=5):
    total_rewards = []
    for _ in range(num_episodes):
        _, _, _, reward = generate_experience(model, env, tokenizer)
        total_rewards.append(reward)
    return np.mean(total_rewards)


# --- TRAINER ---
trainer = GRPOTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
)

# --- TRAINING LOOP ---
reward_history = []

print(" Inizio training...\n")
for episode in range(1, TOTAL_EPISODES + 1):
    states, actions, rewards, total_reward = generate_experience(model, env, tokenizer)
    reward_history.append(total_reward)

    reward_tensor = torch.tensor(rewards, dtype=torch.float)
    trainer.train(prompts=states, samples=actions, rewards=reward_tensor)

    print(f"Episode {episode}/{TOTAL_EPISODES} | Reward: {total_reward:.1f}")

    # --- VALUTAZIONE PERIODICA ---
    if episode % EVAL_INTERVAL == 0:
        avg_reward = evaluate_model(model, env, tokenizer)
        print(f" Eval @ Episode {episode}: Avg reward: {avg_reward:.2f}")

        # --- Salva modello ---
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"💾 Modello salvato in {SAVE_DIR}")

# --- PLOT REWARD ---
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward over Time")
plt.grid()
plt.savefig("reward_plot.png")
plt.show()


