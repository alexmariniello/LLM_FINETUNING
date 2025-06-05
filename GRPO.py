# fine_tune_grpo_cartpole.py

import torch
import gym
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig, AutoModelForCausalLMWithValueHead


# 1. ENV Setup
env = gym.make("CartPole-v1")

# 2. Tokenizer e modello base (lightweight, da HF)
model_name = "sshleifer/tiny-gpt2"  # modello leggero
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Preparazione del modello con una testa per il value (necessaria per GRPO)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# 4. Configurazione GRPO
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
    total_episodes=1000,
    target_kl=0.1,
    mini_batch_size=4,
)


# 5. Reward function
def reward_fn(state, done):
    # Ricompensa 1 per ogni passo se il palo non è caduto
    return 1.0 if not done else -1.0


# 6. Generatore di esperienze
def generate_experience(model, env, tokenizer, max_steps=200):
    state = env.reset()
    done = False
    episode_reward = 0
    states, actions, rewards = [], [], []

    for _ in range(max_steps):
        if done:
            break

        input_ids = torch.tensor([[tokenizer.encode(str(state), add_special_tokens=False)]])
        with torch.no_grad():
            output = model.generate(input_ids=input_ids, max_new_tokens=1)

        action = int(output[0][-1]) % env.action_space.n
        new_state, _, done, _, _ = env.step(action)

        reward = reward_fn(new_state, done)
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        episode_reward += reward
        state = new_state

    return states, actions, rewards, episode_reward


# 7. Trainer
trainer = GRPOTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
)

# 8. Training loop
for episode in range(config.total_episodes):
    states, actions, rewards, total_reward = generate_experience(model, env, tokenizer)
    print(f"[Episode {episode + 1}] Reward: {total_reward}")

    # Convert experience to text-like prompts (mocked)
    prompts = [str(s) for s in states]
    generated_actions = [str(a) for a in actions]
    reward_tensor = torch.tensor(rewards, dtype=torch.float)

    trainer.train(prompts=prompts, samples=generated_actions, rewards=reward_tensor)


