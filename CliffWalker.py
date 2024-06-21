import gymnasium as gym
import pickle
import numpy as np

env = gym.make("CliffWalking-v0")

q_table = np.random.uniform(low=-2, high=0, size=(48, 4))

learning_factor = 0.1
discount_factor = 0.99
epsilon = 0.999
epsilon_decay_value = 0.995
episodes = 200000
batch_size = 10

for episode in range(episodes):
    state, _ = env.reset()
    done = False

    if episode % batch_size == 0:
        print(f"Episode: {episode}, epsilon: {epsilon}")

    total_rewards = 0

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0, 4)

        next_state, reward, done, _, _ = env.step(action)

        max_future_q = np.max(q_table[next_state])
        current_q = q_table[state][action]
        q_table[state][action] = (1 - learning_factor) * current_q + learning_factor * (reward + discount_factor * max_future_q)

        total_rewards += reward
        state = next_state

    print(f"Total Reward collected in episode {episode}: {total_rewards}")

    epsilon = max(epsilon * epsilon_decay_value, 0.01)

env.close()

with open('q_table_cliff_walker.pkl', 'wb') as f:
    pickle.dump(q_table, f)
