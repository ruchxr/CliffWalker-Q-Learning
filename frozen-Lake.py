import gym
import pickle
import numpy as np

env = gym.make("FrozenLake-v1")

q_table = np.random.uniform(low=-2,high=0,size=(16,4))

learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.9852
epsilon_decay_value = 0.999999
episodes = 250000
batch_size = 10

rewards = []

for episode in range(episodes):
    state,_ = env.reset()
    done = False
    total_reward = 0
    
    if episode % batch_size == 0:
        print(f"Episode: {episode}, Epsilon: {epsilon:.4f}")
    
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()
        
        next_state, reward, done, _,_ = env.step(action)

        max_future_q = np.max(q_table[next_state])
        current_q = q_table[state, action]  # Corrected indexing
        
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        q_table[state, action] = new_q  # Corrected indexing
        
        total_reward += reward
        state = next_state
        
        if done and reward == 0:
            q_table[state, action] = reward
        
    rewards.append(total_reward)
    epsilon = max(epsilon * epsilon_decay_value, 0.01)

print("Average reward over episodes:", np.mean(rewards))
print(q_table)

with open('q_table_frozen_lake.pkl', 'wb') as f:
    pickle.dump(q_table, f)

