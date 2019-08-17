import gym
import numpy as np

env = gym.make('FrozenLake-v0')
env.reset()

q_table = np.zeros((env.observation_space.n, env.action_space.n))
episodes = 15000
max_actions_per_episode = 100
epsilon = 1
min_epsilon = 0.01
eps_decay = 0.005
learning_rate = 0.8
discount_factor = 0.95
rewards = []

for i in range(episodes):
    print("Episode: ", i)
    observation = env.reset()
    total_reward_per_episode = 0
    for a in range(max_actions_per_episode):
        # take random action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        # take best action
        else:
            action = np.argmax(q_table[observation, :])
        q_value = q_table[observation, action]

        # take action
        new_observation, reward, done, info = env.step(action)
        new_state_q_value = np.max(q_table[new_observation, :])

        # update q-table
        q_table[observation, action] = q_value + learning_rate * (reward + (discount_factor*new_state_q_value - q_value))

        # track reward per episode
        total_reward_per_episode += reward

        # update state
        observation = new_observation
        if done:
            break
    if epsilon > min_epsilon:
        epsilon -= eps_decay
    rewards.append(total_reward_per_episode)
env.close()

# plot rewards
print("Average reward: ", sum(rewards)/episodes)
