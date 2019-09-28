import fire
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Implementation of Deep Deterministic Policy Gradients on A2C with TD-0 value returns
"""

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class PPO:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.n
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.old_actor = Actor(self.state_shape, self.action_shape)
        self.actor = Actor(self.state_shape, self.action_shape)
        self.critic = Critic(self.state_shape, self.action_shape)
        self.replay_buffer_states = torch.zeros(size=(1, self.state_shape[0]))
        self.replay_buffer_actions = torch.zeros(size=(1, self.action_shape))
        self.replay_buffer_rewards = torch.zeros(size=(1, 1))
        self.replay_buffer_done = torch.zeros(size=(1, 1), dtype=torch.float)
        self.replay_buffer_next_states = torch.zeros(size=(1, self.state_shape[0]))
        self.replay_buffer_size_thresh = 100000
        self.batch_size = 64
        self.episodes = 5000
        self.max_steps = 1000
        self.test_episodes = 1000
        self.discount_factor = 0.99
        self.test_rewards = []
        self.epochs = 10
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.eps_decay = 0.005
        self.default_q_value_actor = -1
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        # range of the action possible for Pendulum-v0
        self.act_range = 2.0
        self.model_path = "models/PPO_CartPole.hdf5"

        # models
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_loss = nn.MSELoss()
        self.hard_update(self.actor, self.old_actor)

    def save_to_memory(self, experience):
        if self.replay_buffer_states.shape[0] > self.replay_buffer_size_thresh:
            self.replay_buffer_states = self.replay_buffer_states[1:, :]
            self.replay_buffer_actions = self.replay_buffer_actions[1:, :]
            self.replay_buffer_rewards = self.replay_buffer_rewards[1:, :]
            self.replay_buffer_done = self.replay_buffer_done[1:, :]
            self.replay_buffer_next_states = self.replay_buffer_next_states[1:, :]
        self.replay_buffer_states = torch.cat([self.replay_buffer_states, experience[0]])
        self.replay_buffer_actions = torch.cat([self.replay_buffer_actions, experience[1]])
        self.replay_buffer_rewards = torch.cat([self.replay_buffer_rewards, experience[2]])
        self.replay_buffer_done = torch.cat([self.replay_buffer_done, experience[3]])
        self.replay_buffer_next_states = torch.cat([self.replay_buffer_next_states, experience[4]])

    def sample_from_memory(self):
        random_rows = np.random.randint(0, self.replay_buffer_states.shape[0], size=self.batch_size)
        return [self.replay_buffer_states[random_rows, :], self.replay_buffer_actions[random_rows, :],
                self.replay_buffer_rewards[random_rows, :], self.replay_buffer_done[random_rows, :],
                self.replay_buffer_next_states[random_rows, :]]

    def take_action(self, state):
        action_probs = self.actor.forward(torch.tensor(state, dtype=torch.float))
        action_probs = action_probs.cpu().detach().numpy()
        action = np.random.choice(range(action_probs.shape[0]), p=action_probs.ravel())
        new_observation, reward, done, info = self.env.step(action)
        return new_observation, action, reward, done

    def fill_empty_memory(self):
        observation = self.env.reset()
        for _ in range(100):
            new_observation, action, reward, done = self.take_action(observation)
            '''
            # Adjust reward based on car position
            reward = observation[0] + 0.5

            # Adjust reward for task completion
            if observation[0] >= 0.5:
                reward += 1
            '''
            reward = reward if not done else -100
            action_one_hot = torch.zeros(size=(1, self.action_shape))
            action_one_hot[0, action] = 1
            self.save_to_memory([torch.tensor(observation, dtype=torch.float).unsqueeze(0),
                                 action_one_hot,
                                 torch.tensor(reward, dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                 torch.tensor(done, dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                 torch.tensor(new_observation, dtype=torch.float).unsqueeze(0)
                                 ])
            if done:
                new_observation = self.env.reset()
            observation = new_observation

    @staticmethod
    def hard_update(source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def clipped_surrogate_objective(old_policy, new_policy, advantages):
        ratio = new_policy / (old_policy + 1e-10)
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        loss = torch.min(ratio*advantages, clipped_ratio*advantages)
        return -torch.mean(loss)

    def optimize_model(self):
        states, actions, rewards, done, next_states = self.sample_from_memory()

        done = 1 - done
        curr_v_vals = self.critic.forward(states)
        next_v_vals = self.critic.forward(next_states)

        target_v_vals = done * (self.discount_factor * next_v_vals)
        target_v_vals += rewards

        old_actor_prediction = self.old_actor.forward(states)
        new_actor_prediction = self.actor.forward(states)
        advantages = target_v_vals - curr_v_vals

        advantages = advantages * actions

        old_actor_prediction = old_actor_prediction.detach()
        advantages = advantages.detach()
        target_v_vals = target_v_vals.detach()

        # actor update
        self.actor.zero_grad()
        actor_loss = self.clipped_surrogate_objective(old_actor_prediction,
                                                      new_actor_prediction,
                                                      advantages)
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # critic update
        self.critic.zero_grad()
        critic_loss = self.critic_loss(self.critic.forward(states), target_v_vals)
        critic_loss.backward()
        self.critic_optim.step()

        self.hard_update(self.actor, self.old_actor)

    def train(self):
        self.fill_empty_memory()
        total_reward = 0

        for p in self.old_actor.parameters():
            p.requires_grad = False

        for ep in range(self.episodes):
            episode_rewards = []
            observation = self.env.reset()
            for step in range(self.max_steps):
                observation = np.squeeze(observation)
                new_observation, action, reward, done = self.take_action(observation)
                '''
                # Adjust reward based on car position
                reward = observation[0] + 0.5

                # Adjust reward for task completion
                if observation[0] >= 0.5:
                    reward += 1
                '''
                reward = reward if not done else -100
                action_one_hot = torch.zeros(size=(1, self.action_shape))
                action_one_hot[0, action] = 1
                self.save_to_memory([torch.tensor(observation, dtype=torch.float).unsqueeze(0),
                                     action_one_hot,
                                     torch.tensor(reward, dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                     torch.tensor(done, dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                     torch.tensor(new_observation, dtype=torch.float).unsqueeze(0)
                                     ])
                episode_rewards.append(reward)
                observation = new_observation
                self.optimize_model()

                if done:
                    break

            # episode summary
            total_reward += np.sum(episode_rewards)
            print("Episode : ", ep)
            print("Episode Reward : ", np.sum(episode_rewards))
            print("Total Mean Reward: ", total_reward / (ep + 1))
            print("==========================================")

            torch.save(self.actor, self.model_path)

    def test(self):
        # test agent
        actor = torch.load(self.model_path)
        for i in range(self.test_episodes):
            observation = np.asarray(list(self.env.reset()))
            total_reward_per_episode = 0
            while True:
                self.env.render()
                action_probs = actor.forward(torch.tensor(observation, dtype=torch.float))
                action_probs = action_probs.cpu().detach().numpy()
                action = np.random.choice(range(action_probs.shape[0]), p=action_probs.ravel())
                print(action)
                new_observation, reward, done, info = self.env.step(action)
                total_reward_per_episode += reward
                observation = new_observation
                if done:
                    break
            self.test_rewards.append(total_reward_per_episode)

        print("Average reward for test agent: ", sum(self.test_rewards) / self.test_episodes)


class Actor(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Actor, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.fc1 = nn.Linear(self.state_shape[0], 24)
        self.fc2 = nn.Linear(24, self.action_shape)

        # initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)

        return x


class Critic(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Critic, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.fc1 = nn.Linear(self.state_shape[0], 24)
        self.fc2 = nn.Linear(24, 1)

        # initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    fire.Fire(PPO)
