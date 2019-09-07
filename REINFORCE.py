import fire
import numpy as np
import gym
from gym import wrappers
from keras.layers import Input, Dense, Softmax
from keras.models import Model
import keras.models as models
import keras.backend as K

"""
Implementation of Vanilla Policy Gradient with Monte Carlo returns, also called REINFORCE
"""


class REINFORCE:
    def __init__(self, save_vid=False):
        self.env = gym.make('CartPole-v1')
        if save_vid:
            self.env = wrappers.Monitor(self.env, 'demos/', force=True)
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.n
        self.model = None
        self.episodes = 15000
        self.max_steps = 10000
        self.gamma = 0.95
        self.test_model = None
        self.test_episodes = 100
        self.test_rewards = []
        self.model_path = "models/REINFORCE.hdf5"

    def create_model(self):
        inputs = Input(shape=self.state_shape)
        labels = Input(shape=(self.action_shape, ))

        fc1 = Dense(10, activation='relu')(inputs)
        fc2 = Dense(self.action_shape, activation='relu')(fc1)
        fc3 = Dense(self.action_shape, activation='relu')(fc2)

        output = Softmax()(fc3)

        model = Model(inputs=[inputs, labels], outputs=output)
        model.add_loss(self.score_function(labels, output))
        model.compile(optimizer='adam', loss=None)

        test_model = Model(inputs=inputs, output=output)
        test_model.add_loss(self.score_function(labels, output))
        test_model.compile(optimizer='adam', loss=None)

        model.summary()

        self.model = model
        self.test_model = test_model

    @staticmethod
    def score_function(y_true, y_pred):
        j_gradient = - K.log(y_pred) * y_true
        return K.sum(j_gradient)

    def take_action(self, state):
        action_probs = self.test_model.predict(np.expand_dims(state, axis=0))
        action = np.random.choice(range(action_probs.shape[1]), p=action_probs.ravel())
        return action, action_probs

    def discounted_normalized_rewards(self, episode_rewards):
        discounted_rewards = np.zeros_like(episode_rewards)
        for idx, rewards in enumerate(episode_rewards):
            future_rewards = episode_rewards[idx:]
            powers = np.array(range(10)) + 1
            discounted_rewards[idx] = np.sum(list(map(lambda x, y: self.gamma**y + x, future_rewards, powers)))

        normalized_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

        return normalized_rewards

    def train(self):
        self.create_model()
        all_rewards = []
        for ep in range(self.episodes):
            observation = self.env.reset()
            episode_states, episode_actions, episode_rewards = [], [], []
            for step in range(self.max_steps):
                action, action_probs = self.take_action(observation)
                new_observation, reward, done, info = self.env.step(action)
                episode_states.append(observation)
                action_one_hot = np.zeros(self.action_shape)
                action_one_hot[action] = 1
                episode_actions.append(action_one_hot)
                episode_rewards.append(reward)

                observation = new_observation

                if done:
                    discounted_rewards = self.discounted_normalized_rewards(episode_rewards)
                    discounted_rewards = np.expand_dims(discounted_rewards, axis=1)
                    episode_actions = np.array(episode_actions) * discounted_rewards

                    model_input = [np.array(episode_states),
                                   episode_actions]
                    self.model.fit(model_input, batch_size=len(episode_states))

                    total_episode_reward = np.sum(episode_rewards)
                    all_rewards.append(total_episode_reward)
                    total_reward = np.sum(all_rewards)

                    print("Episode : ", ep)
                    print("Episode Reward : ", total_episode_reward)
                    print("Total Reward : ", total_reward)
                    print("Total Mean Reward: ", total_reward / (ep + 1))
                    print("==========================================")

                    break

        self.test_model.save(self.model_path)

    def test(self):
        # test agent
        agent = models.load_model(self.model_path, compile=False)
        for i in range(self.test_episodes):
            observation = np.asarray(list(self.env.reset()))
            total_reward_per_episode = 0
            while True:
                self.env.render()
                action_probs = agent.predict(np.expand_dims(observation, axis=0))
                action = np.random.choice(range(action_probs.shape[1]), p=action_probs.ravel())
                new_observation, reward, done, info = self.env.step(action)
                total_reward_per_episode += reward
                observation = new_observation
                if done:
                    break
            self.test_rewards.append(total_reward_per_episode)

        print("Average reward for test agent: ", sum(self.test_rewards) / self.test_episodes)


if __name__ == "__main__":
    fire.Fire(REINFORCE)
