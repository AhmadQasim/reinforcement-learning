import fire
import gym
import numpy as np
import random
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from baselines.common.atari_wrappers import FrameStack, WarpFrame


class DeepQNetwork:
    def __init__(self):
        self.env = gym.make('SpaceInvaders-v0')
        self.replay_buffer = []
        self.replay_buffer_size_thresh = 350
        self.env = WarpFrame(self.env)
        self.env = FrameStack(self.env, 4)
        self.episodes = 100
        self.max_actions_per_episode = 500
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.eps_decay = 0.00025
        self.decay_step = 0
        self.learning_rate = 0.8
        self.discount_factor = 0.99
        self.rewards = []
        self.test_eps = 50
        self.test_rewards = []
        self.model = None
        self.batch_size = 64
        self.model_path = 'models/DQN.hdf5'

    def create_model(self):
        inputs = layers.Input(shape=(84, 84, 4))

        conv1 = layers.Conv2D(32, 8, 2)(inputs)
        batch_norm1 = layers.BatchNormalization()(conv1)
        relu1 = layers.Activation('relu')(batch_norm1)

        conv2 = layers.Conv2D(64, 4, 2)(relu1)
        batch_norm2 = layers.BatchNormalization()(conv2)
        relu2 = layers.Activation('relu')(batch_norm2)

        conv3 = layers.Conv2D(128, 4, 2)(relu2)
        batch_norm3 = layers.BatchNormalization()(conv3)
        relu3 = layers.Activation('relu')(batch_norm3)

        x = layers.Flatten()(relu3)
        fc1 = layers.Dense(512)(x)
        fc2 = layers.Dense(self.env.action_space.n)(fc1)

        model = models.Model(inputs=inputs, outputs=fc2)
        model.compile(optimizer='rmsprop', loss='mse')
        model.summary()
        self.model = model

    def save_to_memory(self, experience):
        # experience = (observation, action, reward, done, new_observation)
        if len(self.replay_buffer) > self.replay_buffer_size_thresh:
            del self.replay_buffer[0]
        self.replay_buffer.append(experience)

    def sample_from_memory(self):
        return random.sample(self.replay_buffer,
                             min(len(self.replay_buffer), self.batch_size))

    def fill_empty_memory(self):
        observation = self.env.reset()
        for _ in range(self.batch_size):
            new_observation, action, reward, done = self.take_action(observation)
            self.save_to_memory((observation, action, reward, done, new_observation))
            if done:
                new_observation = self.env.reset()
            observation = new_observation

    def take_action(self, observation):
        # take random actiom
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        # take best action
        else:
            action = np.argmax(self.model.predict(np.expand_dims(observation, axis=0)))

        # take action
        new_observation, reward, done, info = self.env.step(action)
        new_observation = np.asarray(list(new_observation))
        return new_observation, action, reward, done

    def train(self):
        # initialize deep-q agent
        self.create_model()

        # fill empty memory before starting training
        self.fill_empty_memory()

        for i in range(self.episodes):
            print("Episode: ", i)
            observation = np.asarray(list(self.env.reset()))
            total_reward_per_episode = 0

            for a in range(self.max_actions_per_episode):
                self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * np.exp(-self.eps_decay * self.decay_step)
                self.decay_step += 1

                # take step
                new_observation, action, reward, done = self.take_action(observation)

                # save to experience buffer
                self.save_to_memory((observation, action, reward, done, new_observation))

                # sample minibatch from memory
                minibatch = self.sample_from_memory()
                x_batch = []
                q_targets = []

                # for each experience in minibatch, set q-target
                for idx, (state, act, rew, done, next_state) in enumerate(minibatch):
                    x_batch.append(state)
                    if done:
                        next_state_q_value = rew
                    else:
                        next_state_q_value = np.max(
                            self.model.predict(np.expand_dims(np.asarray(list(next_state)), axis=0)))

                    curr_q_vals = self.model.predict(np.expand_dims(np.asarray(list(state)), axis=0))
                    curr_q_vals[0][act] = rew + self.discount_factor * next_state_q_value
                    q_targets.append(curr_q_vals[0])

                # train agent on minibatch
                self.model.fit(np.asarray(x_batch), np.asarray(q_targets), batch_size=len(minibatch))

                # track reward per episode
                total_reward_per_episode += reward

                # update state
                observation = new_observation

            self.rewards.append(total_reward_per_episode)
            print("Reward: ", total_reward_per_episode)
        self.model.save(self.model_path)
        self.env.close()

        print("Average reward: ", sum(self.rewards) / self.episodes)

    def test(self):
        # test agent
        self.model = models.load_model('models/DQN.hdf5')
        for i in range(self.test_eps):
            observation = np.asarray(list(self.env.reset()))
            total_reward_per_episode = 0
            for _ in range(self.max_actions_per_episode):
                self.env.render()
                action = np.argmax(self.model.predict(np.expand_dims(observation, axis=0)))
                new_observation, reward, done, info = self.env.step(action)
                total_reward_per_episode += reward
                observation = new_observation
                if done:
                    break
            self.test_rewards.append(total_reward_per_episode)

        print("Average reward for test agent: ", sum(self.test_rewards) / self.test_eps)


if __name__ == "__main__":
    fire.Fire(DeepQNetwork)