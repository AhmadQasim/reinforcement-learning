import random
import fire
from keras import models
from keras.layers import Input, Dense, Softmax
from keras.models import Model
import keras.backend as K
import gym
import numpy as np


class a2c:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.n
        self.actor_learn = None
        self.actor_predict = None
        self.critic = None
        self.replay_buffer = []
        self.replay_buffer_size_thresh = 100000
        self.batch_size = 64
        self.episodes = 300
        self.max_steps = 1000
        self.gamma = 0.95
        self.test_episodes = 100
        self.discount_factor = 0.99
        self.test_rewards = []
        self.model_path = "models/a2c.hdf5"

    def create_actor_model(self):
        inputs = Input(shape=self.state_shape)
        labels = Input(shape=(self.action_shape,))
        advantages = Input(shape=(1,))
        fc1 = Dense(10, activation='relu')(inputs)
        fc2 = Dense(self.action_shape, activation='relu')(fc1)
        fc3 = Dense(self.action_shape, activation='linear')(fc2)

        output = Softmax()(fc3)

        model = Model(inputs=[inputs, labels, advantages], outputs=output)
        model.add_loss(self.score_function(labels, output, advantages))
        model.compile(optimizer='adam', loss=None)

        test_model = Model(inputs=inputs, output=output)
        test_model.add_loss(self.score_function(labels, output, advantages))
        test_model.compile(optimizer='adam', loss=None)

        model.summary()

        self.actor_learn = model
        self.actor_predict = test_model

    @staticmethod
    def score_function(y_true, y_pred, advantage):
        j_gradient = - K.log(y_pred) * y_true * advantage
        return K.mean(j_gradient)

    def create_critic_model(self):
        inputs = Input(shape=self.state_shape)
        fc1 = Dense(10, activation='relu')(inputs)
        fc2 = Dense(self.action_shape, activation='relu')(fc1)
        fc3 = Dense(1, activation='linear')(fc2)

        model = Model(inputs=[inputs], outputs=fc3)
        model.compile(optimizer='adam', loss='mse')

        model.summary()

        self.critic = model

    def save_to_memory(self, experience):
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
            observation = new_observation

    def take_action(self, state):
        action_probs = self.actor_predict.predict(np.expand_dims(state, axis=0))
        action = np.random.choice(range(action_probs.shape[1]), p=action_probs.ravel())
        new_observation, reward, done, info = self.env.step(action)
        return new_observation, action, reward, done

    def train(self):
        self.create_actor_model()
        self.create_critic_model()
        self.fill_empty_memory()
        total_reward = 0

        for ep in range(self.episodes):
            episode_rewards = []
            observation = self.env.reset()
            for step in range(self.max_steps):
                new_observation, action, reward, done = self.take_action(observation)
                self.save_to_memory((observation, action, reward, done, new_observation))
                episode_rewards.append(reward)

                minibatch = self.sample_from_memory()
                x_batch = []
                v_targets = []
                advantages = []
                actions_one_hot = []

                for idx, (state, act, rew, done, next_state) in enumerate(minibatch):
                    x_batch.append(state)

                    # actor
                    action_one_hot = np.zeros(self.action_shape)
                    action_one_hot[act] = 1
                    actions_one_hot.append(action_one_hot)

                    # critic
                    if done:
                        next_state_v_value = rew
                    else:
                        next_state_v_value = np.max(
                            self.critic.predict(np.expand_dims(np.asarray(list(next_state)), axis=0)))

                    curr_v_vals = self.critic.predict(np.expand_dims(np.asarray(list(state)), axis=0))
                    old_v = curr_v_vals[0]
                    curr_v_vals[0] = rew + self.discount_factor * next_state_v_value
                    advantage = curr_v_vals[0] # - old_v
                    v_targets.append(curr_v_vals[0])
                    advantages.append(advantage)

                # fit models
                self.actor_learn.fit([np.array(x_batch), np.array(actions_one_hot), np.array(advantages)], batch_size=len(minibatch), verbose=0)
                self.critic.fit(np.asarray(x_batch), np.asarray(v_targets), batch_size=len(minibatch), verbose=0)

                observation = new_observation

            # episode summary
            total_reward += np.sum(episode_rewards)
            print("Episode : ", ep)
            print("Episode Reward : ", np.sum(episode_rewards))
            print("Total Mean Reward: ", total_reward / (ep + 1))
            print("==========================================")

        self.actor_predict.save(self.model_path)

    def test(self):
        # test agent
        self.model = models.load_model(self.model_path, compile=False)
        for i in range(self.test_episodes):
            observation = np.asarray(list(self.env.reset()))
            total_reward_per_episode = 0
            while True:
                self.env.render()
                action_probs = self.model.predict(np.expand_dims(observation, axis=0))
                action = np.random.choice(range(action_probs.shape[1]), p=action_probs.ravel())
                new_observation, reward, done, info = self.env.step(action)
                total_reward_per_episode += reward
                observation = new_observation
                if done:
                    break
            self.test_rewards.append(total_reward_per_episode)

        print("Average reward for test agent: ", sum(self.test_rewards) / self.test_episodes)


if __name__ == '__main__':
    fire.Fire(a2c)
