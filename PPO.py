import random
import fire
from keras import models
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import gym
import numpy as np

"""
Implementation of Proximal Policy Optimization on A2C with TD-0 value returns
"""


class ProximalPolicyOptimization:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.n
        self.old_actor = None
        self.old_actor_predict_only = None
        self.actor = None
        self.actor_predict_only = None
        self.critic = None
        self.replay_buffer = []
        self.replay_buffer_size_thresh = 100000
        self.batch_size = 64
        self.episodes = 1000
        self.max_steps = 1000
        self.gamma = 0.99
        self.test_episodes = 100
        self.discount_factor = 0.99
        self.test_rewards = []
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        self.epochs = 10
        self.model_path = "models/PPO.hdf5"

    def create_actor_model(self):
        inputs = Input(shape=self.state_shape)
        old_policy = Input(shape=(self.action_shape, ))
        advantages = Input(shape=(self.action_shape, ))

        fc1 = Dense(24, activation='relu', kernel_initializer="he_uniform")(inputs)
        output = Dense(self.action_shape, activation='softmax', kernel_initializer='he_uniform')(fc1)

        model = Model(inputs=[inputs, old_policy, advantages], outputs=output)
        model.add_loss(self.clipped_surrogate_objective(old_policy, output, advantages))
        model.compile(optimizer=Adam(lr=self.actor_lr), loss=None)
        model.summary()

        model_predict_only = Model(inputs=inputs, outputs=output)
        model_predict_only.add_loss(self.clipped_surrogate_objective(old_policy, output, advantages))
        model_predict_only.compile(optimizer=Adam(lr=self.actor_lr), loss=None)

        return model, model_predict_only

    @staticmethod
    def clipped_surrogate_objective(old_policy, new_policy, advantages):
        ratio = new_policy / (old_policy + 1e-10)
        clipped_ratio = K.clip(ratio, 0.8, 1.2)
        loss = K.minimum(ratio*advantages, clipped_ratio*advantages)
        return -K.mean(loss)

    def create_critic_model(self):
        inputs = Input(shape=self.state_shape)

        fc1 = Dense(24, activation='relu', kernel_initializer="he_uniform")(inputs)
        output = Dense(1, activation='linear', kernel_initializer='he_uniform')(fc1)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(lr=self.critic_lr), loss='mse')
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
        for _ in range(10000):
            new_observation, action, reward, done = self.take_action(observation)
            reward = reward if not done else -100
            self.save_to_memory((observation, action, reward, done, new_observation))
            if done:
                new_observation = self.env.reset()
            observation = new_observation

    def take_action(self, state):
        action_probs = self.actor_predict_only.predict(np.expand_dims(state, axis=0))
        action = np.random.choice(range(action_probs.shape[1]), p=action_probs.ravel())
        new_observation, reward, done, info = self.env.step(action)
        return new_observation, action, reward, done

    def get_old_actor_prediction(self, state):
        action_probs = self.old_actor_predict_only.predict(np.array(state), batch_size=self.batch_size)
        return action_probs

    def optimize_model(self):
        minibatch = self.sample_from_memory()
        states = []
        v_targets = []
        advantages = []

        # update V targets
        for idx, (state, act, rew, done, next_state) in enumerate(minibatch):
            states.append(state)
            action_one_hot = np.zeros(self.action_shape)
            curr_state_v_vals = self.critic.predict(np.expand_dims(np.asarray(list(state)), axis=0))
            next_state_v_value = self.critic.predict(np.expand_dims(np.asarray(list(next_state)), axis=0))

            if done:
                v_targets.append(rew)
                action_one_hot[act] = rew - curr_state_v_vals[0]
                advantages.append(action_one_hot)
            else:
                old_v = curr_state_v_vals[0].copy()
                curr_state_v_vals[0] = rew + self.discount_factor * next_state_v_value[0]
                action_one_hot[act] = curr_state_v_vals[0] - old_v
                advantages.append(action_one_hot)
                v_targets.append(curr_state_v_vals[0])

        # predict using old policy
        old_actor_prediction = self.get_old_actor_prediction(states)

        ac_input = [np.array(states),
                    np.array(old_actor_prediction),
                    np.array(advantages)]

        # fit models
        self.actor.fit(ac_input, batch_size=len(minibatch), epochs=self.epochs, verbose=0)
        self.critic.fit(np.asarray(states), np.asarray(v_targets), batch_size=len(minibatch), verbose=0)

        self.old_actor.set_weights(self.actor.get_weights())

    def train(self):
        self.actor, self.actor_predict_only = self.create_actor_model()
        self.old_actor, self.old_actor_predict_only = self.create_actor_model()
        self.create_critic_model()
        self.fill_empty_memory()
        total_reward = 0

        for ep in range(self.episodes):
            episode_rewards = []
            observation = self.env.reset()
            for step in range(self.max_steps):
                new_observation, action, reward, done = self.take_action(observation)
                reward = reward if not done else -100

                self.save_to_memory((observation, action, reward, done, new_observation))
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

            self.actor.save(self.model_path)

    def test(self):
        # test agent
        actor = models.load_model(self.model_path, compile=False)
        for i in range(self.test_episodes):
            observation = np.asarray(list(self.env.reset()))
            total_reward_per_episode = 0
            while True:
                self.env.render()
                action_probs = actor.predict(np.expand_dims(observation, axis=0))
                action = np.random.choice(range(action_probs.shape[1]), p=action_probs.ravel())
                new_observation, reward, done, info = self.env.step(action)
                total_reward_per_episode += reward
                observation = new_observation
                if done:
                    break
            self.test_rewards.append(total_reward_per_episode)

        print("Average reward for test agent: ", sum(self.test_rewards) / self.test_episodes)


if __name__ == '__main__':
    fire.Fire(ProximalPolicyOptimization)
