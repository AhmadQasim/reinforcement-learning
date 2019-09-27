import random
import fire
from keras import models
from keras.layers import Input, Dense, concatenate, Flatten, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import gym
import numpy as np
import tensorflow as tf
from utils import OrnsteinUhlenbeckProcess

"""
Implementation of Deep Deterministic Policy Gradients on A2C with TD-0 value returns
"""


class DDPG:
    def __init__(self):
        self.env = gym.make('Pendulum-v0')
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.target_actor = None
        self.target_critic = None
        self.actor = None
        self.critic = None
        self.replay_buffer = []
        self.replay_buffer_size_thresh = 100000
        self.batch_size = 64
        self.episodes = 1000
        self.max_steps = 100
        self.gamma = 0.99
        self.test_episodes = 100
        self.discount_factor = 0.99
        self.test_rewards = []
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        self.epochs = 10
        self.tau = 1e-3
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.eps_decay = 0.005
        self.default_q_value_actor = -1
        self.noise = OrnsteinUhlenbeckProcess(size=self.action_shape)
        # range of the action possible for Pendulum-v0
        self.act_range = 2.0
        self.model_path = "models/DDPG.hdf5"

    def create_actor_model(self):
        input = Input(shape=self.state_shape)
        q_value = Input((1, ))

        fc1 = Dense(256, activation='relu', kernel_initializer="he_uniform")(input)
        x = GaussianNoise(1.0)(fc1)
        x = Dense(128, activation='relu')(x)
        x = GaussianNoise(1.0)(x)
        output = Dense(self.action_shape[0], activation='linear', kernel_initializer='he_uniform')(x)

        model = Model(inputs=[input, q_value], outputs=output)
        model.add_loss(self.actor_loss(q_value, output))
        model.compile(optimizer=Adam(lr=self.actor_lr), loss=None)
        model.summary()

        return model

    @staticmethod
    def actor_loss(q_values, action):
        return -K.mean(q_values) * action / action

    def create_critic_model(self):
        state = Input(shape=self.state_shape)
        action = Input(shape=self.action_shape)

        fc1 = Dense(256, activation='relu', kernel_initializer="he_uniform")(state)
        concatenated = concatenate([fc1, action])
        fc2 = Dense(128, activation='relu', kernel_initializer="he_uniform")(concatenated)
        output = Dense(1, activation='linear', kernel_initializer='he_uniform')(fc2)

        model = Model(inputs=[state, action], outputs=output)
        model.compile(optimizer=Adam(lr=self.critic_lr), loss='mse')
        model.summary()

        return model

    def optimizer(self):
        action_gdts = K.placeholder(shape=(None, self.action_shape[0]))
        params_grad = tf.gradients(self.actor.output, self.actor.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.actor.trainable_weights)
        return K.function(inputs=[self.actor.input, action_gdts], outputs=[],
                          updates=[tf.train.AdamOptimizer(1e-3).apply_gradients(grads)])

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
            new_observation, action, reward, done = self.take_action(observation, use_epsilon=False)
            reward = reward if not done else -100
            self.save_to_memory((observation, action, reward, done, new_observation))
            if done:
                new_observation = self.env.reset()
            observation = new_observation

    def take_action(self, state, use_epsilon=True):
        action = self.actor.predict([np.expand_dims(state, axis=0),
                                     np.expand_dims(self.default_q_value_actor, axis=0)])
        action = action[0]
        new_observation, reward, done, info = self.env.step(action)
        return new_observation, action, reward, done

    def soft_update_weights(self, model, target_model):
        W, target_W = model.get_weights(), target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]

        return target_W

    def gradients(self, input):
        return self.action_grads(input)

    def optimize_model(self):
        minibatch = self.sample_from_memory()
        states = []
        q_targets = []
        actions = []
        q_values = []

        # update Q targets
        for idx, (state, act, rew, done, next_state) in enumerate(minibatch):
            states.append(state)
            actions.append(act)
            target_act = self.target_actor.predict([np.expand_dims(np.asarray(list(next_state)), axis=0),
                                                    np.expand_dims(self.default_q_value_actor, axis=0)])
            target_state_q_vals = self.target_critic.predict([np.expand_dims(np.asarray(list(next_state)), axis=0),
                                                              target_act])
            curr_state_q_vals = self.critic.predict([np.expand_dims(np.asarray(list(state)), axis=0), act])
            q_values.append(curr_state_q_vals[0])

            if done:
                q_targets.append(rew)
            else:
                curr_state_q_vals[0] = rew + self.discount_factor * target_state_q_vals[0]
                q_targets.append(curr_state_q_vals[0])

        # fit models
        #self.actor.fit([np.array(states), np.array(q_values)], batch_size=len(minibatch),
        #               verbose=0)
        grads = self.gradients([np.array(states), np.array(actions)])
        self.adam_optimizer([np.array(states), np.array(grads).reshape((-1, self.action_shape[0]))])
        self.critic.fit([np.array(states), np.array(actions)], np.array(q_targets), batch_size=len(minibatch),
                        verbose=0)

        self.target_actor.set_weights(self.soft_update_weights(self.actor, self.target_actor))
        self.target_critic.set_weights(self.soft_update_weights(self.critic, self.target_critic))

    def train(self):
        self.actor = self.create_actor_model()
        self.target_actor = self.create_actor_model()
        self.critic = self.create_critic_model()
        self.target_critic = self.create_critic_model()
        self.fill_empty_memory()
        total_reward = 0

        self.adam_optimizer = self.optimizer()
        self.action_grads = K.function([self.critic.input[0], self.critic.input[1]],
                                       K.gradients(self.critic.output, [self.critic.input[1]]))

        for ep in range(self.episodes):
            episode_rewards = []
            observation = self.env.reset()
            for step in range(self.max_steps):
                observation = np.squeeze(observation)
                new_observation, action, reward, done = self.take_action(observation)
                reward -= 100
                action = np.clip(action+self.noise.generate(step), -self.act_range, self.act_range)

                self.save_to_memory((observation, action, reward, done, new_observation))
                episode_rewards.append(reward)
                observation = new_observation
                self.optimize_model()

                self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * np.exp(-self.eps_decay * ep)

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
    fire.Fire(DDPG)
