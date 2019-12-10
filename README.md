# Reinforcement Learning
Self-study and implementations of deep reinforcement learning papers/algorithms with a friend.

The following algorithms can be found in the repo:

1. **Tabular Q-Learning**
2. **Deep Q-Learning**  
[Paper: [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)]
3. **REINFORCE (Vanilla Policy Gradient with Monte Carlo returns)**
4. **Advantage Actor Critic (A2C)**  
[Paper: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)]
5. **Proximal Policy Optimization (PPO)**  
[Paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)]
6. **Deep Deterministic Policy Gradients (DDPG)**  
[Paper: [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)]
7. **Dynamics Randomization for RL Transfer Learning**  
[Paper: [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/pdf/1710.06537.pdf)]


### Usage
1. Install dependencies using `pip3 install -r requirements.txt`
2. Each script has `train` and `test` methods. To call them, do
`python3 <script_name> <method_name>`. For example: `python3 REINFORCE.py train`
3. The `test` method will load a model from the `models` directory. Pre-trained models 
for some algorithms can be found in this repo.
