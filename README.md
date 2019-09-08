# RL
Self-study and implementation of Deep Reinforcement Learning algorithms with a friend.

The following algorithms can be found in the repo:

1. Tabular Q-Learning
2. Deep Q-Learning
3. REINFORCE (Vanilla Policy Gradient with Monte Carlo returns)
4. Advantage Actor Critic (A2C)
5. Proximal Policy Optimization (PPO)

### Usage
1. Install dependencies using `pip3 install -r requirements.txt`
2. Each script has `train` and `test` methods. To call them, do
`python3 <script_name> <method_name>`. For example: `python3 REINFORCE.py train`
3. The `test` method will load a model from the `models` directory. Pre-trained models 
for some algorithms can be found in this repo.