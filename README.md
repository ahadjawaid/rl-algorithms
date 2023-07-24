# rl-algorithms
This package contains the reinforcement learning algorithms I implement to learn 

## Installation 
```
pip install git+https://github.com/ahadjawaid/rl-algorithms
```

or 

```
git clone https://github.com/ahadjawaid/rl-algorithms.git

cd rl-algorithms

pip install .
```

## Usage

### Dynamic Programming Algorithms
Currently we have the following dynamic programming algorithms:

`policy_iteration`, `value_iteration`

Here's an example of using the algorithms:

```python
from rl_algorithms.dp import value_iteration, policy_iteration
import gym

env = gym.make("FrozenLake-v1")

P = env.unwrapped.P

V, policy, _ = value_iteration(P)

V, policy, _ = policy_iteration(P)
```

### Multi-armed Bandits Algorithms

Currently we have the following exploration strategies:

`PureExploitation`, `PureExploration`, `EpsilonGreedy`, `EpsilonGreedyLinearDecay`, `EpsilonGreedyExponentialDecay`, `OptimisticInitialization`


You can use a strategy over a number episodes by doing the following:
```python
from rl_algorithms.mab import PureExploitation
import multi_armed_bandits
import gym

env = gym.make("2-Armed-Bandit")

strategy = PureExploitation(env, n_episodes=1000)

action_value, rewards = strategy.run()
```
```output
PureExploitation episodes: 100% 1000/1000 [00:00<00:00, 124364.11it/s]
```

And if you want a summarized results done for you. You can use the following method:

```python
from rl_algorithms.mab import PureExploitation
import multi_armed_bandits

PureExploitation.test_strategy(n_arms=2, seed=42, n_episodes=1000)
```
```output
PureExploitation episodes: 100% 1000/1000 [00:00<00:00, 85948.85it/s]
True action-value function: [-0.75275929  1.39196365]
Predicted Action-value function: [-1.36461784  1.39664592]
Average episode reward: 1.39
```
