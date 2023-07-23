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
```python
from rl_algorithms import dp
import gym

env = gym.make("FrozenLake-v1")

P = env.unwrapped.P

dp.value_iteration(P)
```
