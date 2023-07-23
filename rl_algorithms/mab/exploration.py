import numpy as np
from tqdm.auto import tqdm
import gym

def pure_exploitation(env: gym.Env, n_episodes: int = 5000):
    # Since there is no state information in a Multi Armed Bandit (MAB) we only need to initalize actions
    env.reset()\
    
    Q, N = np.zeros((env.action_space.n)), np.zeros((env.action_space.n))

    for _ in tqdm(range(n_episodes), desc=f"Pure exploration episodes", leave=True):
        action = np.argmax(Q)
        _, reward, _, _, _ = env.step(action)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]  # Updates the average

    return Q

