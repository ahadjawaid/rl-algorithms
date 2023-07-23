import numpy as np
from tqdm.auto import tqdm
import gym
import multi_armed_bandits
from typing import Callable
from functools import partial
from gym import Env

class Strategy:
    def __init__(self, env: Env, n_episodes: int = 5000) -> None:
        self.env = env
        self.n_episodes = n_episodes
        self.n_actions = env.action_space.n
        self.name = self.__class__.__name__

        self.Q, self.N = np.zeros((self.n_actions)), np.zeros((self.n_actions))

    def _action_selection(self):
        raise Exception("Not Implemented _action_selection")
    
    def run(self, display=True, *args, **kwargs):
        self.env.reset()

        rewards = np.zeros((self.n_episodes, self.n_actions))

        for e in tqdm(range(self.n_episodes), desc=f"{self.name} episodes", leave=True, disable=(not display)):
            action = self._action_selection(*args, **kwargs)
            _, reward, _, _, _ = self.env.step(action)

            rewards[e] = reward

            self.N[action] += 1
            self.Q[action] += (reward - self.Q[action]) / self.N[action]  # Updates the average

        return self.Q, rewards

    @classmethod
    def test_strategy(cls, n_arms: int = 2, rounding_value: int = 2, seed: int = None, 
                      display_progress: bool = True, *args, **kwargs) -> None:  

        np.random.seed(seed)
        env = gym.make(f"{n_arms}-Armed-Bandit")

        strategy = cls(env=env)
        Q, rewards = strategy.run(display_progress, *args, **kwargs)

        true_actions = np.array(list(map(lambda bandit: bandit.mean, env.unwrapped.bandits)))

        print(f"True action-value function: {true_actions}")
        print(f"Predicted Action-value function: {Q}")
        print(f"Total Rewards: {round(np.sum(rewards), rounding_value)}")


class PureExploitation(Strategy):
    def _action_selection(self):
        return np.argmax(self.Q)
    

class PureExploration(Strategy):
    def _action_selection(self):
        return np.random.randint(self.n_actions)
    
    
class EpsilonGreedy(Strategy):
    def _action_selection(self, epsilon):
        action = np.argmax(self.Q)
        if np.random.random() > epsilon:
            action = np.random.randint(self.n_actions)

        return action