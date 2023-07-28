import numpy as np
from tqdm.auto import tqdm
import gym
import multi_armed_bandits
from gym import Env


class Strategy:
    def __init__(self, env: Env, n_episodes: int = 5000, *args, **kwargs) -> None:
        self.env = env
        self.n_episodes = n_episodes
        self.n_actions = env.action_space.n
        self.name = self.__class__.__name__

        self.Q, self.N = np.zeros((self.n_actions)), np.zeros((self.n_actions))

    def _action_selection(self, *args, **kwargs):
        raise Exception("Not Implemented _action_selection")
    
    def run(self, display=True, *args, **kwargs):
        self.env.reset()

        rewards = np.zeros((self.n_episodes))

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
        env = gym.make(f"{n_arms}-armed-bandit")

        strategy = cls(*args, env=env, **kwargs)
        Q, rewards = strategy.run(display_progress, *args, **kwargs)

        true_actions = np.array(list(map(lambda bandit: bandit.mean, env.unwrapped.bandits)))

        print(f"True action-value function: {true_actions}")
        print(f"Predicted Action-value function: {Q}")
        print(f"Average episode reward: {round(np.sum(rewards) / strategy.n_episodes, rounding_value)}")



class PureExploitation(Strategy):
    def _action_selection(self, *args, **kwargs):
        return np.argmax(self.Q)
    

class PureExploration(Strategy):
    def _action_selection(self, *args, **kwargs):
        return np.random.randint(self.n_actions)
    
    
class EpsilonGreedy(Strategy):
    def _action_selection(self, epsilon, *args, **kwargs):
        action = np.argmax(self.Q)
        if np.random.random() > epsilon:
            action = np.random.randint(self.n_actions)

        return action


class EpsilonGreedyLinearDecay(Strategy):
    def __init__(self, env: Env, n_episodes: int = 5000, init_epsilon=1.0, min_epsilon=0.05, decay_ratio=0.05) -> None:
        super().__init__(env, n_episodes)

        self.decay_steps = iter(linear_decay(self.n_episodes, decay_ratio, init_epsilon, min_epsilon))

    def _action_selection(self, *args, **kwargs):
        action = np.argmax(self.Q)
        if np.random.random() > next(self.decay_steps):
            action = np.random.randint(self.n_actions)

        return action
    

class EpsilonGreedyExponentialDecay(Strategy):
    def __init__(self, env: Env, n_episodes: int = 5000, init_epsilon=1.0, min_epsilon=0.05, decay_ratio=0.05) -> None:
        super().__init__(env, n_episodes)

        self.decay_steps = iter(exponential_decay(n_episodes, decay_ratio , init_epsilon, min_epsilon))

    def _action_selection(self, *args, **kwargs):
        action = np.argmax(self.Q)
        if np.random.random() > next(self.decay_steps):
            action = np.random.randint(self.n_actions)

        return action
    

class OptimisticInitialization(PureExploitation):
    def __init__(self, env: Env, n_episodes: int = 5000, optimisitic_value: float = 1.0, *args, **kwargs) -> None:
        super().__init__(env, n_episodes, *args, **kwargs)

        self.Q = np.full((self.n_actions), optimisitic_value, np.float64)
        self.N = np.full((self.n_actions), optimisitic_value, np.float64)


class Softmax(Strategy):
    def __init__(self, env: Env, n_episodes: int = 5000, init_temp=1.0, min_temp=0.05, decay_ratio=0.05):
        super().__init__(env, n_episodes)

        self.temperature_values = iter(linear_decay(n_episodes, decay_ratio, init_temp, min_temp))

    def _action_selection(self, *args, **kwargs):
        temperature = next(self.temperature_values)
        scaled_Q = self.Q / temperature
        normalized_Q = scaled_Q - np.max(scaled_Q)
        probs = np.exp(normalized_Q) / np.sum(np.exp(normalized_Q))

        assert np.isclose(probs.sum(), 1.0)
        
        action = np.random.choice(a=np.arange(len(probs)), size=1, p=probs)
        return int(action)


class UpperConfidenceBound(Strategy):
    def __init__(self, env: Env, n_episodes: int = 5000, confidence_value: float = 2., *args, **kwargs) -> None:
        super().__init__(env, n_episodes)
        self.confidence_value = confidence_value
        self.episode = 0

    def _action_selection(self, *args, **kwargs):
        # Goes over all actions first to avoid divide by zero
        if self.episode < self.n_actions:
            action = self.episode
        else:
            uncertainty_bonus = self.confidence_value * np.sqrt(np.log(self.episode) / self.N)

            action = np.argmax(self.Q + uncertainty_bonus)

        self.episode += 1
        return action

def linear_decay(n_episodes, decay_ratio, init_value, min_value):
    decay_episodes = int(n_episodes * decay_ratio)

    epsilon = init_value
    decay_step = (init_value - min_value) / decay_episodes
    for e in range(n_episodes):
        if e < decay_episodes:
            epsilon -= decay_step

        yield epsilon

def exponential_decay(n_episodes, decay_ratio, init_value, min_value):
    decay_episodes = int(n_episodes * decay_ratio)
    rem_episodes = n_episodes - decay_episodes
    epsilons = 0.01
    epsilons /= np.logspace(-2, 0, decay_episodes) 
    epsilons *= init_value - min_value 
    epsilons += min_value
    epsilons = np.pad(epsilons, (0, rem_episodes), 'edge')

    return epsilons