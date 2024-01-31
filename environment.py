import gymnasium as gym
from gymnasium.spaces import Discrete, Text, Tuple
import random

class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.prompt_starters = env_config.prompt_starters
        self.adversarial_model = env_config.adversarial_model
        
        self.action_space = Tuple(Discrete(env_config.rating_size), Text(env_config.min_response_length, env_config.max_response_length))
        self.observation_space = Text(env_config.min_response_length, env_config.max_response_length)
        self.reset(seed=env_config.worker_index * env_config.num_workers)
    
    def reset(self, seed, options):
        random.seed(seed)
        selected_option = random.choice(options)

        return selected_option, {}

    def step(self, action):
        return <obs>, <reward: float>, <terminated: bool>, <truncated: bool>, <info: dict>