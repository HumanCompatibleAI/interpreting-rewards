
from abc import ABC, abstractmethod

import numpy as np
import scipy.ndimage

import torch as th

from stable_baselines3.common.type_aliases import GymStepReturn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper


class DummyWrapper(VecEnvWrapper):
    """
    Wraps the venv but does absolutely nothing.
    """

    def __init__(self, venv):
        super(DummyWrapper, self).__init__(venv)

    def step_wait(self) -> 'GymStepReturn':
        observations, rewards, dones, infos = self.venv.step_wait()
        return observations, rewards, dones, infos

    def reset(self) -> np.ndarray:
        return self.venv.reset()


class CustomRewardWrapper(VecEnvWrapper):
    """
    Wrapper for overriding the environment reward with a custom reward function.
    """

    def __init__(self, venv, reward_function):
        super(CustomRewardWrapper, self).__init__(venv)
        self.reward_function = reward_function

    def step_wait(self) -> 'GymStepReturn':
        observations, rewards, dones, infos = self.venv.step_wait()
        custom_rewards = self.reward_function(observations)
        if type(custom_rewards) is th.Tensor:
            custom_rewards = custom_rewards.cpu().detach().numpy()
        if len(custom_rewards.shape) == 2:
            custom_rewards = custom_rewards.flatten()
        elif len(custom_rewards.shape) == 1:
            pass
        else:
            raise Exception("Weirdly shaped reward from custom reward function")
        return observations, custom_rewards, dones, infos

    def reset(self) -> np.ndarray:
        return self.venv.reset()


class CustomRewardSSWrapper(VecEnvWrapper):
    """
    Overrides environment reward with a given reward function R(s, s').
    """

    def __init__(self, venv, reward_function):
        super(CustomRewardSSWrapper, self).__init__(venv)
        self.reward_function = reward_function
        self.prev_obs = None

    def step_wait(self) -> 'GymStepReturn':
        obs, rewards, dones, infos = self.venv.step_wait()
        custom_rewards = []
        for k in range(len(rewards)):
            if dones[k]:
                reward_input = np.array((self.prev_obs[k], infos[k]['terminal_observation'])).astype(np.float32)
                reward_input = np.expand_dims(reward_input, axis=0)
                custom_rewards.append(self.reward_function(reward_input))
            else:
                reward_input = np.array((self.prev_obs[k], obs[k])).astype(np.float32)
                reward_input = np.expand_dims(reward_input, axis=0)
                custom_rewards.append(self.reward_function(reward_input))
        custom_rewards = np.array(custom_rewards)
        if type(custom_rewards) is th.Tensor:
            custom_rewards = custom_rewards.cpu().detach().numpy()
        if len(custom_rewards.shape) == 2:
            custom_rewards = custom_rewards.flatten()
        elif len(custom_rewards.shape) == 1:
            pass
        else:
            raise Exception("Weirdly shaped reward from custom reward function")
        self.prev_obs = obs
        return obs, custom_rewards, dones, infos

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        self.prev_obs = obs
        return obs
