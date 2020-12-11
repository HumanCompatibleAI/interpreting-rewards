
from abc import ABC, abstractmethod

import numpy as np
import scipy.ndimage

import torch as th

import gym
from stable_baselines3.common.type_aliases import GymStepReturn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper

from .models import MazeRewardModel, AtariRewardModel


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


class CustomRewardVecWrapper(VecEnvWrapper):
    """
    Wrapper for overriding the environment reward with a custom reward function.
    """

    def __init__(self, venv, reward_function):
        super(CustomRewardVecWrapper, self).__init__(venv)
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


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_function):
        super(CustomRewardWrapper, self).__init__(env)
        self.reward_function = reward_function

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        custom_reward = self.reward_function(next_state)
        if type(custom_reward) is th.Tensor:
            custom_reward = custom_reward.cpu().detach().numpy()
        if len(custom_reward.shape) == 2:
            custom_reward = custom_reward.flatten()
        elif len(custom_reward.shape) == 1:
            pass
        else:
            raise Exception("Weirdly shaped reward from custom reward function")
        return observations, custom_reward, dones, infos


class CustomRewardSSVecWrapper(VecEnvWrapper):
    """
    Overrides environment reward with a given reward function R(s, s').
    """

    def __init__(self, venv, reward_function):
        super(CustomRewardSSVecWrapper, self).__init__(venv)
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


class CustomRewardSSWrapper(gym.Wrapper):
    """
    Overrides environment reward with a given reward function R(s, s').
    """

    def __init__(self, env, reward_function):
        super(CustomRewardSSWrapper, self).__init__(env)
        self.reward_function = reward_function
        self.prev_obs = None

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # if done:
        #     reward_input = np.array((self.prev_obs, info['terminal_observation'])).astype(np.float32)
        #     reward_input = np.expand_dims(reward_input, axis=0)
        #     custom_reward = self.reward_function(reward_input)
        # else:
        reward_input = np.array((self.prev_obs, next_state)).astype(np.float32)
        reward_input = np.expand_dims(reward_input, axis=0)
        custom_reward = self.reward_function(reward_input)
        self.prev_obs = next_state
        return next_state, custom_reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.prev_obs = obs
        return obs


def WrapWithRewardModel(env, model_path, device='cuda'):
    if 'Maze' in model_path:
        rm = MazeRewardModel(env, device)
        rm.load_state_dict(th.load(model_path))
        return CustomRewardSSWrapper(env, rm)
    elif 'NoFrameskip' in model_path:
        rm = AtariRewardModel(env, device)
        rm.load_state_dict(th.load(model_path))
        return CustomRewardWrapper(env, rm)
    else:
        raise Exception("Only Maze or Atari environments are supported by this 'wrapper'.")

