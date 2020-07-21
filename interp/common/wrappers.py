
from abc import ABC, abstractmethod

import numpy as np
import scipy.ndimage

import torch as th

from stable_baselines3.common.type_aliases import GymStepReturn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper


breakout_mask = np.zeros((84, 84))
breakout_mask[1:7, 18:43] = 1
seaquest_mask = np.zeros((84, 84))
seaquest_mask[3:8, 38:56] = 1
pong_mask = np.zeros((84, 84))
pong_mask[0:9, 18:26] = 1
pong_mask[0:9, 60:68] = 1
spaceinvaders_mask = np.zeros((84, 84))
spaceinvaders_mask[3:9, 1:35] = 1
spaceinvaders_mask[3:9, 43:77] = 1
tennis_mask = np.zeros((84, 84))
tennis_mask[10:13, 18:37] = 1
tennis_mask[10:13, 50:72] = 1

supported_games = {}
    'Breakout':      {'mask': breakout_mask,      'background': 0},
    'Seaquest':      {'mask': seaquest_mask,      'background': 64},
    'Pong':          {'mask': pong_mask,          'background': 87},
    'SpaceInvaders': {'mask': spaceinvaders_mask, 'background': 0},
    'Tennis':        {'mask': tennis_mask,        'background': 95}
}


def get_id(env):
    """
    Attempt to determine the env id of `env`.
    """
    if getattr(env, 'spec', None):
        return env.spec.id
    if getattr(env, 'envs', None):
        for subenv in env.envs:
            subenv_id = get_id(subenv)
            if subenv_id:
                return subenv_id
    if getattr(env, 'unwrapped', None):
        if env.unwrapped is not env:
            return get_id(env.unwrapped)
    return None


def game_from_id(name: str) -> str:
    for game_type in supported_games:
        if game_type in name:
            return game_type


class ScoreMaskerWrapper(VecEnvWrapper, ABC):
    """
    Base class for wrappers which modify a masked region of an Atari observation.
    """

    def __init__(self, venv):
        assert venv.observation_space.shape == (84, 84, 4)
        super(ScoreMaskerWrapper, self).__init__(venv)
        env_id = get_id(venv)
        if env_id and game_from_id(env_id):
            self.game = game_from_id(env_id)
        else:
            raise Exception("Environment not yet supported.")
        self.mask = supported_games[self.game]['mask']
        self.background = supported_games[self.game]['background']

    @abstractmethod
    def modify_obs(self, obs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def modify_rgb(self, rgb_array: np.ndarray) -> np.ndarray:
        pass

    def step_wait(self) -> 'GymStepReturn':
        observations, rewards, dones, infos = self.venv.step_wait()
        return self.modify_obs(observations), rewards, dones, infos

    def reset(self) -> np.ndarray:
        return self.modify_obs(self.venv.reset())

    def render(self, mode: str = 'human'):
        if mode == 'rgb_array':
            return self.modify_rgb(self.venv.render(mode))
        else:
            self.venv.render(mode)


class BlurAtariScore(ScoreMaskerWrapper):
    """
    Gaussian-blur out the score of an Atari Game.
    """

    def __init__(self, venv):
        super(BlurAtariScore, self).__init__(venv)

    def modify_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Blur out the score in `obs`.
        """
        blurred_obs = np.zeros(obs.shape)
        for n in range(obs.shape[0]):
            for i in range(4):
                frame = obs[n, :, :, i]
                A = scipy.ndimage.gaussian_filter(frame, 3)
                blurred = frame * (1 - self.mask) + (A * self.mask)
                blurred_obs[n, :, :, i] = blurred
        return blurred_obs


class RemoveAtariScore(ScoreMaskerWrapper):
    """
    Completely removes score by setting pixels to the background color.
    """
    
    def __init__(self, venv):
        super(BlurAtariScore, self).__init__(venv)

    def modify_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Remove the score in `obs`.
        """
        modified_obs = np.copy(obs)
        for n in range(obs.shape[0]):
            for i in range(4):
                frame = obs[n, :, :, i]
                modified_obs[n, :, :, i] = frame * (1 - self.mask) + (self.mask * self.background)
        return modified_obs


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
