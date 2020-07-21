
from abc import ABC, abstractmethod

import numpy as np
import scipy.ndimage

import torch as th

from stable_baselines3.common.type_aliases import GymStepReturn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper


breakout_obs_mask = np.zeros((84, 84))
breakout_obs_mask[1:6, 18:43] = 1
breakout_render_mask = np.zeros((210, 160, 3))
breakout_render_mask[4:16, 35:81, :] = 1

seaquest_obs_mask = np.zeros((84, 84))
seaquest_obs_mask[3:8, 38:56] = 1
seaquest_render_mask = np.zeros((210, 160, 3))
seaquest_render_mask[7:19, 70:120, :] = 1

pong_obs_mask = np.zeros((84, 84))
pong_obs_mask[0:9, 10:34] = 1
pong_obs_mask[0:9, 52:76] = 1
pong_render_mask = np.zeros((210, 160, 3))
pong_render_mask[0:22, 15:145, :] = 1

spaceinvaders_obs_mask = np.zeros((84, 84))
spaceinvaders_obs_mask[3:9, 1:35] = 1
spaceinvaders_obs_mask[3:9, 43:77] = 1
spaceinvaders_render_mask = np.zeros((210, 160, 3))
spaceinvaders_render_mask[8:22, 0:70, :] = 1
spaceinvaders_render_mask[8:22, 80:150, :] = 1

tennis_obs_mask = np.zeros((84, 84))
tennis_obs_mask[10:13, 18:37] = 1
tennis_obs_mask[10:13, 50:72] = 1
tennis_render_mask = np.zeros((250, 160, 3))
tennis_render_mask[30:39, 30:72, :] = 1
tennis_render_mask[30:39, 94:136, :] = 1

supported_games = {
    'Breakout': {
        'obs_mask': breakout_obs_mask,
        'obs_background': 0,
        'render_mask': breakout_render_mask,
        'render_background': np.array([0, 0, 0]) 
    },
    'Seaquest': {
        'obs_mask': seaquest_obs_mask,
        'obs_background': 64,
        'render_mask': seaquest_render_mask,
        'render_background': np.array([ 45,  50, 184])
    },
    'Pong': {
        'obs_mask': pong_obs_mask,
        'obs_background': 87,
        'render_mask': pong_render_mask,
        'render_background': np.array([144,  72,  17])
    },
    'SpaceInvaders': {
        'obs_mask': spaceinvaders_obs_mask, 
        'obs_background': 0,
        'render_mask': spaceinvaders_render_mask,
        'render_background': np.array([0, 0, 0])
    },
    'Tennis': {
        'obs_mask': tennis_obs_mask,        
        'obs_background': 95,
        'render_mask': tennis_render_mask,
        'render_background': np.array([168,  48, 143])
    }
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
    Base class for wrappers which modify a the region of an Atari observation containing the game score.
    """

    def __init__(self, venv):
        assert len(venv.observation_space.shape) == 3
        assert venv.observation_space.shape[0:2] == (84, 84)
        super(ScoreMaskerWrapper, self).__init__(venv)
        env_id = get_id(venv)
        if env_id and game_from_id(env_id):
            self.game = game_from_id(env_id)
        else:
            raise Exception("Environment not yet supported.")
        self.obs_mask = supported_games[self.game]['obs_mask']
        self.obs_background = supported_games[self.game]['obs_background']
        self.render_mask = supported_games[self.game]['render_mask']
        self.render_background = supported_games[self.game]['render_background']

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
            for i in range(obs.shape[-1]):
                frame = obs[n, :, :, i]
                A = scipy.ndimage.gaussian_filter(frame, 3)
                blurred = frame * (1 - self.obs_mask) + (A * self.obs_mask)
                blurred_obs[n, :, :, i] = blurred
        return blurred_obs

    def modify_rgb(self, rgb_array: np.ndarray) -> np.ndarray:
        A = scipy.ndimage.gaussian_filter(rgb_array, 3)
        blurred_array = rgb_array * (1 - self.render_mask) + (A * self.render_mask)
        return blurred_array.astype(np.uint8)


class RemoveAtariScore(ScoreMaskerWrapper):
    """
    Completely removes score by setting pixels to the background color.
    """
    
    def __init__(self, venv):
        super(RemoveAtariScore, self).__init__(venv)

    def modify_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Remove the score in `obs`.
        """
        modified_obs = np.copy(obs)
        for n in range(obs.shape[0]):
            for i in range(obs.shape[-1]):
                frame = obs[n, :, :, i]
                modified_obs[n, :, :, i] = frame * (1 - self.obs_mask) + (self.obs_mask * self.obs_background)
        return modified_obs

    def modify_rgb(self, rgb_array: np.ndarray) -> np.ndarray:
        modified_array = rgb_array * (1 - self.render_mask) + (self.render_mask * self.render_background)
        return modified_array.astype(np.uint8)

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
