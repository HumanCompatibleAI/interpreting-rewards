
from abc import ABC, abstractmethod

import numpy as np
import scipy.ndimage

import gym
from gym.envs.atari import AtariEnv

supported_games = {
    'breakout': {
        'regions': [
            (
                (4, 16), # row range to modify 
                (35, 81) # column range to modify
            )
        ],
        'color': np.array([0, 0, 0]),
    },
    'seaquest': {
        'regions': [
            (
                (7, 19), 
                (70, 120)
            )
        ],
        'color': np.array([ 45,  50, 184])
    },
    'pong': {
        'regions': [
            (
                (0, 22), 
                (15, 145)
            )
        ],
        'color': np.array([144,  72,  17])
    },
    'space_invaders': {
        'regions': [
            (
                (8, 22), 
                (0, 70)
            ), 
            (
                (8, 22), 
                (80, 150)
            )
        ],
        'color': np.array([0, 0, 0])
    },
    'tennis': {
        'regions': [
            (
                (30, 39),
                (30, 72)
            ),
            (
                (30, 39), 
                (94, 136)
            ),
        ],
        'color': np.array([168,  48, 143])
    }
}


class ScoreMaskerWrapper(gym.Wrapper, ABC):
    """Base class for wrappers which modify the region displaying the
    game score of an Atari observation."""

    def __init__(self, env):
        assert env is AtariEnv
        assert env.game in supported_games
        super(ScoreMaskerWrapper, self).__init__(env)
        self.mask = np.zeros((210, 160, 3))
        for ((r0, r1), (c0, c1)) in supported_games[self.env.game]['regions']:
            self.mask[r0:r1, c0:c1, :] = 1

    @abstractmethod
    def modify_image(self, image: np.ndarray) -> np.ndarray:
        del image
    
    def _get_image(self):
        return self.modify_image(self.env._get_image())


class BlurAtariScore(ScoreMaskerWrapper):
    def __init__(self, env):
        super(BlurAtariScore, self).__init__(env)
    
    def modify_image(self, image: np.ndarray) -> np.ndarray:
        A = scipy.ndimage.gaussian_filter(image, 3)
        blurred_array = image * (1 - self.mask) + (A * self.mask)
        return blurred_array.astype(np.uint8)


def RemoveAtariScore(ScoreMaskerWrapper):
    def __init__(self, env):
        super(BlurAtariScore, self).__init__(env)

    def modify_image(self, image: np.ndarray) -> np.ndarray:
        return image * (1 - self.mask) + (self.mask * supported_games[self.env.game]['color'])





