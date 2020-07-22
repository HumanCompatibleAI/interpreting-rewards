from abc import ABC, abstractmethod

import numpy as np
import scipy.ndimage

import gym
from gym.envs.atari import AtariEnv

supported_games = {
    'breakout': {
        'shape': (210, 160, 3),
        'regions': [
            (
                (4, 16), # row range to modify 
                (35, 81) # column range to modify
            )
        ],
        'color': np.array([0, 0, 0]),
    },
    'seaquest': {
        'shape': (210, 160, 3),
        'regions': [
            (
                (7, 19), 
                (70, 120)
            )
        ],
        'color': np.array([ 45,  50, 184])
    },
    'pong': {
        'shape': (210, 160, 3),
        'regions': [
            (
                (0, 22), 
                (15, 145)
            )
        ],
        'color': np.array([109, 118,  43])
    },
    'space_invaders': {
        'shape': (210, 160, 3),
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
        'shape': (250, 160, 3),
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


class AtariEnvModifiedScoreBase(AtariEnv, ABC):
    """
    Base, abstract class for Atari environments which obscure their score in some way.
    """

    def __init__(self, *args, **kwargs):
        super(AtariEnvModifiedScoreBase, self).__init__(*args, **kwargs)
        assert self.game in supported_games
        self.mask = np.zeros(supported_games[self.game]['shape'])
        for ((r0, r1), (c0, c1)) in supported_games[self.game]['regions']:
            self.mask[r0:r1, c0:c1, :] = 1

    @abstractmethod
    def _get_image(self):
        pass

class AtariEnvNoScore(AtariEnvModifiedScoreBase):
    """
    An Atari Environment without the score displayed.
    """
   
    def __init__(self, *args, **kwargs):
        """
        Create Atari Environment without the score displayed.

        Args:
            game (str)
            mode=None
            difficulty=None,
            obs_type='ram',
            frameskip=(2, 5),
            repeat_action_probability=0.,
            full_action_space=False):
        """
        super(AtariEnvNoScore, self).__init__(*args, **kwargs)

    def _get_image(self):
        image = self.ale.getScreenRGB2()
        image = image * (1 - self.mask) + (self.mask * supported_games[self.game]['color'])
        return image.astype(np.uint8)


class AtariEnvBlurScore(AtariEnvModifiedScoreBase):
    """
    An Atari Environment with the score blurred out.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Create Atari Environment with the score blurred out.

        Args:
            game (str)
            mode=None
            difficulty=None,
            obs_type='ram',
            frameskip=(2, 5),
            repeat_action_probability=0.,
            full_action_space=False):
        """
        super(AtariEnvBlurScore, self).__init__(*args, **kwargs)

    def _get_image(self):
        image = self.ale.getScreenRGB2()
        A = scipy.ndimage.gaussian_filter(image, 3)
        image = image * (1 - self.mask) + (A * self.mask)
        return image.astype(np.uint8)



