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
        'sample_background_loc': (10, 10)
    },
    'seaquest': {
        'shape': (210, 160, 3),
        'regions': [
            (
                (7, 19), 
                (70, 120)
            )
        ],
        'sample_background_loc': (8, 65)
    },
    'pong': {
        'shape': (210, 160, 3),
        'regions': [
            (
                (0, 22), 
                (15, 145)
            )
        ],
        'sample_background_loc': (10, 10)
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
        'sample_background_loc': (7, 7)
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
        'sample_background_loc': (31, 29)
    }
}


class AtariEnvModifiedScoreBase(AtariEnv, ABC):
    """
    Base, abstract class for Atari environments which obscure their score in some way.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get('obs_type') and kwargs['obs_type'] == 'ram':
            raise Exception("Only image-based observations can have their score obscurred.")
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
            game (str): The name of the Atari game, in lowercase. Ex: 'breakout'
            mode: As far as I can tell, this parameter is never used. Defaults to None.
            difficulty: Also appears to not be used. Defaults to None.
            obs_type (str): 'image' or 'ram'
            frameskip (tuple or int): Set to 1 for NoFrameskip. 
            repeat_action_probability (float): Does what you think.
            full_action_space (bool): whether to use the full action space.
        """
        super(AtariEnvNoScore, self).__init__(*args, **kwargs)

    def _get_image(self):
        image = self.ale.getScreenRGB2()
        sx, sy = supported_games[self.game]['sample_background_loc']
        color = image[sx, sy]
        image = image * (1 - self.mask) + (self.mask * color)
        return image.astype(np.uint8)


class AtariEnvBlurScore(AtariEnvModifiedScoreBase):
    """
    An Atari Environment with the score blurred out.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Create Atari Environment with the score blurred out.

        Args:
            game (str): The name of the Atari game, in lowercase. Ex: 'breakout'
            mode: As far as I can tell, this parameter is never used. Defaults to None.
            difficulty: Also appears to not be used. Defaults to None.
            obs_type (str): 'image' or 'ram'
            frameskip (tuple or int): Set to 1 for NoFrameskip. 
            repeat_action_probability (float): Does what you think. Defaults to 0.
            full_action_space (bool): whether to use the full action space. Defaults to False.
        """
        super(AtariEnvBlurScore, self).__init__(*args, **kwargs)

    def _get_image(self):
        image = self.ale.getScreenRGB2()
        A = scipy.ndimage.gaussian_filter(image, sigma=(3, 3, 0))
        image = image * (1 - self.mask) + (A * self.mask)
        return image.astype(np.uint8)



