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


class AtariEnvNoScore(AtariEnv):
    """
    An Atari Environment without the score displayed.
    """
    
    def __init__(self, *args, **kwargs):
        super(AtariEnvNoScore, self).__init__(*args, **kwargs)
        assert self.game in supported_games
        self.mask = np.zeros((210, 160, 3))
        for ((r0, r1), (c0, c1)) in supported_games[self.env.game]['regions']:
            self.mask[r0:r1, c0:c1, :] = 1

    @override
    def _get_image(self):
        image = self.ale.getScreenRGB2()
        return image * (1 - self.mask) + (self.mask * supported_games[self.env.game]['color'])

