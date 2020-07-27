
import sys
import pathlib
import logging

import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.spaces import Box
from gym.wrappers import AtariPreprocessing, FrameStack

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from drlhp import HumanPreferencesEnvWrapper
from drlhp.reward_predictor_core_network import net_cnn


class VanillaFrameStack(FrameStack):
    """Stacks frames but without using LazyFrame."""

    def __init__(self, env, n_stack=4):
        """ 
        Wraper for `env` which stacks `n_stack` frames.

        Args:
            env (gym.Env): Environment to wrap
            n_stack (int): Number of observations to stack
        """
        super(VanillaFrameStack, self).__init__(env, n_stack)
        self.observation_space = Box(0, 255, shape=[84, 84, 4], dtype=env.observation_space.dtype)

    def _get_observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        stack = np.array(list(self.frames))         # (4, 84, 84)
        stack = np.transpose(stack, axes=(1, 2, 0)) # (84, 84, 4)
        return stack


if __name__ == '__main__':
    save_dir = pathlib.Path() / 'drlhp_pong'
    if not save_dir.exists():
        save_dir.mkdir()
    env = gym.make('PongNoFrameskip-v4')
    env = AtariPreprocessing(env, frame_skip=1)
    obs = env.reset()
    print(type(obs), obs.shape)
    env = VanillaFrameStack(env, n_stack=4)
    obs = env.reset()
    print(type(obs), obs.shape)
    preferences_env = HumanPreferencesEnvWrapper(env=env,
                            reward_predictor_network=net_cnn,
                            segment_length=100,
                            synthetic_prefs=True,
                            log_dir="drlhp-pong/drlhp_logs/",
                            n_initial_training_steps=0)
    model = A2C('CnnPolicy', preferences_env, verbose=1, device='cpu')
    model.learn(total_timesteps=int(1e6), eval_env=env)
    model.save(save_dir / 'model.zip')

