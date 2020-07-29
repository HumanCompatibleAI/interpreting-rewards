
import sys
import pathlib
import os.path as osp
import logging
import glob

import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver

import gym
from gym.spaces import Box
from gym.wrappers import AtariPreprocessing, FrameStack

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from drlhp import HumanPreferencesEnvWrapper
from drlhp.reward_predictor_core_network import net_cnn

from interp.utils import get_latest_run_id, AtariFrameStack

ex = Experiment()


ALGOS = {
    "a2c": A2C,
    "ppo": PPO,
    "dqn": DQN
}

@ex.config
def my_config():
    env_id = "PongNoFrameskip-v4"        # the gym environment to use
    algo = "ppo"                         # the RL algorithm to use
    timesteps = int(1e7)                 # number of timesteps to train the policy on
    synthetic_preferences = True         # whether to use synthetic preferences
    segment_length = 40                  # the length of the segments to be compared
    device = get_device()                # the device to put/train the SB3 model on
    eval_freq = timesteps // 1000        # frequency to evaluate a SB3 model and possibly save new best


@ex.main
def run(
        env_id,
        algo,
        timesteps,
        synthetic_preferences,
        segment_length,
        device,
        eval_freq
):
    if 'NoFrameskip' not in env_id:
        raise Exception("Currently, only Atari environments are supported.")
    if algo not in ALGOS:
        raise Exception(f"Algorithm {algo} not recognized or yet supported.")
    save_dir = osp.join("output", 'drlhp', algo)
    save_dir = osp.join(save_dir, f"{env_id}_{get_latest_run_id(save_dir, env_id) + 1}")
    print(f"Saving to {save_dir}")
    env = gym.make(env_id)
    env = AtariPreprocessing(env, frame_skip=1)
    env = AtariFrameStack(env, n_stack=4)
    preferences_env = HumanPreferencesEnvWrapper(env=env,
                            reward_predictor_network=net_cnn,
                            segment_length=segment_length,
                            synthetic_prefs=synthetic_preferences,
                            log_dir=save_dir,
                            n_initial_training_steps=5,
                            reward_predictor_ckpt_interval=5)
    model = ALGOS[algo]('CnnPolicy', preferences_env, verbose=1, device=device)
    dummy_model = ALGOS[algo]('CnnPolicy', env)
    eval_callback = EvalCallback(dummy_model.env, 
                             best_model_save_path=save_dir,
                             eval_freq=eval_freq,
                             deterministic=False, 
                             render=False)
    model.learn(total_timesteps=timesteps, eval_env=env, callback=eval_callback)
    model.save(save_dir / 'model.zip')


def main_console():
    observer = FileStorageObserver.create(osp.join("output", "sacred"))
    ex.observers.append(observer)
    ex.run_commandline()

if __name__ == "__main__":
    main_console()

