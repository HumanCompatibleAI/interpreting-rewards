import os
import sys
import importlib
from pathlib import Path
from itertools import product
import argparse

import h5py

import gym
import mazelab

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage
import torch as th
import torch.nn as nn

from tqdm import tqdm

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv, DummyVecEnv
from stable_baselines3.common.vec_env import VecTransposeImage

sys.path.insert(1, "../rl-baselines3-zoo")
import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils.utils import StoreDict
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams


parser = argparse.ArgumentParser()
parser.add_argument('--algo', help='RL Algorithm', default='ppo',
                        type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('--env', type=str, default="CartPole-v1", help='environment ID')
parser.add_argument('--exp-id', type=int, default=1, help="experiment ID")
parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=25000,
                        type=int)
parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
args = parser.parse_args()


########### Set Device ############
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
dtype = th.float32
th.set_default_dtype(dtype)
print("Using device: {}".format(device))


########### Set Params ############
env_id = args.env 
folder = args.log_folder
algo = args.algo
num_threads = -1
n_envs = 1
exp_id = args.exp_id
verbose = 1
no_render = False
deterministic = False
load_best = True
load_checkpoint = None
norm_reward = False
seed = 0
reward_log = ''
env_kwargs = None


# Sanity checks
if exp_id > 0:
    log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, exp_id))
else:
    log_path = os.path.join(folder, algo)
    
found = False
for ext in ['zip']:
    model_path = os.path.join(log_path, f'{env_id}.{ext}')
    found = os.path.isfile(model_path)
    if found:
        break

if load_best:
    model_path = os.path.join(log_path, "best_model.zip")
    found = os.path.isfile(model_path)

if load_checkpoint is not None:
    model_path = os.path.join(log_path, f"rl_model_{load_checkpoint}_steps.zip")
    found = os.path.isfile(model_path)

if not found:
    raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

if algo in ['dqn', 'ddpg', 'sac', 'td3']:
    n_envs = 1

set_random_seed(seed)

if num_threads > 0:
    if verbose > 1:
        print(f"Setting torch.num_threads to {num_threads}")
    th.set_num_threads(num_threads)

is_atari = 'NoFrameskip' in env_id

stats_path = os.path.join(log_path, env_id)
hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)
env_kwargs = {} if env_kwargs is None else env_kwargs

log_dir = reward_log if reward_log != '' else None

# env = create_test_env(env_id, n_envs=n_envs,
#                       stats_path=stats_path, seed=seed, log_dir=log_dir,
#                       should_render=not no_render,
#                       hyperparams=hyperparams,
#                       env_kwargs=env_kwargs)
env = gym.make(env_id)
obs_shape = env.reset().shape

model = ALGOS[algo].load(model_path, env=env)


database = h5py.File(f"rewards_{env_id}_ss.hdf5", 'a')

SAMPLES = args.n_timesteps
train = database.create_group('train')
inputs = train.create_dataset('inputs', (SAMPLES, 2, *obs_shape))
outputs = train.create_dataset('outputs', (SAMPLES, 1))
# zeros_inputs = train.create_dataset('zeros-inputs', (SAMPLES//2, 84, 84, 4))
# zeros_labels = train.create_dataset('zeros-labels', (SAMPLES//2, 1))
# ones_inputs = train.create_dataset('ones-inputs', (SAMPLES//2, 84, 84, 4))
# ones_labels = train.create_dataset('ones-labels', (SAMPLES//2, 1))
state = env.reset()
next_state = None
for i in tqdm(range(SAMPLES)):
    action, _states = model.predict(state, deterministic=False)
    next_state, reward, done, info = env.step(action)
    inputs[i] = np.array((state, next_state))
    outputs[i] = reward
    if done:
        state = env.reset()
    else:
         state = next_state
    
   
   
SAMPLES = args.n_timesteps // 2
test = database.create_group('test')
inputs = test.create_dataset('inputs', (SAMPLES, 2, *obs_shape))
outputs = test.create_dataset('outputs', (SAMPLES, 1))
# zeros_inputs = train.create_dataset('zeros-inputs', (SAMPLES//2, 84, 84, 4))
# zeros_labels = train.create_dataset('zeros-labels', (SAMPLES//2, 1))
# ones_inputs = train.create_dataset('ones-inputs', (SAMPLES//2, 84, 84, 4))
# ones_labels = train.create_dataset('ones-labels', (SAMPLES//2, 1))
state = env.reset()
next_state = None
for i in tqdm(range(SAMPLES)):
    action, _states = model.predict(state, deterministic=False)
    next_state, reward, done, info = env.step(action)
    inputs[i] = np.array((state, next_state))
    outputs[i] = reward
    if done:
        state = env.reset()
    else:
        state = next_state


database.close()


