import os
import sys
import importlib
from pathlib import Path
from itertools import product
import h5py
import random

import gym
import numpy as np
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage.transform
import torch as th
import torch.nn as nn

from tqdm.auto import tqdm

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage

sys.path.insert(1, "../rl-baselines3-zoo")
import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils.utils import StoreDict
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams

import interp
from interp.common.models import AtariRewardModel
from celluloid import Camera


########### Set Device ############
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
dtype = th.float32
th.set_default_dtype(dtype)
print("Using device: {}".format(device))

def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = scipy.ndimage.gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

def occlude(img, mask):
    assert img.shape[1:] == (84, 84, 4)
    img = np.copy(img)
    for k in range(4):
        I = img[0, :, :, k]
        img[0, :, :, k] = I*(1-mask) + scipy.ndimage.gaussian_filter(I, sigma=3)*mask
    return img

def compute_saliency_map(reward_model, obs, stride=5, radius=5):
    baseline = reward_model(obs).detach().cpu().numpy()
    scores = np.zeros((84 // stride + 1, 84 // stride + 1))
    for i in range(0, 84, stride):
        for j in range(0, 84, stride):
            mask = get_mask(center=(i, j), size=(84, 84), r=radius)
            obs_perturbed = occlude(obs, mask)
            perturbed_reward = reward_model(obs_perturbed).detach().cpu().numpy()
            scores[i // stride, j // stride] = 0.5 * np.abs(perturbed_reward - baseline) ** 2
    pmax = scores.max()
    scores = skimage.transform.resize(scores, output_shape=(210, 160))
    scores = scores.astype(np.float32)
#     return pmax * scores / scores.max()
    return scores / scores.max()

def add_saliency_to_frame(frame, saliency, channel=1):
#     def saliency_on_atari_frame(saliency, atari, fudge_factor, channel=2, sigma=0):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = saliency.max()
    I = frame.astype('uint16')
    I[:, :, channel] += (frame.max() * saliency).astype('uint16')
    I = I.clip(1,255).astype('uint8')
    return I

env_id = "BreakoutNoFrameskip-v4"
folder = "../agents"
algo = "ppo"
n_timesteps = 10000
num_threads = -1
n_envs = 1
exp_id = 1
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

env = create_test_env(env_id, n_envs=n_envs,
                      stats_path=stats_path, seed=seed, log_dir=log_dir,
                      should_render=not no_render,
                      hyperparams=hyperparams,
                      env_kwargs=env_kwargs)

model = ALGOS[algo].load(model_path, env=env, device=device)

obs = env.reset()

rm = AtariRewardModel(env, device)
rm.load_state_dict(th.load(f"../reward-models/BreakoutNoFrameskip-v4-reward_model.pt"))
rm = rm.to(device)


random.seed(0)
np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

breakout_images = []
obs = env.reset()
for _ in tqdm(range(120)):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
    sal = compute_saliency_map(rm, obs)
    screenshot = env.render(mode='rgb_array')
    image = add_saliency_to_frame(screenshot, sal)
    breakout_images.append(image)



env_id = "SeaquestNoFrameskip-v4"
folder = "../agents"
algo = "ppo"
n_timesteps = 10000
num_threads = -1
n_envs = 1
exp_id = 1
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

env = create_test_env(env_id, n_envs=n_envs,
                      stats_path=stats_path, seed=seed, log_dir=log_dir,
                      should_render=not no_render,
                      hyperparams=hyperparams,
                      env_kwargs=env_kwargs)

model = ALGOS[algo].load(model_path, env=env, device=device)

obs = env.reset()

rm = AtariRewardModel(env, device)
rm.load_state_dict(th.load(f"../reward-models/SeaquestNoFrameskip-v4-reward_model.pt"))
rm = rm.to(device)

random.seed(0)
np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

seaquest_images = []
obs = env.reset()
for _ in tqdm(range(122)):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
    sal = compute_saliency_map(rm, obs, radius=6, stride=6)
    screenshot = env.render(mode='rgb_array')
    image = add_saliency_to_frame(screenshot, sal)
    seaquest_images.append(image)


fig, ax = plt.subplots(1, 1, figsize=(5, 3.09375))
camera = Camera(fig)
ax.axis('off')
for img in breakout_images:
    ax.imshow(img)
    camera.snap()

animation = camera.animate(interval=80)
animation.save('../videos/breakout.mp4')

fig, ax = plt.subplots(1, 1, figsize=(5, 3.09375))
camera = Camera(fig)
ax.axis('off')
for img in seaquest_images[20:]:
    ax.imshow(img)
    camera.snap()

animation = camera.animate(interval=80)
animation.save('../videos/seaquest.mp4')

fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(5, 3.09375))
camera = Camera(fig)
ax1.axis('off')
ax2.axis('off')
for i in range(100):
    ax1.imshow(breakout_images[i])
    ax2.imshow(seaquest_images[i+20])
    camera.snap()

animation = camera.animate(interval=80)
animation.save('../videos/breakout-and-seaquest.mp4')




