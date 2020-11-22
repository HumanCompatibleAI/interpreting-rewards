import argparse
import difflib
import os
import sys
import importlib
import time
import uuid
import warnings
from collections import OrderedDict
from pprint import pprint

from tqdm.auto import tqdm
import h5py

import yaml
import gym
import numpy as np
import torch as th
# For custom activation fn
import torch.nn as nn  # noqa: F401 pytype: disable=unused-import

from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import constant_fn, get_device
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import A2C

import mazelab

sys.path.append('../rl-baselines3-zoo/')
# Register custom envs
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class
from utils.hyperparams_opt import hyperparam_optimization
from utils.callbacks import SaveVecNormalizeCallback
from utils.noise import LinearNormalActionNoise
from utils.utils import StoreDict, get_callback_class

class RewardData(th.utils.data.Dataset):
    def __init__(self, env_id, train=True):
        self.f = h5py.File(f"../datasets/rewards_{env_id}.hdf5", 'r')
        if train:
            self.group = self.f['train']
        else:
            self.group = self.f['test']
    
    def __getitem__(self, k):
        input = self.group['inputs'][k]
        output = self.group['outputs'][k]
        return (input, output)
    
    def __len__(self):
        return self.group['inputs'].shape[0]
    
    def close(self):
        self.f.close()
 

class RewardModel(nn.Module):
    """A reward model using an A2C feature extractor"""
    def __init__(self, env, device):
        super(RewardModel, self).__init__()
        self.ac_model = A2C('MlpPolicy', env).policy
        self.reward_net = nn.Linear(64, 1).to(device)
        self.device = device
    
    def forward(self, obs):
        latent, _, _= self.ac_model._get_latent(th.tensor(obs).to(self.device))
        return self.reward_net(latent)
    
    def freeze_extractor(self):
        for p in self.ac_model.policy.features_extractor.parameters():
            p.requires_grad = False


if __name__ == '__main__':  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="BreakoutNoFrameskip-v4", help='environment ID')
    parser.add_argument('-e', '--epochs', help='Number of epochs to train for', default=5,
                        type=int)
    args = parser.parse_args()

    device = get_device()
    print(f"Using {device} device.")

    env_id = args.env
    if 'maze' not in env_id.lower():
        raise Exception(f"env {env_id} is not a maze env")
    env = gym.make(env_id)
    print(f"Created env with obs.shape = {env.reset().shape}.")
    
    train = RewardData(env_id, train=True)
    test = RewardData(env_id, train=False)
    
    train_loader = th.utils.data.DataLoader(train, batch_size=20, shuffle=True, num_workers=0)
    test_loader = th.utils.data.DataLoader(test, batch_size=20, shuffle=False, num_workers=0)
    
    reward_model = RewardModel(env, device)
    optimizer = th.optim.Adam(reward_model.parameters())
    loss_fn = th.nn.MSELoss(reduction="sum")
    
    num_batches = 0
    for e in range(args.epochs):        
        for samples, targets in tqdm(train_loader):
            optimizer.zero_grad()
            batch_loss = loss_fn(reward_model(samples), targets.to(device))
            batch_loss.backward()
            optimizer.step()
            num_batches += 1
        test_loss = 0
        for samples, targets in test_loader:
            with th.no_grad():
                test_loss += loss_fn(reward_model(samples), targets.to(device))
        print("Epoch {:3d} | Test Loss: {:.4f}".format(e, float(test_loss) / len(test)))
    
    th.save(reward_model.state_dict(), f"../reward-models/{env_id}-reward_model.pt")
        
