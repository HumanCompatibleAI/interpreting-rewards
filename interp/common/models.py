
import numpy as np

import torch as th
import torch.nn as nn

from stable_baselines3 import A2C


class MazeRewardModel(nn.Module):
    """A simple 2-hidden-layer MLP reward model for mazelab environments.""" 
    def __init__(self, env, device):
        """Iniitalize a reward model with a Gym environment and a PyTorch device."""
        super(MazeRewardModel, self).__init__()
        w, h = env.observation_space.shape
        features = 2 * w * h
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(features, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, 64, bias=True),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        ).to(device)
        self.device = device

    def forward(self, obs):
        """Evaluate the reward model on an np.ndarray observation (s, s')."""
        return self.net(th.tensor(obs).to(self.device))
    
    def tforward(self, ss):
        """Evaluate the reward model on an a th.Tensor observation (s, s')."""
        return self.net(ss)


class AtariRewardModel(nn.Module):
    """A reward model for Atari, using the CNN feature extractor that SB3 policies use."""
    def __init__(self, env, device):
        super(RewardModel, self).__init__()
        self.ac_model = A2C('CnnPolicy', env).policy
        self.reward_net = nn.Linear(512, 1).to(device)
        self.device = device
    
    def forward(self, obs):
        obs_transposed = VecTransposeImage.transpose_image(obs)
        latent, _, _= self.ac_model._get_latent(th.tensor(obs_transposed).to(self.device))
        return self.reward_net(latent)
    
    def forward_tensor(self, obs):
        """obs is a tensor which has already been transposed correctly."""
        latent, _, _= self.ac_model._get_latent(obs.to(self.device))
        return self.reward_net(latent)
    
    def freeze_extractor(self):
        for p in self.ac_model.policy.features_extractor.parameters():
            p.requires_grad = False
