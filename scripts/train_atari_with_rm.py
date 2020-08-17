import argparse
import difflib
import os
import sys
import importlib
import time
import uuid
import warnings
import re
from collections import OrderedDict
from pprint import pprint
from pathlib import Path

import yaml
import gym
import seaborn
import numpy as np
import torch as th
# For custom activation fn
import torch.nn as nn  # noqa: F401 pytype: disable=unused-import

from sacred import Experiment
from sacred.observers import FileStorageObserver

from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import constant_fn, get_device
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import A2C, PPO, DQN

sys.path.append('../rl-baselines3-zoo/')
# Register custom envs
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class
from utils.hyperparams_opt import hyperparam_optimization
from utils.callbacks import SaveVecNormalizeCallback
from utils.noise import LinearNormalActionNoise
from utils.utils import StoreDict, get_callback_class

from imitation.util import sacred as sacred_util
from imitation.util.reward_wrapper import RewardVecEnvWrapper
import interp
from interp.utils import get_latest_run_id
from interp.common.wrappers import CustomRewardWrapper, DummyWrapper

seaborn.set()

ex = Experiment()

ALGOS = {
    "a2c": A2C,
    "ppo": PPO,
    "dqn": DQN
}

class RewardModel(nn.Module):
    """A reward model using an A2C feature extractor"""
    def __init__(self, env, device):
        super(RewardModel, self).__init__()
        self.ac_model = A2C('CnnPolicy', env).policy
        self.reward_net = nn.Linear(512, 1).to(device)
        self.device = device

    def forward(self, obs):
        obs_transposed = VecTransposeImage.transpose_image(obs)
        latent, _, _= self.ac_model._get_latent(th.tensor(obs_transposed).to(self.device))
        return self.reward_net(latent)

    def freeze_extractor(self):
        for p in self.ac_model.policy.features_extractor.parameters():
            p.requires_grad = False


@ex.config
def config():
    env_id = "PongNoFrameskip-v4"        # the gym environment to use
    algo = "ppo"                         # the RL algorithm to use
    if 'NoFrameskip' not in env_id:
        raise Exception("Only Atari environments allowed")
    timesteps = int(1e7)                 # number of timesteps to train the policy on
    device = get_device()                # the device to put/train the SB3 model on
    eval_freq = timesteps // 1000        # frequency to evaluate a SB3 model and possibly save new best
    regressed_reward = False            # whether to override the environment reward with a reward model
    verbose = 1                          # verbosity to the max
    use_uuid = False                     # whether to use a unique uuid for the experiment
    eval_episodes = 10                   # how many episodes to use when evaluating return

@ex.main
def run(
    _run,
    env_id,
    algo,
    timesteps,
    device,
    eval_freq,
    regressed_reward,
    seed,
    verbose,
    use_uuid,
    eval_episodes,
    ):

    with open(f'hyperparams/{algo}.yml', 'r') as f:
        hyperparams_dict = yaml.safe_load(f)
        hyperparams = hyperparams_dict['atari']

    uuid_str = f'_{uuid.uuid4()}' if use_uuid else ''
    set_random_seed(seed)

    tensorboard_log = None 

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {seed}")

    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
    # env_kwargs = {} if args.env_kwargs is None else args.env_kwargs
    env_kwargs = {}

    algo_ = algo

    if verbose > 0:
        pprint(saved_hyperparams)

    n_envs = hyperparams.get('n_envs', 1)

    if verbose > 0:
        print(f"Using {n_envs} environments")

    # Create schedules
    for key in ['learning_rate', 'clip_range', 'clip_range_vf']:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split('_')
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], (float, int)):
            # Negative value: ignore (ex: for clipping)
            if hyperparams[key] < 0:
                continue
            hyperparams[key] = constant_fn(float(hyperparams[key]))
        else:
            raise ValueError(f'Invalid value for {key}: {hyperparams[key]}')

    # Should we overwrite the number of timesteps?
    if timesteps > 0:
        if verbose:
            print(f"Overwriting n_timesteps with n={timesteps}")
    else:
        timesteps = int(hyperparams['n_timesteps'])

    normalize = False
    normalize_kwargs = {}
    if 'normalize' in hyperparams.keys():
        normalize = hyperparams['normalize']
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams['normalize']

    if 'policy_kwargs' in hyperparams.keys():
        # Convert to python object if needed
        if isinstance(hyperparams['policy_kwargs'], str):
            hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

    # Delete keys so the dict can be pass to the model constructor
    if 'n_envs' in hyperparams.keys():
        del hyperparams['n_envs']
    del hyperparams['n_timesteps']

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']

    log_path = f"output/regressed_exps/{algo}/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    save_path = os.path.join(log_path, f"{env_id}_{get_latest_run_id(log_path, env_id) + 1}{uuid_str}")
    params_path = f"{save_path}/{env_id}"
    os.makedirs(params_path, exist_ok=True)
    sacred_util.build_sacred_symlink(save_path, _run)

    callbacks = get_callback_class(hyperparams)
    if 'callback' in hyperparams.keys():
        del hyperparams['callback']

    def create_env(n_envs, eval_env=False, regressed_reward=False, no_log=False):
        """
        Create the environment and wrap it if necessary
        :param n_envs: (int)
        :param eval_env: (bool) Whether is it an environment used for evaluation or not
        :param no_log: (bool) Do not log training when doing hyperparameter optim
            (issue with writing the same file)
        :return: (Union[gym.Env, VecEnv])
        """
        # global hyperparams
        # global env_kwargs

        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env or no_log else save_path

        if n_envs == 1:
            env = DummyVecEnv([make_env(env_id, 0, seed,
                               wrapper_class=env_wrapper, log_dir=log_dir,
                               env_kwargs=env_kwargs)])
        else:
            # env = SubprocVecEnv([make_env(env_id, i, seed) for i in range(n_envs)])
            # On most env, SubprocVecEnv does not help and is quite memory hungry
            env = DummyVecEnv([make_env(env_id, i, seed, log_dir=log_dir, env_kwargs=env_kwargs,
                                        wrapper_class=env_wrapper) for i in range(n_envs)])
        if normalize:
            # Copy to avoid changing default values by reference
            local_normalize_kwargs = normalize_kwargs.copy()
            # Do not normalize reward for env used for evaluation
            if eval_env:
                if len(local_normalize_kwargs) > 0:
                    local_normalize_kwargs['norm_reward'] = False
                else:
                    local_normalize_kwargs = {'norm_reward': False}

            if verbose > 0:
                if len(local_normalize_kwargs) > 0:
                    print(f"Normalization activated: {local_normalize_kwargs}")
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **local_normalize_kwargs)

        # Optional Frame-stacking
        if hyperparams.get('frame_stack', False):
            n_stack = hyperparams['frame_stack']
            env = VecFrameStack(env, n_stack)
            print(f"Stacking {n_stack} frames")

        if regressed_reward:
            name = env_id.split("NoFrameskip")[0]
            rm_path = f"../reward-models/{name}NoFrameskip-v4-reward_model.pt"
            if not os.path.exists(rm_path):
                raise Exception(f"Cannot find reward reward model at {rm_path}")
            rm = RewardModel(env, device)
            rm.load_state_dict(th.load(rm_path))
            # def reward_fn(obs, action, next_obs, infos):
            #     return rm(next_obs).cpu().detach().numpy()
            # env = RewardVecEnvWrapper(env, reward_fn)
            env = CustomRewardWrapper(env, rm)
        else:
            env = DummyWrapper(env)

        if is_image_space(env.observation_space):
            if verbose > 0:
                print("Wrapping into a VecTransposeImage")
            env = VecTransposeImage(env)
        return env

    env = create_env(n_envs, regressed_reward=regressed_reward)

    # Create test env if needed, do not normalize reward
    eval_env = None
    if eval_freq > 0:
        # Account for the number of parallel environments
        eval_freq = max(eval_freq // n_envs, 1)

        if 'NeckEnv' in env_id:
            # Use the training env as eval env when using the neck
            # because there is only one robot
            # there will be an issue with the reset
            eval_callback = EvalCallback(env, callback_on_new_best=None,
                                         best_model_save_path=save_path,
                                         log_path=save_path, eval_freq=eval_freq)
            callbacks.append(eval_callback)
        else:
            if verbose > 0:
                print("Creating test environment")

            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=params_path)
            eval_callback = EvalCallback(create_env(1, eval_env=True, regressed_reward=False), 
                                         callback_on_new_best=save_vec_normalize,
                                         best_model_save_path=save_path, n_eval_episodes=eval_episodes,
                                         log_path=save_path, eval_freq=eval_freq,
                                         deterministic=not True)

            callbacks.append(eval_callback)

    # TODO: check for hyperparameters optimization
    # TODO: check What happens with the eval env when using frame stack
    if 'frame_stack' in hyperparams:
        del hyperparams['frame_stack']

    # Stop env processes to free memory
    # if args.optimize_hyperparameters and n_envs > 1:
    #     env.close()

    # Parse noise string for DDPG and SAC
#     if algo_ in ['ddpg', 'sac', 'td3'] and hyperparams.get('noise_type') is not None:
#         noise_type = hyperparams['noise_type'].strip()
#         noise_std = hyperparams['noise_std']
#         n_actions = env.action_space.shape[0]
#         if 'normal' in noise_type:
#             if 'lin' in noise_type:
#                 final_sigma = hyperparams.get('noise_std_final', 0.0) * np.ones(n_actions)
#                 hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
#                                                                       sigma=noise_std * np.ones(n_actions),
#                                                                       final_sigma=final_sigma,
#                                                                       max_steps=n_timesteps)
#             else:
#                 hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
#                                                                 sigma=noise_std * np.ones(n_actions))
#         elif 'ornstein-uhlenbeck' in noise_type:
#             hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
#                                                                        sigma=noise_std * np.ones(n_actions))
#         else:
#             raise RuntimeError(f'Unknown noise type "{noise_type}"')
#         print(f"Applying {noise_type} noise with std {noise_std}")
#         del hyperparams['noise_type']
#         del hyperparams['noise_std']
#         if 'noise_std_final' in hyperparams:
#             del hyperparams['noise_std_final']
# 
#     if args.trained_agent.endswith('.zip') and os.path.isfile(args.trained_agent):
#         # Continue training
#         print("Loading pretrained agent")
#         # Policy should not be changed
#         del hyperparams['policy']
# 
#         if 'policy_kwargs' in hyperparams.keys():
#             del hyperparams['policy_kwargs']
# 
#         model = ALGOS[args.algo].load(args.trained_agent, env=env, seed=seed,
#                                       tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)
# 
#         exp_folder = args.trained_agent.split('.zip')[0]
#         if normalize:
#             print("Loading saved running average")
#             stats_path = os.path.join(exp_folder, env_id)
#             if os.path.exists(os.path.join(stats_path, 'vecnormalize.pkl')):
#                 env = VecNormalize.load(os.path.join(stats_path, 'vecnormalize.pkl'), env)
#             else:
#                 # Legacy:
#                 env.load_running_average(exp_folder)
# 
#         replay_buffer_path = os.path.join(os.path.dirname(args.trained_agent), 'replay_buffer.pkl')
#         if os.path.exists(replay_buffer_path):
#             print("Loading replay buffer")
#             model.load_replay_buffer(replay_buffer_path)
# 
#     elif args.optimize_hyperparameters:
# 
#         if args.verbose > 0:
#             print("Optimizing hyperparameters")
# 
#         if args.storage is not None and args.study_name is None:
#             warnings.warn(f"You passed a remote storage: {args.storage} but no `--study-name`."
#                           "The study name will be generated by Optuna, make sure to re-use the same study name "
#                           "when you want to do distributed hyperparameter optimization.")
# 
#         def create_model(*_args, **kwargs):
#             """
#             Helper to create a model with different hyperparameters
#             """
#             return ALGOS[args.algo](env=create_env(n_envs, no_log=True), tensorboard_log=tensorboard_log,
#                                     verbose=0, **kwargs)
# 
#         data_frame = hyperparam_optimization(args.algo, create_model, create_env, n_trials=args.n_trials,
#                                              n_timesteps=n_timesteps, hyperparams=hyperparams,
#                                              n_jobs=args.n_jobs, seed=seed,
#                                              sampler_method=args.sampler, pruner_method=args.pruner,
#                                              n_startup_trials=args.n_startup_trials, n_evaluations=args.n_evaluations,
#                                              storage=args.storage, study_name=args.study_name,
#                                              verbose=args.verbose, deterministic_eval=not is_atari)
# 
#         report_name = (f"report_{env_id}_{args.n_trials}-trials-{n_timesteps}"
#                        f"-{args.sampler}-{args.pruner}_{int(time.time())}.csv")
# 
#         log_path = os.path.join(args.log_folder, args.algo, report_name)
# 
#         if args.verbose:
#             print(f"Writing report to {log_path}")
# 
#         os.makedirs(os.path.dirname(log_path), exist_ok=True)
#         data_frame.to_csv(log_path)
#         exit()
#     else:
#         # Train an agent from scratch
#         model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log,
#                                  seed=seed, verbose=args.verbose, **hyperparams)

    model = ALGOS[algo](env=env, tensorboard_log=tensorboard_log,
                                  seed=seed, verbose=verbose, **hyperparams)

    kwargs = {}
    # if args.log_interval > -1:
    #     kwargs = {'log_interval': args.log_interval}

    if len(callbacks) > 0:
        kwargs['callback'] = callbacks

    # Save hyperparams
    with open(os.path.join(params_path, 'config.yml'), 'w') as f:
        yaml.dump(saved_hyperparams, f)

    # save command line arguments
    # with open(os.path.join(params_path, 'args.yml'), 'w') as f:
    #     ordered_args = OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
    #     yaml.dump(ordered_args, f)

    # print(f"Log path: {save_path}")

    try:
        model.learn(timesteps, eval_log_path=save_path, eval_env=eval_env, eval_freq=eval_freq, **kwargs)
    except KeyboardInterrupt:
        pass

    # Save trained model

    print(f"Saving to {save_path}")
    model.save(f"{save_path}/{env_id}")

    # if hasattr(model, 'save_replay_buffer') and args.save_replay_buffer:
    #     print("Saving replay buffer")
    #     model.save_replay_buffer(os.path.join(save_path, 'replay_buffer.pkl'))

    if normalize:
        # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(os.path.join(params_path, 'vecnormalize.pkl'))
        # Deprecated saving:
        # env.save_running_average(params_path)


def main_console():
    observer = FileStorageObserver.create(os.path.join("output", "sacred"))
    ex.observers.append(observer)
    ex.run_commandline()


if __name__ == "__main__":
    main_console()




