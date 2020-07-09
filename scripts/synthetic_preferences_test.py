
import sys
import logging

import gym

from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import A2C
from stable_baselines3.common.atari_wrappers import AtariWrapper


sys.path.insert(1, "/home/eric/Code/chai/test/learning-from-human-preferences")
sys.path.insert(2, "/home/eric/Code/chai/test/learning-from-human-preferences/drlhp")
from drlhp import HumanPreferencesEnvWrapper
from drlhp.reward_predictor_core_network import net_cnn

#import warnings
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    import easy_tf_log
##    from drlhp import reward_predictor
#    from drlhp.reward_predictor_core_network import net_cnn
##    from drlhp.HumanPreferencesEnvWrapper import _make_reward_predictor
#    from drlhp import HumanPreferencesEnvWrapper



if __name__ == '__main__':
    env = gym.make('PongNoFrameskip-v4')
    env = AtariWrapper(env)
    env = DummyVecEnv([lambda: env])
    #env = VecFrameStack(env, n_stack=4)
#    env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
#    env = VecFrameStack(env, n_stack=4)
    env.reward_range = (float('-inf'), float('inf'))
    #preferences_env = env
    preferences_env = HumanPreferencesEnvWrapper(env=env,
                            reward_predictor_network=net_cnn,
                            synthetic_prefs=True)
    #preferences_env = DummyVecEnv([lambda: preferences_env]) 
    model = A2C('CnnPolicy', preferences_env, verbose=1)
    model.learn(total_timesteps=int(1e5))

