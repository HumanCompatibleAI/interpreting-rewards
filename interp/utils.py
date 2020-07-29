
import glob

import numpy as np

from gym.spaces import Box
from gym.wrappers import FrameStack

def get_latest_run_id(log_path, env_id):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: (str) path to log folder
    :param env_id: (str)
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + "/{}_[0-9]*".format(env_id)):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


class AtariFrameStack(FrameStack):
    """Stacks frames but without using LazyFrame."""

    def __init__(self, env, n_stack=4):
        """ 
        Wraper for `env` which stacks `n_stack` frames.

        Args:
            env (gym.Env): Environment to wrap
            n_stack (int): Number of observations to stack
        """
        super(AtariFrameStack, self).__init__(env, n_stack)
        # TODO: modify this line to make the wrapper work for more general types of environments, not just 84x84 Atari
        self.observation_space = Box(0, 255, shape=[84, 84, 4], dtype=env.observation_space.dtype)

    def _get_observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        stack = np.array(list(self.frames))         # (4, 84, 84)
        stack = np.transpose(stack, axes=(1, 2, 0)) # (84, 84, 4)
        return stack

