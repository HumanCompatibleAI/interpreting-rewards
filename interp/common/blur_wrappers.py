
import numpy as np
import scipy.ndimage

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper


class BlurWrapper(VecEnvWrapper):
    """
    Base class for wrappers blurring out a region of an Atari Game.
    """

    def __init__(self, venv, mask):
        assert venv.observation_space.shape == (84, 84, 4)
        super(BlurWrapper, self).__init__(venv)
        self.mask = mask

    @staticmethod
    def blur_score(obs: np.ndarray) -> np.ndarray:
        """
        Blur out the score region of `obs`
        """
        blurred_obs = np.zeros(obs.shape)
        for n in range(obs.shape[0]):
            for i in range(4):
                frame = obs[n, :, :, i]
                A = scipy.ndimage.gaussian_filter(frame, 3)
                blurred = frame*(1 - self.mask) + A*self.mask
                blurred_obs[n, :, :, i] = blurred
        return blurred_obs

    def step_wait(self) -> 'GymStepReturn':
        observations, rewards, dones, infos = self.venv.step_wait()
        return self.blur_score(observations), rewards, dones, infos

    def reset(self) -> np.ndarray:
        return self.blur_score(self.venv.reset())


class BlurBreakoutScore(BlurWrapper):
    """
    Blurs out the score region in Breakout.
    """
    
    def __init__(self, venv):
        assert venv.observation_space.shape == (84, 84, 4)
        M = np.zeros((84, 84))
        M[1:7, 18:43] = 1.
        super(BlurBreakoutScore, self).__init__(venv, M)


class BlurSeaquestScore(BlurWrapper):
    """
    Blurs out the score region in Breakout.
    """
    
    def __init__(self, venv):
        assert venv.observation_space.shape == (84, 84, 4)
        M = np.zeros((84, 84))
        M[3:7, 38:56] = 1.
        super(BlurBreakoutScore, self).__init__(venv, M)
