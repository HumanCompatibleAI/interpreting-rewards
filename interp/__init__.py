
import gym
from .common.atari_no_score import AtariEnvNoScore, supported_games

for game in supported_games:
    name = ''.join([g.capitalize() for g in game.split('_')])
    obs_type = 'image'
    nondeterministic = False
    frameskip = 4
    gym.register(
            id='{}NoFrameskipNoScore-v4'.format(name),
            entry_point=AtariEnvNoScore,
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
    )
