# interpreting-rewards
This repository accompanies the paper [Understanding Learned Reward Functions](https://ericjmichaud.com/rewards.pdf) by [Eric J. Michaud](https://ericjmichaud.com), [Adam Gleave](https://gleave.me) and [Stuart Russell](https://people.eecs.berkeley.edu/~russell/). It aims to enable easy reproduction of the results from that paper and to serve as a branching-off point for future iterations of the work.


## Installation

First, clone the repository:
```
git clone -b dev-abridged --recurse-submodules https://github.com/HumanCompatibleAI/interpreting-rewards.git
```
Which will also clone `rl-baselines3-zoo` as a submodule.

To install dependencies, run
```
pip install -r requirements.txt
pip install -r rl-baselines3-zoo/requirements.txt
```

## Usage

**To replicate gridworld results from the very beginning**, first train a policy on the gridworld environment:
```
./prepare
cd rl-baselines3-zoo
python train.py --algo ppo --env EmptyMaze-10x10-FixedGoal-v3 -f ../agents --seed 0 --gym-packages mazelab
cd ..
```
Then use this policy to create a dataset of (transition, reward) pairs, to train a reward model on via regression:
```
cd scripts
python maze_create_dataset.py --algo ppo --env EmptyMaze-10x10-CoinFlipGoal-v3 -f ../agents --seed 0
```
And train the reward model:
```
python maze_train_reward_model.py --env EmptyMaze-10x10-CoinFlipGoal-v3 --epochs 5 --seed 0
```
From here, the saliency map figures from the paper can be created with a single command:
```
python maze_figures.py 
```
Which will create and save the figures to the `./figures` directory of the repository root directory. 

**To replicate Atari results**, first train policies on Breakout and Seaquest:
```
cd rl-baselines3-zoo
python train.py --algo ppo --env BreakoutNoFrameskip-v4 -f ../agents
python train.py --algo ppo --env SeaquestNoFrameskip-v4 -f ../agents
```
Create datasets for each:
```
cd ../scripts
python atari_create_dataset.py --algo ppo --env BreakoutNoFrameskip-v4 -f ../agents --seed 0
python atari_create_dataset.py --algo ppo --env SeaquestNoFrameskip-v4 -f ../agents --seed 0
```
Train reward models:
```
python atari_train_reward_model.py --algo ppo --env BreakoutNoFrameskip-v4 -f ../agents --seed 0 --epochs 5
python atari_train_reward_model.py --algo ppo --env SeaquestNoFrameskip-v4 -f ../agents --seed 0 --epochs 5
```

## TODO:
* Finish testing the Atari pipeline, add a script for creating Atari figures
* Fix the scripts which train maze agents and Atari agents using a learned reward function. Due to changes in `rl-baselines3-zoo` since I last ran these, they no longer work. The Atari script will need a code review, as the results that it generated before were weird (large differences the performance of the policy trained on ground-truth reward vs. reward model, despite the reward model having close to 0 test error. This could be an issue with the reward model only being trained on transitions from an expert policy, but I worry it could be something squirrelly with the script or my custom wrappers.
* Add scripts for downloading parts of the pipeline from AWS without having to run the policy training, dataset creation, reward model training scripts, etc.
* Clean up the repo by removing lots of notebooks from `./notebooks` 


