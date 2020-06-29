# interpreting-rewards
Eric and Adam's repo for exploring reward model interpretability

# Installation
Various notebooks require different packages. To run most of them, you will need to have `rl-baselines3-zoo` as a subdirectory here, so run:
```
git clone https://github.com/DLR-RM/rl-baselines3-zoo.git
```
You will also need to install the `stable-baselines3` package: https://github.com/DLR-RM/stable-baselines3.git

For DRL-HP reward model analysis, you'll need to install this implementation of `learning-from-human-preferences`: https://github.com/decodyng/learning-from-human-preferences/tree/big_refactor, in the parent directory of `interpreting-rewards`:
```
..
├── interpreting-rewards
├── learning-from-human-preferences
```

# Structure 
```
.
├── agents
├── datasets
├── notebooks
├── README.md
├── reward-models
├── rl-baselines3-zoo
└── videos
```

To download the contents of `videos`, `reward-models`, and `agents`, use
```
bash download.sh
```
If you make additions to these folders, use `upload.sh` to upload them to s3. 

## Training reward models with regression
Inside the `datasets` repository there is code for creating datasets of (obs, reward) pairs for various atari games. Datasets are saved as hdf5 files, which are too large for upload/download (16G for each environment). Generate them yourself!



