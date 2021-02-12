# TODO: alphabetize imports 
import gym
from collections import OrderedDict
import os
import tensorflow as tf
import pandas as pd
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, VecEnvWrapper
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN, A2C, PPO2
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from callback_a2c import CustomCallbackA
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import make_vec_env
import os, datetime
import argparse

# get rid of distracting TF errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# TODO: make flags for num steps and num envs
parser = argparse.ArgumentParser()
parser.add_argument('--lives', help='env has lives', action='store_true', default=False)
args = parser.parse_args()
isLives = args.lives

# set num timesteps (per environment)
num_steps = 10
# set num envs
num_envs = 5

num_steps = num_steps * num_envs

# TODO: get actions inside callback instead 
actions = make_atari('MsPacmanNoFrameskip-v4').unwrapped.get_action_meanings()
# kwargs suppress reward scaling 
env = make_atari_env('MsPacmanNoFrameskip-v4', num_env=num_envs, seed=0, wrapper_kwargs={'clip_rewards':False})
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

# create folder and subfolders for data
dir = 'a2c_data_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'
os.makedirs(dir)
subfolder = os.path.join(dir, 'screen')
os.makedirs(subfolder)

# define callback object
step_callback = CustomCallbackA(0, actions, env,  num_steps, dir, isLives, make_atari('MsPacmanNoFrameskip-v4'), num_envs)

# TODO: make option to save new model or use pretrained model
model = A2C('CnnPolicy', env, verbose=1, n_steps=5)
model.learn(total_timesteps=num_steps, callback=step_callback)
model.save("test_model")


