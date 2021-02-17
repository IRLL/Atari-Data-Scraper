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

# TODO: handle invalid args 
parser = argparse.ArgumentParser()
parser.add_argument('--lives', help='env has lives', action='store_true', default=False)
parser.add_argument('--num_envs', help='set the number of environments', type=int, default=1)
parser.add_argument('--num_steps', help='set the number of environments', type=int, default=1000)
parser.add_argument('--algo', help='set the algorithm with which to train this model', type=str, default="DQN")
parser.add_argument('--save', help='save the trained model', action='store_true', default=False)
# add arg for atari env 

args = parser.parse_args()
isLives = args.lives
# set num timesteps (per environment)
num_steps = args.num_steps
# set num envs
num_envs = args.num_envs
# set algorithm
algo = args.algo.upper()
isSave = args.save


# create folder and subfolders for data
tmp_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
dir = algo + '_data_' +  tmp_name + '/'
os.makedirs(dir)
subfolder = os.path.join(dir, 'screen')
os.makedirs(subfolder)


if(algo == "A2C" or algo == "PPO2"):
    num_steps = num_steps * num_envs
    if(algo == "A2C"):
        # TODO: get actions inside callback instead 
        
        # kwargs suppress reward scaling 
        env = make_atari_env('MsPacmanNoFrameskip-v4', num_env=num_envs, seed=0, wrapper_kwargs={'clip_rewards':False})
        # Stack 4 frames
        env = VecFrameStack(env, n_stack=4)

        # define callback object
        step_callback = CustomCallbackA(0, actions, env,  num_steps, dir, isLives, make_atari('MsPacmanNoFrameskip-v4'), num_envs)
        
        # TODO: make option to save new model or use pretrained model
        model = A2C('CnnPolicy', env, verbose=1, n_steps=5)
        model.learn(total_timesteps=num_steps, callback=step_callback)
        if isSave:
            model.save("A2C_model_" + tmp_name)

    if(algo == "PPO2"):
        actions = make_atari('MsPacmanNoFrameskip-v4').unwrapped.get_action_meanings()
        env = make_atari_env('MsPacmanNoFrameskip-v4', num_env=num_envs, seed=0, wrapper_kwargs={'clip_rewards':False})
        # Stack 4 frames
        env = VecFrameStack(env, n_stack=4)
        step_callback = CustomCallbackA(0, actions, env,  num_steps, dir, isLives, make_atari('MsPacmanNoFrameskip-v4'), num_envs)
        n_steps = 5
        model = PPO2('CnnPolicy', env, verbose=1,n_steps = n_steps, nminibatches = n_steps*num_envs)
        
        model.learn(total_timesteps=num_steps, callback=step_callback)
        if isSave:
            model.save("PPO2_model_" + tmp_name)
# n_updates = total_timesteps // self.n_batch
# self.n_batch = self.n_envs * self.n_steps

