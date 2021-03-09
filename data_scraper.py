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
from callback import CustomCallback
from collector_additional_information import Collector
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import make_vec_env
import os, datetime
import argparse
import sys

# get rid of distracting TF errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# mapping formal environment name to shorthand and viceversa
environment_to_name = {}
environment_to_name["MsPacmanNoFrameskip-v4"] = "Pacman"
environment_to_name["PongNoFrameskip-v4"] = "Pong"
name_to_environment = {}
environment_to_name["Pacman"] = "MsPacmanNoFrameskip-v4"
environment_to_name["Pong"] = "PongNoFrameskip-v4"

# TODO: handle invalid args and add more checks to args
# example: ensure that Pacman has --lives enabled 
parser = argparse.ArgumentParser()
parser.add_argument('--lives', help='env has lives', action='store_true', default=False)
parser.add_argument('--num_envs', help='set the number of environments', type=int, default=1)
parser.add_argument('--num_steps', help='set the number of steps/actions you want the agent to take', type=int, default=1000)
parser.add_argument('--algo', help='set the algorithm with which to train this model', type=str, default="DQN")
parser.add_argument('--save', help='save the trained model', action='store_true', default=False)
parser.add_argument('--environment', help='environment to use in training', type=str, default="MsPacmanNoFrameskip-v4")
parser.add_argument('--model', help = 'use a saved pretrained model', type=str, default = "")
parser.add_argument('--collect_extra', help='collect the extra information from game', action='store_true', default=False)
parser.add_argument('--remove_screenshots', help='delete the screenshots', action='store_true', default=False)

args = parser.parse_args()
isLives = args.lives
# set num timesteps (per environment)
num_steps = args.num_steps
# set num envs (number of games trained concurrently)
num_envs = args.num_envs
# set algorithm (DQN, A2C, PPO2)
algo = args.algo.upper()
# set environment
environment = args.environment
# set shorthand env name
env_name = environment_to_name[environment]
isSave = args.save
presaved_model = args.model
isCollectExtra = args.collect_extra
isRemoveScreenshots = args.remove_screenshots

print("isremove ", isRemoveScreenshots)

# create folder and subfolders for data
tmp_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
dir = algo + '_'+ env_name + '_data_' +  tmp_name + '/'
os.makedirs(dir)
subfolder = os.path.join(dir, 'screen')
os.makedirs(subfolder)

# TODO: make saved model names more flexible, instead of relying on it to follow the pattern:
# <ALGORITHM_NAME>_<GAME>_model_<TIMESTAMP>
# train based off a previously trained agent 
if(presaved_model != ""):
    parse_model_name = []
    parse_model_name =  presaved_model.split("_")

    if len(parse_model_name) < 2:
        print("Invalid presaved model name. Please check model naming convention or directory location.")
        sys.exit(1)

    model_algo = parse_model_name[0]
    # override algo flag param in case it is different from the presaved model
    algo = model_algo
    # override game enviroment flag param in case user accidentally passes 
    # game environment param that is different from the presaved model
    env_name = parse_model_name[1]
    environment = name_to_environment[env_name]

    if(model_algo == "DQN"):
        env = make_atari(environment)
        model = DQN.load(presaved_model)
        num_envs = 1
    
    elif(algo == "A2C" or algo == "PPO2"):
        num_steps = num_steps * num_envs
        env = make_atari_env(environment, num_env=num_envs, seed=0, wrapper_kwargs={'clip_rewards':False})
        # Stack 4 frames
        env = VecFrameStack(env, n_stack=4)
        actions = make_atari(environment).unwrapped.get_action_meanings()
        step_callback = CustomCallback(0,actions, env,  num_steps, dir, isLives, env, num_envs, algo, env_name)
        if(algo == "A2C"):
            model = A2C.load(presaved_model)

        elif(algo == "PPO2"):
            model = PPO2.load(presaved_model)
    else:
        print("Invalid presaved model name. Please check model naming convention or directory location.") 
        sys.exit(1)

    model.set_env(env)
    step_callback = CustomCallback(0,actions, env,  num_steps, dir, isLives, env, num_envs, algo, env_name)
    model.learn(total_timesteps=num_steps, callback = step_callback)
    if isSave:
        model.save(model_algo + "_" + env_name + "_model_" + tmp_name)

elif(algo == "A2C" or algo == "PPO2"):
    num_steps = num_steps * num_envs
    env = make_atari_env(environment, num_env=num_envs, seed=0, wrapper_kwargs={'clip_rewards':False})
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)
    actions = make_atari(environment).unwrapped.get_action_meanings()
    step_callback = CustomCallback(0,actions, env,  num_steps, dir, isLives, env, num_envs, algo, env_name)

    if(algo == "A2C"):
        model = A2C('CnnPolicy', env, verbose=1, n_steps=5)
        model.learn(total_timesteps=num_steps, callback=step_callback)
        
    elif(algo == "PPO2"):
        n_steps = 5
        model = PPO2('CnnPolicy', env, verbose=1, n_steps = n_steps, nminibatches = n_steps*num_envs)
        model.learn(total_timesteps=num_steps, callback=step_callback)
        
    if isSave:
        model.save(algo + "_" + env_name + "_model_" + tmp_name)

    if isCollectExtra:
        collector = Collector(dir, num_steps, num_envs)

        if(env_name == "Pacman"):
            collector.find_item_locations_pacman()
            collector.find_life_game_info()
        elif(env_name == "Pong"):
            collector.find_item_locations_pong()
        collector.output_modified_csv()
        if(isRemoveScreenshots):
            collector.remove_screenshots()
        
elif(algo == "DQN"):
    env = make_atari(environment)
    step_callback = CustomCallback(0,env.unwrapped.get_action_meanings(), env,  num_steps, dir, isLives, env, 1, "DQN", env_name)
    model = DQN(CnnPolicy, env, verbose=1)
    model.learn(total_timesteps=num_steps, callback = step_callback)
    if isSave:
        model.save("DQN_" + env_name + "_model_" + tmp_name)
    
    if isCollectExtra:
        print("collect extra ")
        collector = Collector(dir, num_steps, 1)
        if(env_name == "Pacman"):
            collector.find_item_locations_pacman()
            collector.find_life_game_info_dqn()
        elif(env_name == "Pong"):
            collector.find_item_locations_pong()
        collector.output_modified_csv()
        if(isRemoveScreenshots):
            collector.remove_screenshots()
else:
    print("Incorrect algorithm. Select pass --algo params: DQN or A2C or PPO2.")
    sys.exit(1)





