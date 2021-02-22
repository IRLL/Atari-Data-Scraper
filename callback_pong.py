from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from collections import OrderedDict
from ghost_tracker import GhostTracker
import colour_detection as cd


class CustomCallbackPong(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    actions = []
    directory = 'results/'
    save_file_screen = os.path.join(directory, 'screen', 'screen')
    env = None
    step = 1
    main_data_dict = OrderedDict()
    df_list = []
    df_list_mod = []
    num_steps = 10
    isLives = False

    def __init__(self, verbose=0, env_actions=[], env=None, num_steps=10, dir='results/', isLives=False):
        super(CustomCallbackPong, self).__init__(verbose)
        self.actions = env_actions
        self.env = env.unwrapped
        self.num_steps = num_steps
        self.directory = dir
        self.isLives = isLives
        print("game has lives? ", self.isLives)
        # env <MaxAndSkipEnv<NoopResetEnv<TimeLimit<AtariEnv<MsPacmanNoFrameskip-v4>>>>>
        print("actions ", self.actions)
        print("dir ", self.directory)
        print("env", env.unwrapped)
        print("mod ",  self.model)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    # dataframe is a db table
    def make_dataframes(self, df):
        # Make the main Dataframe
        main_df = pd.DataFrame.from_dict(
            CustomCallbackPong.main_data_dict, orient='index')

        # call to save last items as seperate df
        # self.save_last_line(args.stream_folder, main_df)

        # Now that we've parsed down the main df, load all into our list
        # of DFs and our list of Names
        # self.df_list.append(main_df)
        df.append(main_df)

    def df_to_csv(self, filename, df_list):
        for df in df_list:
            # filename = "df.csv"
            filepath = os.path.join(self.directory, filename)
            print("Making csvs and path is: ")
            print(filepath)
            if os.path.exists(filepath):
                df.to_csv(filepath, mode='a', index=False, header=False)
            else:
                df.to_csv(filepath, mode='a', index=False)

    def df_to_parquet(self):
        for df in self.df_list:
            filename = "df.parquet"
            filepath = os.path.join(self.directory, filename)
            print("Making parquets and path is: ")
            print(filepath)
            table = pa.Table.from_pandas(df)
            # Parquet with Brotli compression
            pq.write_table(table, filepath, compression='BROTLI')

    def save_frame(self, array, save_file, frame):
        if not (os.path.isdir(save_file)):
            os.makedirs(save_file)
            os.rmdir(save_file)
        plt.imsave(save_file + '_' + str(frame) + '.png', array)

    def save_observations(self, observations):
        for i in range(len(observations)):
            index = str("_") + '_' + str(i)
            observation = observations[i]
            self.save_frame(observation, self.save_file_screen, index)

     # TODO: maybe move to a separate file
    def util(self):
        total_life = total_game = steps_life = steps_game = 1
        prev_life = 3
        episode_reward = 0
        total_reward = 0
        game_reward = 0
        print("in util func")
        for key, value in CustomCallbackPong.main_data_dict.items():
            # DO NOT UNCOMMENT
            # if(key < 2):
            #     CustomCallbackPong.main_data_dict[key]['step_reward'] = value['cumulative_episode_reward']
            # else:
            #     if( CustomCallbackPong.main_data_dict[key-1]['lives'] == 0):
            #         CustomCallbackPong.main_data_dict[key]['step_reward'] = 0
            #     else:
            #         CustomCallbackPong.main_data_dict[key]['step_reward'] = value['cumulative_episode_reward'] - \
            #             CustomCallbackPong.main_data_dict[key-1]['cumulative_episode_reward']
                
            # episode_reward += CustomCallbackPong.main_data_dict[key]['curr_score'] 

            if(self.isLives):
                # game over (epoch)
                if(value['lives'] == 0):
                    # not sure if this is correct
                    # total_reward += game_reward
                    CustomCallbackPong.main_data_dict[key]['game_reward'] = game_reward
                    # CustomCallbackPong.main_data_dict[key]['total_life'] = total_life
                    # CustomCallbackPong.main_data_dict[key]['episode_reward'] = episode_reward
                    # reset values
                    total_game += 1
                    # total_life += 1
                    steps_game = steps_life = 0
                    game_reward = 0
                    episode_reward = 0

                # lost a life (episode)
                # elif(value['lives'] != prev_life and prev_life != 0):
                # record BEFORE lives is decremented
                elif(key != self.num_steps and value['lives'] != CustomCallbackPong.main_data_dict[key+1]['lives']):
                    # not sure if this is correct
                    CustomCallbackPong.main_data_dict[key]['total_life'] = total_life
                    CustomCallbackPong.main_data_dict[key]['episode_reward'] = episode_reward
                    game_reward += episode_reward
                    total_reward += episode_reward
                    total_life += 1
                    # steps_game += steps_life
                    steps_life = 1
                    episode_reward = 0

                # normal step
                prev_life = value['lives']
                CustomCallbackPong.main_data_dict[key]['steps_life'] = steps_life
                
                CustomCallbackPong.main_data_dict[key]['steps_game'] = steps_game
                CustomCallbackPong.main_data_dict[key]['total_game'] = total_game

                CustomCallbackPong.main_data_dict[key]['total_reward'] = total_reward

                steps_life += 1
                steps_game += 1
            else:
                total_reward += CustomCallbackPong.main_data_dict[key]['curr_score'] 
                CustomCallbackPong.main_data_dict[key]['total_reward'] = total_reward
            # find coordinates of pacman and ghosts
            subfolder = os.path.join(self.directory, 'screen')
            dir = self.directory.replace("/", "")
            filepath = dir + "\screen\screenshot" + str(key) + ".png"

            
            ball_coord, green_paddle_coord, brown_paddle_coord, distance = cd.find_pong_coords(filepath)
            CustomCallbackPong.main_data_dict[key]['ball_coord_x'] = ball_coord[0]
            CustomCallbackPong.main_data_dict[key]['ball_coord_y'] = ball_coord[1]
            CustomCallbackPong.main_data_dict[key]['green_paddle_coord_x'] = green_paddle_coord[0]
            CustomCallbackPong.main_data_dict[key]['green_paddle_coord_y'] = green_paddle_coord[1]
            CustomCallbackPong.main_data_dict[key]['brown_paddle_coord_x'] = brown_paddle_coord[0]
            CustomCallbackPong.main_data_dict[key]['brown_paddle_coord_y'] = brown_paddle_coord[1]
            CustomCallbackPong.main_data_dict[key]['paddle_ball_distance'] = distance
 
    def _on_step(self) -> bool:
        if(CustomCallbackPong.step == 1):
            
            print("locs ", self.locals)

        # save screenshot to folder
        subfolder = os.path.join(self.directory, 'screen')
        filepath = os.path.join(
            subfolder, 'screenshot' + str(CustomCallbackPong.step) + '.png')
        self.env.ale.saveScreenPNG(filepath)
        step_stats = {CustomCallbackPong.step: {
            'action': self.locals['env_action'],
            'action_name': self.actions[self.locals['env_action']],
            'curr_score': self.locals['episode_rewards'][-1],
            'state': CustomCallbackPong.step
            
            }
        }
        # add step to dict
        CustomCallbackPong.main_data_dict.update(step_stats)
        if(self.isLives == True):
            CustomCallbackPong.main_data_dict[CustomCallbackPong.step]['lives'] = self.locals['info']['ale.lives']
        
        # TODO: edge detection for ball location? paddle location? 
        if(CustomCallbackPong.step % 1000 == 0):
            print("at step ", CustomCallbackPong.step)

        if(CustomCallbackPong.step == self.num_steps):
            # print("dictionary ", CustomCallbackPong.main_data_dict)
            self.make_dataframes(self.df_list)
            self.df_to_csv("df_og.csv", self.df_list)
            self.df_to_parquet()
            # test if parquet file is correctly created
            # print("reading parquet file")
            # print(pd.read_parquet(os.path.join(self.directory,  "df.parquet")))

            # calculate new info
            self.util()
            self.make_dataframes(self.df_list_mod)
            self.df_to_csv("df_mod.csv", self.df_list_mod)
            # self.df_to_parquet()
            print("done! inside callback")
        CustomCallbackPong.step = CustomCallbackPong.step + 1