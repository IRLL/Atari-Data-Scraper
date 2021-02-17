from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from collections import OrderedDict
from ghost_tracker import GhostTracker
import colour_detection as cd
from stable_baselines.common.atari_wrappers import make_atari
import cv2 as cv
from PIL import Image as im 

class CustomCallbackA(BaseCallback):
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
    num_envs = 4
    og_env = make_atari('MsPacmanNoFrameskip-v4')
    # (0, actions, env,  num_steps, dir, isLives, make_atari('MsPacmanNoFrameskip-v4'), num_envs)
    def __init__(self, verbose=0, env_actions=[], env=None, num_steps=10, dir='results/', isLives=False, og_env = "", num_envs = 4):
        super(CustomCallbackA, self).__init__(verbose)
        self.actions = env_actions
        self.env = env
        self.num_steps = num_steps
        self.directory = dir
        self.isLives = isLives
        self.og_env = og_env.unwrapped
        self.num_envs = num_envs
        print("num stepss", self.num_steps)
        print("num timetepss", self.num_timesteps)
        print("game has lives? ", self.isLives)
        # env <MaxAndSkipEnv<NoopResetEnv<TimeLimit<AtariEnv<MsPacmanNoFrameskip-v4>>>>>
        print("dir ", self.directory)
        print("env", self.env)
        print("og_env", self.og_env)
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
            CustomCallbackA.main_data_dict, orient='index')

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

    def find_item_locations(self):
        subfolder = os.path.join(self.directory, 'screen/')
        for screen_num in range(self.num_envs, self.num_timesteps + self.num_envs , self.num_envs):
            for i in range(self.num_envs):
                filepath = subfolder + "env_" + str(i) + "_screenshot_" + str(screen_num) + "_.png"
                # print("filepath ", filepath)
                key = screen_num
                pacman_coord, pink_ghost_coord, red_ghost_coord, green_ghost_coord, orange_ghost_coord, to_pink_ghost, to_red_ghost, to_green_ghost, to_orange_ghost, pill_eaten, pill_dist, hasBlueGhost = cd.find_all_coords(
                    filepath)
                CustomCallbackA.main_data_dict[key]['pacman_coord_x_env_'+ str(i)] = pacman_coord[0]
                CustomCallbackA.main_data_dict[key]['pacman_coord_y_env_'+ str(i)] = pacman_coord[1]
                CustomCallbackA.main_data_dict[key]['pink_ghost_coord_x_env_'+ str(i)] = pink_ghost_coord[0]
                CustomCallbackA.main_data_dict[key]['pink_ghost_coord_y_env_'+ str(i)] = pink_ghost_coord[1]
                CustomCallbackA.main_data_dict[key]['to_pink_ghost_env_' + str(i)] = to_pink_ghost
                CustomCallbackA.main_data_dict[key]['red_ghost_coord_x_env_' + str(i) ] = red_ghost_coord[0]
                CustomCallbackA.main_data_dict[key]['red_ghost_coord_y_env_'+ str(i)] = red_ghost_coord[1]
                CustomCallbackA.main_data_dict[key]['to_red_ghost_env_'+ str(i)] = to_red_ghost
                CustomCallbackA.main_data_dict[key]['green_ghost_coord_x_env_'+ str(i)] = green_ghost_coord[0]
                CustomCallbackA.main_data_dict[key]['green_ghost_coord_y_env_'+ str(i)] = green_ghost_coord[1]
                CustomCallbackA.main_data_dict[key]['to_green_ghost_env_'+ str(i)] = to_green_ghost
                CustomCallbackA.main_data_dict[key]['orange_ghost_coord_x_env_'+ str(i)] = orange_ghost_coord[0]
                CustomCallbackA.main_data_dict[key]['orange_ghost_coord_y_env_'+ str(i)] = orange_ghost_coord[1]
                CustomCallbackA.main_data_dict[key]['to_orange_ghost_env_'+ str(i)] = to_orange_ghost

                CustomCallbackA.main_data_dict[key]['pill_one_eaten_env_'+ str(i)] = pill_eaten[0]
                CustomCallbackA.main_data_dict[key]['to_pill_one_env_'+ str(i)] = pill_dist[0]
                CustomCallbackA.main_data_dict[key]['pill_two_eaten_env_'+ str(i)] = pill_eaten[1]
                CustomCallbackA.main_data_dict[key]['to_pill_two_env_'+ str(i)] = pill_dist[1]
                CustomCallbackA.main_data_dict[key]['pill_three_eaten_env_'+ str(i)] = pill_eaten[2]
                CustomCallbackA.main_data_dict[key]['to_pill_three_env_'+ str(i)] = pill_dist[2]
                CustomCallbackA.main_data_dict[key]['pill_four_eaten_env_'+ str(i)] = pill_eaten[3]
                CustomCallbackA.main_data_dict[key]['to_pill_four_env_'+ str(i)] = pill_dist[3]

                # find blue ghosts, if any
                if(hasBlueGhost):
                    imagePeeler = GhostTracker()
                    # print("About to seek pacman at ", CustomCallbackA.step)
                    # ghost_coords = imagePeeler.wheresPacman(obs)
                    # ghost_coords = imagePeeler.wheresPacman(self.locals['obs'])
                    ghost_coords = imagePeeler.wheresPacman(cv.imread(filepath))
                    if(ghost_coords[0] != -1):
                        CustomCallbackA.main_data_dict[key]['dark_blue_ghost1_coord_x_env_'+ str(i)] = ghost_coords[0]
                        CustomCallbackA.main_data_dict[key]['dark_blue_ghost1_coord_y_env_'+ str(i)] = ghost_coords[1]
                    if(ghost_coords[2] != -1):
                        CustomCallbackA.main_data_dict[key]['dark_blue_ghost2_coord_x_env_'+ str(i)] = ghost_coords[2]
                        CustomCallbackA.main_data_dict[key]['dark_blue_ghost2_coord_y_env_'+ str(i)] = ghost_coords[3]
                    if(ghost_coords[4] != -1):
                        CustomCallbackA.main_data_dict[key]['dark_blue_ghost3_coord_x_env_'+ str(i)] = ghost_coords[4]
                        CustomCallbackA.main_data_dict[key]['dark_blue_ghost3_coord_y_env_'+ str(i)] = ghost_coords[5]
                    if(ghost_coords[6] != -1):
                        CustomCallbackA.main_data_dict[key]['dark_blue_ghost4_coord_x_env_'+ str(i)] = ghost_coords[6]
                        CustomCallbackA.main_data_dict[key]['dark_blue_ghost4_coord_y_env_'+ str(i)] = ghost_coords[7]

    def find_life_game_info(self):
        env_info = OrderedDict()
        for env_num in range(self.num_envs):
            env_info[env_num] = {}
            env_info[env_num]['total_life'] = env_info[env_num]['total_game'] = \
                env_info[env_num]['steps_life'] = env_info[env_num]['steps_game'] = 1
            env_info[env_num]['prev_life'] = 3
            env_info[env_num]['life_reward'] = env_info[env_num]['game_reward'] = \
                env_info[env_num]['total_reward'] = 0

        # TODO: move this check elsewhere
        if(self.isLives):
            # TODO: rewrite as a loop
            for key, value in CustomCallbackA.main_data_dict.items():  
                for i in range(self.num_envs):
                    is_end_of_game = False

                    CustomCallbackA.main_data_dict[key]['total_game_env_'+str(i)] = env_info[i]['total_game']
                    
                    # end of game
                    if(value['lives_env_' + str(i)] == 0):
                        CustomCallbackA.main_data_dict[key]['game_reward_env_'+str(i)] = env_info[i]['game_reward']
                        # reset_life
                        env_info[i]['total_game'] += 1
                        env_info[i]['steps_life'] = 0
                        env_info[i]['steps_game'] = 0
                        env_info[i]['life_reward'] = 0
                        env_info[i]['game_reward'] = 0
                        is_end_of_game = True

                    env_info[i]['total_reward'] += value['step_reward_env_'+str(i)]
                    env_info[i]['life_reward']  += value['step_reward_env_'+str(i)]  
                    env_info[i]['game_reward']  += value['step_reward_env_'+str(i)] 
                    env_info[i]['prev_life'] = value['lives_env_'+str(i)]

                    # update info in main dict
                    CustomCallbackA.main_data_dict[key]['steps_life_env_'+str(i)] = env_info[i]['steps_life'] 
                    CustomCallbackA.main_data_dict[key]['steps_game_env_'+str(i)] = env_info[i]['steps_game']
                    # CustomCallbackA.main_data_dict[key]['total_game_env_'+str(i)] = env_info[i]['total_game']
                    CustomCallbackA.main_data_dict[key]['life_reward_env_'+str(i)] = env_info[i]['life_reward']
                    CustomCallbackA.main_data_dict[key]['total_reward_env_'+str(i)] = env_info[i]['total_reward']
                    CustomCallbackA.main_data_dict[key]['is_end_of_game_env_'+str(i)] = is_end_of_game

    
                    # lost a life (episode)
                    # record BEFORE lives is decremented
                    if(key != self.num_steps and value['lives_env_'+str(i)] != CustomCallbackA.main_data_dict[key+self.num_envs]['lives_env_'+str(i)]
                        and CustomCallbackA.main_data_dict[key+self.num_envs]['lives_env_'+str(i)] != 0):
                        CustomCallbackA.main_data_dict[key]['total_life_env_'+str(i)] = env_info[i]['total_life']
                        
                        env_info[i]['total_life']  += 1
                        env_info[i]['steps_life'] = 0
                        env_info[i]['life_reward'] = 0

                    env_info[i]['steps_life'] += 1
                    env_info[i]['steps_game'] += 1

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        
        # what timestep you think
        print("timestep ",CustomCallbackA.step)
        # what timestep a2c learn is on 
        print("a2c num timestep",self.num_timesteps)
       
        # TODO: add flag to save screenshots or not
        subfolder = os.path.join(self.directory, 'screen/')
        filepath = os.path.join(subfolder)
        # self.locals['obs'] gives black and white imgs
        obs = self.env.get_images()
        img_name = '_screenshot_' + str(self.num_timesteps)

        for i in range(self.num_envs):
            # TODO: make cleaner
            mpl.image.imsave(subfolder+"env_" + str(i) + img_name + "_.png", obs[i])

        # TODO: move out of this dict and add to dict with loop
        step_stats = {self.num_timesteps: {
            'num_timesteps': self.num_timesteps,
            'state': self.num_timesteps/self.num_envs,
            }
        }
        # add step to dict
        CustomCallbackA.main_data_dict.update(step_stats)
        key = self.num_timesteps
        # TODO: move back the action + step + lives stuff above, and move the rest to utils? 
        for i in range(self.num_envs):
            CustomCallbackA.main_data_dict[key]['action_env_'+str(i)] =  self.locals['actions'][i]
            CustomCallbackA.main_data_dict[key]['action_name_env_'+str(i)] =  self.actions[self.locals['actions'][i]]
            CustomCallbackA.main_data_dict[key]['step_reward_env_'+str(i)] = self.locals['rewards'][i]
            if(self.isLives == True):
                if(CustomCallbackA.step == 1):
                    CustomCallbackA.main_data_dict[key]['lives_env_'+str(i)] = 3
                if(CustomCallbackA.step >= 2):
                    CustomCallbackA.main_data_dict[key]['lives_env_'+str(i)] = self.locals['infos'][i]['ale.lives']
        
        # print("numsteps" , self.num_steps)
        # print("numenvs" , self.num_envs)
        if(CustomCallbackA.step == (self.num_steps/self.num_envs)):
            self.make_dataframes(self.df_list)
            self.df_to_csv("df_og.csv", self.df_list)
            self.df_to_parquet()
            
            # calculate new info
            self.find_item_locations()
            self.find_life_game_info()
            self.make_dataframes(self.df_list_mod)
            self.df_to_csv("df_mod.csv", self.df_list_mod)
            print("done!")
        
        CustomCallbackA.step = CustomCallbackA.step + 1
        return True
