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
    def __init__(self, verbose=0, env_actions=[], env=None, num_steps=10, dir='results/', isLives=False, og_env = "", num_envs = 4):
        super(CustomCallbackA, self).__init__(verbose)
        self.actions = env_actions
        self.env = env
        self.num_steps = num_steps
        self.directory = dir
        self.isLives = isLives
        self.og_env = og_env.unwrapped
        self.num_envs = num_envs
        print("num stepss", self.num_timesteps)
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

     # TODO: maybe move to a separate file
    def util(self):
        total_life_0 = total_life_1 = total_life_2 = total_life_3 = 1
        total_game_0 = total_game_1 = total_game_2 = total_game_3 = 1
        steps_life_0 = steps_life_1 = steps_life_2 = steps_life_3 = 1
        steps_game_0 = steps_game_1 = steps_game_2 = steps_game_3 = 1
        prev_life_0 = prev_life_1 = prev_life_2 = prev_life_3 = 3
        episode_reward_0 = episode_reward_1 = episode_reward_2 = episode_reward_3 = 0
        total_reward_0 = total_reward_1 = total_reward_2 = total_reward_3 = 0
        game_reward_0 = game_reward_1 = game_reward_2 = game_reward_3 = 0

        print("in util func")
        for key, value in CustomCallbackA.main_data_dict.items():  
            if(self.isLives):
                # env_0
                # game over (epoch)
                if(value['lives_env_0'] == 0):
                    total_game_0 += 1
                    steps_game_0 = steps_life_0 = 0
                    game_reward_0 = 0
                    episode_reward_0 = 0

                # lost a life (episode)
                # record BEFORE lives is decremented
                elif(key != self.num_steps and value['lives_env_0'] != CustomCallbackA.main_data_dict[key+self.num_envs]['lives_env_0']):
                    CustomCallbackA.main_data_dict[key]['total_life_env_0'] = total_life_0
                    total_life_0 += 1
                    steps_life_0 = 1
                    episode_reward_0 = 0

                total_reward_0 += CustomCallbackA.main_data_dict[key]['step_reward_env_0'] 
                prev_life = value['lives_env_0']
                CustomCallbackA.main_data_dict[key]['steps_life_env_0'] = steps_life_0
                CustomCallbackA.main_data_dict[key]['steps_game_env_0'] = steps_game_0
                CustomCallbackA.main_data_dict[key]['total_game_env_0'] = total_game_0
                CustomCallbackA.main_data_dict[key]['total_reward_env_0'] = total_reward_0

                steps_life_0 += 1
                steps_game_0 += 1

                # env_1
                # game over (epoch)
                if(value['lives_env_1'] == 0):
                    total_game_1 += 1
                    steps_game_1 = steps_life_1 = 0
                    game_reward_1 = 0
                    episode_reward_1 = 0

                # lost a life (episode)
                # record BEFORE lives is decremented
                elif(key != self.num_steps and value['lives_env_1'] != CustomCallbackA.main_data_dict[key+self.num_envs]['lives_env_1']):
                    CustomCallbackA.main_data_dict[key]['total_life_env_1'] = total_life_1
                    total_life_1 += 1
                    steps_life_1 = 1
                    episode_reward_1 = 0

                total_reward_1 += CustomCallbackA.main_data_dict[key]['step_reward_env_1'] 
                prev_life_1 = value['lives_env_1']
                CustomCallbackA.main_data_dict[key]['steps_life_env_1'] = steps_life_1
                
                CustomCallbackA.main_data_dict[key]['steps_game_env_1'] = steps_game_1
                CustomCallbackA.main_data_dict[key]['total_game_env_1'] = total_game_1
                CustomCallbackA.main_data_dict[key]['total_reward_env_1'] = total_reward_1

                steps_life_1 += 1
                steps_game_1 += 1

                # env_2
                # game over (epoch)
                if(value['lives_env_2'] == 0):
                    total_game_2 += 1
                    steps_game_2 = steps_life_2 = 0
                    game_reward_2 = 0
                    episode_reward_2 = 0

                # lost a life (episode)
                # record BEFORE lives is decremented
                elif(key != self.num_steps and value['lives_env_2'] != CustomCallbackA.main_data_dict[key+self.num_envs]['lives_env_2']):
                    CustomCallbackA.main_data_dict[key]['total_life_env_2'] = total_life_2
                    total_life_2 += 1
                    steps_life_2 = 1
                    episode_reward_2 = 0

                total_reward_2 += CustomCallbackA.main_data_dict[key]['step_reward_env_2'] 
                prev_life_2 = value['lives_env_2']
                CustomCallbackA.main_data_dict[key]['steps_life_env_2'] = steps_life_2
                
                CustomCallbackA.main_data_dict[key]['steps_game_env_2'] = steps_game_2
                CustomCallbackA.main_data_dict[key]['total_game_env_2'] = total_game_2
                CustomCallbackA.main_data_dict[key]['total_reward_env_2'] = total_reward_2

                steps_life_2 += 1
                steps_game_2 += 1

                # env_3
                # game over (epoch)
                if(value['lives_env_3'] == 0):
                    total_game_3 += 1
                    steps_game_3 = steps_life_3 = 0
                    game_reward_3 = 0
                    episode_reward_3 = 0

                # lost a life (episode)
                # record BEFORE lives is decremented
                elif(key != self.num_steps and value['lives_env_3'] != CustomCallbackA.main_data_dict[key+self.num_envs]['lives_env_3']):
                    CustomCallbackA.main_data_dict[key]['total_life_env_3'] = total_life_3
                    total_life_3 += 1
                    steps_life_3 = 1
                    episode_reward_3 = 0

                total_reward_3 += CustomCallbackA.main_data_dict[key]['step_reward_env_3'] 
                prev_life_3 = value['lives_env_3']
                CustomCallbackA.main_data_dict[key]['steps_life_env_3'] = steps_life_3
                
                CustomCallbackA.main_data_dict[key]['steps_game_env_3'] = steps_game_3
                CustomCallbackA.main_data_dict[key]['total_game_env_3'] = total_game_3
                CustomCallbackA.main_data_dict[key]['total_reward_env_3'] = total_reward_3

                steps_life_3 += 1
                steps_game_3 += 1
            
        
 

    # TODO: display total reward somewhere?? 
    def total_episode_reward_logger(self, rew_acc, rewards, masks, writer, steps):
        """
        calculates the cumulated episode reward, and prints to tensorflow log the output
        :param rew_acc: (np.array float) the total running reward
        :param rewards: (np.array float) the rewards
        :param masks: (np.array bool) the end of episodes
        :param writer: (TensorFlow Session.writer) the writer to log to
        :param steps: (int) the current timestep
        :return: (np.array float) the updated total running reward
        :return: (np.array float) the updated total running reward
        """

        for env_idx in range(rewards.shape[0]):
            dones_idx = np.sort(np.argwhere(masks[env_idx]))

            if len(dones_idx) == 0:
                rew_acc[env_idx] += sum(rewards[env_idx])
            else:
                rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
                # summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                # writer.add_summary(summary, steps + dones_idx[0, 0])
                for k in range(1, len(dones_idx[:, 0])):
                    rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k - 1, 0]:dones_idx[k, 0]])
                    #summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                    #writer.add_summary(summary, steps + dones_idx[k, 0])
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])

        return rew_acc
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
            # 'action_env_0': self.locals['actions'][0],
            # 'action_env_1': self.locals['actions'][1],
            # 'action_env_2': self.locals['actions'][2],
            # 'action_env_3': self.locals['actions'][3],
            # 'action_name_env_0': self.actions[self.locals['actions'][0]],
            # 'action_name_env_1': self.actions[self.locals['actions'][1]],
            # 'action_name_env_2': self.actions[self.locals['actions'][2]],
            # 'action_name_env_3': self.actions[self.locals['actions'][3]],
            # 'step_reward_env_0': self.locals['rewards'][0],
            # 'step_reward_env_1': self.locals['rewards'][1],
            # 'step_reward_env_2': self.locals['rewards'][2],
            # 'step_reward_env_3': self.locals['rewards'][3],
            'state': self.num_timesteps/self.num_envs,
            #'lives':self.locals['infos']['ale.lives']
            }
        }
        # add step to dict
        CustomCallbackA.main_data_dict.update(step_stats)
        key = self.num_timesteps
        for i in range(self.num_envs):
            CustomCallbackA.main_data_dict[key]['action_env_'+str(i)] =  self.locals['actions'][i]
            CustomCallbackA.main_data_dict[key]['action_name_env_'+str(i)] =  self.actions[self.locals['actions'][i]]
            CustomCallbackA.main_data_dict[key]['step_reward_env_'+str(i)] = self.locals['rewards'][i]
            if(self.isLives == True):
                if(CustomCallbackA.step == 1):
                    CustomCallbackA.main_data_dict[key]['lives_env_'+str(i)] = 3
                if(CustomCallbackA.step >= 2):
                    CustomCallbackA.main_data_dict[key]['lives_env_'+str(i)] = self.locals['infos'][i]['ale.lives']

        # if(self.isLives == True and CustomCallbackA.step == 1):
        #     for i in range(self.num_envs):
        #         CustomCallbackA.main_data_dict[key]['lives_env_'+str(i)] = 3
        # if(self.isLives == True and CustomCallbackA.step >= 2):
        #     for i in range(self.num_envs):
        #         CustomCallbackA.main_data_dict[key]['lives_env_'+str(i)] = self.locals['infos'][i]['ale.lives']
        
        pacman_coord, pink_ghost_coord, red_ghost_coord, green_ghost_coord, orange_ghost_coord, to_pink_ghost, to_red_ghost, to_green_ghost, to_orange_ghost, pill_eaten, pill_dist, hasBlueGhost = cd.find_all_coords(
            subfolder+"env_0"+img_name+"_.png")
        CustomCallbackA.main_data_dict[key]['pacman_coord_x'] = pacman_coord[0]
        CustomCallbackA.main_data_dict[key]['pacman_coord_y'] = pacman_coord[1]
        CustomCallbackA.main_data_dict[key]['pink_ghost_coord_x'] = pink_ghost_coord[0]
        CustomCallbackA.main_data_dict[key]['pink_ghost_coord_y'] = pink_ghost_coord[1]
        CustomCallbackA.main_data_dict[key]['to_pink_ghost'] = to_pink_ghost
        CustomCallbackA.main_data_dict[key]['red_ghost_coord_x'] = red_ghost_coord[0]
        CustomCallbackA.main_data_dict[key]['red_ghost_coord_y'] = red_ghost_coord[1]
        CustomCallbackA.main_data_dict[key]['to_red_ghost'] = to_red_ghost
        CustomCallbackA.main_data_dict[key]['green_ghost_coord_x'] = green_ghost_coord[0]
        CustomCallbackA.main_data_dict[key]['green_ghost_coord_y'] = green_ghost_coord[1]
        CustomCallbackA.main_data_dict[key]['to_green_ghost'] = to_green_ghost
        CustomCallbackA.main_data_dict[key]['orange_ghost_coord_x'] = orange_ghost_coord[0]
        CustomCallbackA.main_data_dict[key]['orange_ghost_coord_y'] = orange_ghost_coord[1]
        CustomCallbackA.main_data_dict[key]['to_orange_ghost'] = to_orange_ghost

        CustomCallbackA.main_data_dict[key]['pill_one_eaten'] = pill_eaten[0]
        CustomCallbackA.main_data_dict[key]['to_pill_one'] = pill_dist[0]
        CustomCallbackA.main_data_dict[key]['pill_two_eaten'] = pill_eaten[1]
        CustomCallbackA.main_data_dict[key]['to_pill_two'] = pill_dist[1]
        CustomCallbackA.main_data_dict[key]['pill_three_eaten'] = pill_eaten[2]
        CustomCallbackA.main_data_dict[key]['to_pill_three'] = pill_dist[2]
        CustomCallbackA.main_data_dict[key]['pill_four_eaten'] = pill_eaten[3]
        CustomCallbackA.main_data_dict[key]['to_pill_four'] = pill_dist[3]

        # find blue ghosts, if any
        if(hasBlueGhost):
            imagePeeler = GhostTracker()
            # print("About to seek pacman at ", CustomCallbackA.step)
            # ghost_coords = imagePeeler.wheresPacman(obs)
            # ghost_coords = imagePeeler.wheresPacman(self.locals['obs'])
            ghost_coords = imagePeeler.wheresPacman(cv.imread(subfolder+"env_0"+img_name+"_0.png"))
            if(ghost_coords[0] != -1):
                CustomCallbackA.main_data_dict[key]['dark_blue_ghost1_coord_x'] = ghost_coords[0]
                CustomCallbackA.main_data_dict[key]['dark_blue_ghost1_coord_y'] = ghost_coords[1]
            if(ghost_coords[2] != -1):
                CustomCallbackA.main_data_dict[key]['dark_blue_ghost2_coord_x'] = ghost_coords[2]
                CustomCallbackA.main_data_dict[key]['dark_blue_ghost2_coord_y'] = ghost_coords[3]
            if(ghost_coords[4] != -1):
                CustomCallbackA.main_data_dict[key]['dark_blue_ghost3_coord_x'] = ghost_coords[4]
                CustomCallbackA.main_data_dict[key]['dark_blue_ghost3_coord_y'] = ghost_coords[5]
            if(ghost_coords[6] != -1):
                CustomCallbackA.main_data_dict[key]['dark_blue_ghost4_coord_x'] = ghost_coords[6]
                CustomCallbackA.main_data_dict[key]['dark_blue_ghost4_coord_y'] = ghost_coords[7]

        if(CustomCallbackA.step == (self.num_steps/self.num_envs)):
            # print("dictionary ", CustomCallbackA.main_data_dict)
            self.make_dataframes(self.df_list)
            self.df_to_csv("df_og.csv", self.df_list)
            self.df_to_parquet()
            # test if parquet file is correctly created
            # print("reading parquet file")
            # print(pd.read_parquet(os.path.join(self.directory,  "df.parquet")))

        #     # calculate new info
            self.util()
            self.make_dataframes(self.df_list_mod)
            self.df_to_csv("df_mod.csv", self.df_list_mod)
            # self.df_to_parquet()
            print("done!")
        CustomCallbackA.step = CustomCallbackA.step + 1
        return True
