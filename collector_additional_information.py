import pandas as pd
import numpy as np
from collections import OrderedDict
import os
import colour_detection as cd
import argparse
import shutil

class Collector():
    def __init__(self, directory='results', num_timesteps=10, num_envs = 1):
        self.directory = directory
        self.num_timesteps = num_timesteps
        self.num_envs = num_envs
        path_to_original_csv = os.path.join(self.directory, 'df_og.csv')
        self.csv_input = pd.read_csv(path_to_original_csv)
        self.dict_orig = self.csv_input.to_dict()
        self.main_data_dict = OrderedDict()

    def make_dataframes(self):
        # Make the main Dataframe
        main_df = pd.DataFrame.from_dict(
            self.main_data_dict, orient='index')
        return main_df

    def find_life_game_info(self):
            env_info = OrderedDict()
            for env_num in range(self.num_envs):
                env_info[env_num] = {}
                env_info[env_num]['total_life'] = env_info[env_num]['total_game'] = \
                    env_info[env_num]['steps_life'] = env_info[env_num]['steps_game'] = 1
                env_info[env_num]['prev_life'] = 3
                env_info[env_num]['life_reward'] = env_info[env_num]['game_reward'] = \
                    env_info[env_num]['total_reward'] = 0

            if(True):
                num_rows = len(self.dict_orig["state"])
                for key in range (num_rows):  
                    for i in range(self.num_envs): 
                        is_end_of_game = False
                        self.main_data_dict[key]['total_game_env_'+str(i)] = env_info[i]['total_game']
                
                        # end of game
                        if(self.dict_orig['lives_env_' + str(i)][key]== 0):
                            self.main_data_dict[key]['game_reward_env_'+str(i)] = env_info[i]['game_reward']
                            # reset_life
                            env_info[i]['total_game'] += 1
                            env_info[i]['steps_life'] = 0
                            env_info[i]['steps_game'] = 0
                            env_info[i]['life_reward'] = 0
                            env_info[i]['game_reward'] = 0
                            is_end_of_game = True

                        env_info[i]['total_reward'] += self.dict_orig['step_reward_env_'+str(i)][key]
                        env_info[i]['life_reward']  += self.dict_orig['step_reward_env_'+str(i)][key]
                        env_info[i]['game_reward']  += self.dict_orig['step_reward_env_'+str(i)][key]
                        env_info[i]['prev_life'] = self.dict_orig['lives_env_'+str(i)][key]

                        # update info in main dict
                        self.main_data_dict[key]['steps_life_env_'+str(i)] = env_info[i]['steps_life'] 
                        self.main_data_dict[key]['steps_game_env_'+str(i)] = env_info[i]['steps_game']
                        self.main_data_dict[key]['life_reward_env_'+str(i)] = env_info[i]['life_reward']
                        self.main_data_dict[key]['total_reward_env_'+str(i)] = env_info[i]['total_reward']
                        self.main_data_dict[key]['is_end_of_game_env_'+str(i)] = is_end_of_game

                        # lost a life (episode)
                        # record BEFORE lives is decremented
                        if(key != num_rows-1 and self.dict_orig['lives_env_'+str(i)][key] != self.dict_orig['lives_env_'+str(i)][key+1]
                            and self.dict_orig['lives_env_'+str(i)][key+1] != 0):
                            self.main_data_dict[key]['total_life_env_'+str(i)] = env_info[i]['total_life']
                            
                            env_info[i]['total_life']  += 1
                            env_info[i]['steps_life'] = 0
                            env_info[i]['life_reward'] = 0

                        env_info[i]['steps_life'] += 1
                        env_info[i]['steps_game'] += 1

    def find_item_locations_pacman(self):
        print("find locs")
        key = 0
        for screen_num in range(self.num_envs, self.num_timesteps + self.num_envs , self.num_envs):
            self.main_data_dict[key] = {}
            for i in range(self.num_envs):
                filepath = self.directory + "/screen/env_" + str(i) + "_screenshot_" + str(screen_num) + "_.png"
                pacman_coord, pink_ghost_coord, red_ghost_coord, green_ghost_coord, orange_ghost_coord, to_pink_ghost, to_red_ghost, to_green_ghost, to_orange_ghost, pill_eaten, pill_dist, hasBlueGhost = cd.find_pacman_coords(
                    filepath)
                self.main_data_dict[key]['pacman_coord_x_env_'+ str(i)] = pacman_coord[0]
                self.main_data_dict[key]['pacman_coord_y_env_'+ str(i)] = pacman_coord[1]
                self.main_data_dict[key]['pink_ghost_coord_x_env_'+ str(i)] = pink_ghost_coord[0]
                self.main_data_dict[key]['pink_ghost_coord_y_env_'+ str(i)] = pink_ghost_coord[1]
                self.main_data_dict[key]['to_pink_ghost_env_' + str(i)] = to_pink_ghost
                self.main_data_dict[key]['red_ghost_coord_x_env_' + str(i) ] = red_ghost_coord[0]
                self.main_data_dict[key]['red_ghost_coord_y_env_'+ str(i)] = red_ghost_coord[1]
                self.main_data_dict[key]['to_red_ghost_env_'+ str(i)] = to_red_ghost
                self.main_data_dict[key]['green_ghost_coord_x_env_'+ str(i)] = green_ghost_coord[0]
                self.main_data_dict[key]['green_ghost_coord_y_env_'+ str(i)] = green_ghost_coord[1]
                self.main_data_dict[key]['to_green_ghost_env_'+ str(i)] = to_green_ghost
                self.main_data_dict[key]['orange_ghost_coord_x_env_'+ str(i)] = orange_ghost_coord[0]
                self.main_data_dict[key]['orange_ghost_coord_y_env_'+ str(i)] = orange_ghost_coord[1]
                self.main_data_dict[key]['to_orange_ghost_env_'+ str(i)] = to_orange_ghost

                self.main_data_dict[key]['pill_one_eaten_env_'+ str(i)] = pill_eaten[0]
                self.main_data_dict[key]['to_pill_one_env_'+ str(i)] = pill_dist[0]
                self.main_data_dict[key]['pill_two_eaten_env_'+ str(i)] = pill_eaten[1]
                self.main_data_dict[key]['to_pill_two_env_'+ str(i)] = pill_dist[1]
                self.main_data_dict[key]['pill_three_eaten_env_'+ str(i)] = pill_eaten[2]
                self.main_data_dict[key]['to_pill_three_env_'+ str(i)] = pill_dist[2]
                self.main_data_dict[key]['pill_four_eaten_env_'+ str(i)] = pill_eaten[3]
                self.main_data_dict[key]['to_pill_four_env_'+ str(i)] = pill_dist[3]

                # find blue ghosts, if any
                if(hasBlueGhost):
                    imagePeeler = GhostTracker()
                    ghost_coords = imagePeeler.wheresPacman(cv.imread(filepath))
                    if(ghost_coords[0] != -1):
                        self.main_data_dict[key]['dark_blue_ghost1_coord_x_env_'+ str(i)] = ghost_coords[0]
                        self.main_data_dict[key]['dark_blue_ghost1_coord_y_env_'+ str(i)] = ghost_coords[1]
                    if(ghost_coords[2] != -1):
                        self.main_data_dict[key]['dark_blue_ghost2_coord_x_env_'+ str(i)] = ghost_coords[2]
                        self.main_data_dict[key]['dark_blue_ghost2_coord_y_env_'+ str(i)] = ghost_coords[3]
                    if(ghost_coords[4] != -1):
                        self.main_data_dict[key]['dark_blue_ghost3_coord_x_env_'+ str(i)] = ghost_coords[4]
                        self.main_data_dict[key]['dark_blue_ghost3_coord_y_env_'+ str(i)] = ghost_coords[5]
                    if(ghost_coords[6] != -1):
                        self.main_data_dict[key]['dark_blue_ghost4_coord_x_env_'+ str(i)] = ghost_coords[6]
                        self.main_data_dict[key]['dark_blue_ghost4_coord_y_env_'+ str(i)] = ghost_coords[7]
            
            key += 1

    def find_item_locations_pong(self):
        subfolder = os.path.join(self.directory, 'screen/')
        key = 0
        for screen_num in range(self.num_envs, self.num_timesteps + self.num_envs , self.num_envs):
            self.main_data_dict[key] = {}
            for i in range(self.num_envs):
                filepath = subfolder + "env_" + str(i) + "_screenshot_" + str(screen_num) + "_.png"
                ball_coord, green_paddle_coord, brown_paddle_coord, distance = cd.find_pong_coords(filepath)
                self.main_data_dict[key]['ball_coord_x_env_'+ str(i)] = ball_coord[0]
                self.main_data_dict[key]['ball_coord_y_env_'+ str(i)] = ball_coord[1]
                self.main_data_dict[key]['green_paddle_coord_x_env_'+ str(i)] = green_paddle_coord[0]
                self.main_data_dict[key]['green_paddle_coord_y_env_'+ str(i)] = green_paddle_coord[1]
                self.main_data_dict[key]['brown_paddle_coord_x_env_'+ str(i)] = brown_paddle_coord[0]
                self.main_data_dict[key]['brown_paddle_coord_y_env_'+ str(i)] = brown_paddle_coord[1]
                self.main_data_dict[key]['paddle_ball_distance_env_'+ str(i)] = distance
            key += 1

    def find_life_game_info_dqn(self):
        total_life = total_game = steps_life = steps_game = 1
        prev_life = 3
        life_reward = 0
        total_reward = 0
        game_reward = 0
        num_rows = len(self.dict_orig["state"])
        for key in range (num_rows): 
            if(key < 2):
                self.main_data_dict[key]['step_reward'] = self.dict_orig['cumulative_episode_reward'][key]
            else:
                if(self.dict_orig['lives'][key] == 0):
                    self.main_data_dict[key]['step_reward'] = 0
                else:
                    self.main_data_dict[key]['step_reward'] = self.dict_orig['cumulative_episode_reward'][key] - \
                        self.dict_orig['cumulative_episode_reward'][key-1]
                
            life_reward += self.main_data_dict[key]['step_reward'] 
            total_reward += self.main_data_dict[key]['step_reward'] 
            if(True):
                # game over (epoch)
                self.main_data_dict[key]['steps_life'] = steps_life
                if(self.dict_orig['lives'][key] == 0):
                    self.main_data_dict[key]['game_reward'] = game_reward
                    # reset values
                    total_game += 1
                    steps_game = steps_life = 0
                    game_reward = 0
                    life_reward = 0
                self.main_data_dict[key]['steps_game'] = steps_game
                self.main_data_dict[key]['total_game'] = total_game

                self.main_data_dict[key]['total_reward'] = total_reward

                if(key != num_rows-1 and self.dict_orig['lives'][key] != self.dict_orig['lives'][key+1]
                    and self.dict_orig['lives'][key] != 0):

                    self.main_data_dict[key]['total_life'] = total_life
                    self.main_data_dict[key]['life_reward'] = life_reward
                    game_reward += life_reward
                    total_life += 1
                    steps_life = 0
                    life_reward = 0
                # normal step
                prev_life = self.dict_orig['lives'][key]
                steps_life += 1
                steps_game += 1


    def output_modified_csv(self):
        val = self.make_dataframes()
        for v in val:
            self.csv_input[v] = val[v]
        self.csv_input.to_csv(self.directory + "/df_mod.csv", index=False) 
        filepath = os.path.join(self.directory, "df_mod.csv")
        print("Making modified csv with additional info and path is: ")
        print(filepath) 
   
    def remove_screenshots(self):
        screenshot_path = os.path.join(self.directory, "screen")
        if os.path.exists(screenshot_path):
            print("Removing all screenshots from ", screenshot_path)
            shutil.rmtree(screenshot_path)

def main(directory, num_steps, num_envs, algo, env_name, isRemoveScreenshots):
    collector = Collector(directory, num_steps, num_envs)
    if(algo == "A2C" or algo == "PPO2"):
        if(env_name == "Pacman"):
            collector.find_item_locations_pacman()
            collector.find_life_game_info()
        elif(env_name == "Pong"):
            collector.find_item_locations_pong()
        collector.output_modified_csv()
    elif(algo == "DQN"):
        if(env_name == "Pacman"):
            collector.find_item_locations_pacman()
            collector.find_life_game_info_dqn()
        elif(env_name == "Pong"):
            collector.find_item_locations_pong()
        collector.output_modified_csv()
    if(isRemoveScreenshots):
        collector.remove_screenshots()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', help='relative path to the csv file', type=str, default="results")
    parser.add_argument('--num_envs', help='set the number of environments', type=int, default=1)
    parser.add_argument('--num_steps', help='set the number of steps/actions you want the agent to take', type=int, default=1000)
    parser.add_argument('--algo', help='set the algorithm with which to train this model', type=str, default="DQN")
    parser.add_argument('--env_name', help='environment to use in training', type=str, default="Pacman")
    parser.add_argument('--remove_screenshots', help='delete the screenshots', action='store_true', default=False)
    
    args = parser.parse_args()
    filepath = args.filepath
    num_steps = args.num_steps
    num_envs = args.num_envs
    algo = args.algo.upper()
    env_name = args.env_name
    isRemoveScreenshots = args.remove_screenshots
    main(filepath, num_steps * num_envs, num_envs, algo, env_name, isRemoveScreenshots)
    