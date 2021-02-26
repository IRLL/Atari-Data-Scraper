import pandas as pd
import numpy as np
from collections import OrderedDict
import os
import colour_detection as cd

main_data_dict = OrderedDict()
# csv_input['New_col'] = df["column1"]
# csv_input.to_csv('A2C_test/mod_output.csv', index=False)
# dataframe is a db table

# directory = "A2C_test"
directory = "DQN_test"
# num_envs = 2
# num_timesteps = 20
num_envs = 1
num_timesteps = 10
df_list = []
csv_input = pd.read_csv(directory + '/df_og.csv')
dict_orig = csv_input.to_dict()
# print("value content ", dict_orig["lives_env_0"][0])
# print("value ", value)

def make_dataframes(df):
    # Make the main Dataframe
    main_df = pd.DataFrame.from_dict(
        main_data_dict, orient='index')
    
    # call to save last items as seperate df
    # self.save_last_line(args.stream_folder, main_df)

    # Now that we've parsed down the main df, load all into our list
    # of DFs and our list of Names
    # self.df_list.append(main_df)
    df.append(main_df)
    return main_df

def df_to_csv(filename, df_list):
    for df in df_list:
        # filename = "df.csv"
        filepath = os.path.join(directory, filename)
        print("Making csvs and path is: ")
        print(filepath)
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', index=False, header=False)
        else:
            df.to_csv(filepath, mode='a', index=False)

def find_life_game_info():
        env_info = OrderedDict()
        for env_num in range(num_envs):
            env_info[env_num] = {}
            env_info[env_num]['total_life'] = env_info[env_num]['total_game'] = \
                env_info[env_num]['steps_life'] = env_info[env_num]['steps_game'] = 1
            env_info[env_num]['prev_life'] = 3
            env_info[env_num]['life_reward'] = env_info[env_num]['game_reward'] = \
                env_info[env_num]['total_reward'] = 0

        # TODO: move this check elsewhere?
        if(True):
            #for key, value in dict_orig.items():
            for i in range(num_envs):
                for key in range (len(dict_orig)):  
                    print("env ", i, "step ", key)
                
                    is_end_of_game = False

                    main_data_dict[key]['total_game_env_'+str(i)] = env_info[i]['total_game']
            
                    # end of game
                    if(dict_orig['lives_env_' + str(i)][key]== 0):
                        main_data_dict[key]['game_reward_env_'+str(i)] = env_info[i]['game_reward']
                        # reset_life
                        env_info[i]['total_game'] += 1
                        env_info[i]['steps_life'] = 0
                        env_info[i]['steps_game'] = 0
                        env_info[i]['life_reward'] = 0
                        env_info[i]['game_reward'] = 0
                        is_end_of_game = True

                    env_info[i]['total_reward'] += dict_orig['step_reward_env_'+str(i)][key]
                    env_info[i]['life_reward']  += dict_orig['step_reward_env_'+str(i)][key]
                    env_info[i]['game_reward']  += dict_orig['step_reward_env_'+str(i)][key]
                    env_info[i]['prev_life'] = dict_orig['lives_env_'+str(i)][key]

                    # update info in main dict
                    main_data_dict[key]['steps_life_env_'+str(i)] = env_info[i]['steps_life'] 
                    main_data_dict[key]['steps_game_env_'+str(i)] = env_info[i]['steps_game']
                    # CustomCallbackA.main_data_dict[key]['total_game_env_'+str(i)] = env_info[i]['total_game']
                    main_data_dict[key]['life_reward_env_'+str(i)] = env_info[i]['life_reward']
                    main_data_dict[key]['total_reward_env_'+str(i)] = env_info[i]['total_reward']
                    main_data_dict[key]['is_end_of_game_env_'+str(i)] = is_end_of_game

                    # lost a life (episode)
                    # record BEFORE lives is decremented
                    if(key != len(dict_orig)-1 and dict_orig['lives_env_'+str(i)][key] != dict_orig['lives_env_'+str(i)][key+1]
                        and dict_orig['lives_env_'+str(i)][key+1] != 0):
                        main_data_dict[key]['total_life_env_'+str(i)] = env_info[i]['total_life']
                        
                        env_info[i]['total_life']  += 1
                        env_info[i]['steps_life'] = 0
                        env_info[i]['life_reward'] = 0

                    env_info[i]['steps_life'] += 1
                    env_info[i]['steps_game'] += 1

def find_item_locations_pacman():
        # subfolder = os.path.join(directory, 'screen/')
        # TODO: change for loop
        key = 0
        for screen_num in range(num_envs, num_timesteps + num_envs , num_envs):
            # key = (screen_num / num_envs) - 1
            # print("new ind ", key)
            
            main_data_dict[key] = {}
            for i in range(num_envs):
                filepath = directory + "/screen/env_" + str(i) + "_screenshot_" + str(screen_num) + "_.png"
                print("filepath ", filepath)
                pacman_coord, pink_ghost_coord, red_ghost_coord, green_ghost_coord, orange_ghost_coord, to_pink_ghost, to_red_ghost, to_green_ghost, to_orange_ghost, pill_eaten, pill_dist, hasBlueGhost = cd.find_all_coords(
                    filepath)
                main_data_dict[key]['pacman_coord_x_env_'+ str(i)] = pacman_coord[0]
                main_data_dict[key]['pacman_coord_y_env_'+ str(i)] = pacman_coord[1]
                main_data_dict[key]['pink_ghost_coord_x_env_'+ str(i)] = pink_ghost_coord[0]
                main_data_dict[key]['pink_ghost_coord_y_env_'+ str(i)] = pink_ghost_coord[1]
                main_data_dict[key]['to_pink_ghost_env_' + str(i)] = to_pink_ghost
                main_data_dict[key]['red_ghost_coord_x_env_' + str(i) ] = red_ghost_coord[0]
                main_data_dict[key]['red_ghost_coord_y_env_'+ str(i)] = red_ghost_coord[1]
                main_data_dict[key]['to_red_ghost_env_'+ str(i)] = to_red_ghost
                main_data_dict[key]['green_ghost_coord_x_env_'+ str(i)] = green_ghost_coord[0]
                main_data_dict[key]['green_ghost_coord_y_env_'+ str(i)] = green_ghost_coord[1]
                main_data_dict[key]['to_green_ghost_env_'+ str(i)] = to_green_ghost
                main_data_dict[key]['orange_ghost_coord_x_env_'+ str(i)] = orange_ghost_coord[0]
                main_data_dict[key]['orange_ghost_coord_y_env_'+ str(i)] = orange_ghost_coord[1]
                main_data_dict[key]['to_orange_ghost_env_'+ str(i)] = to_orange_ghost

                main_data_dict[key]['pill_one_eaten_env_'+ str(i)] = pill_eaten[0]
                main_data_dict[key]['to_pill_one_env_'+ str(i)] = pill_dist[0]
                main_data_dict[key]['pill_two_eaten_env_'+ str(i)] = pill_eaten[1]
                main_data_dict[key]['to_pill_two_env_'+ str(i)] = pill_dist[1]
                main_data_dict[key]['pill_three_eaten_env_'+ str(i)] = pill_eaten[2]
                main_data_dict[key]['to_pill_three_env_'+ str(i)] = pill_dist[2]
                main_data_dict[key]['pill_four_eaten_env_'+ str(i)] = pill_eaten[3]
                main_data_dict[key]['to_pill_four_env_'+ str(i)] = pill_dist[3]

                # find blue ghosts, if any
                if(hasBlueGhost):
                    imagePeeler = GhostTracker()
                    ghost_coords = imagePeeler.wheresPacman(cv.imread(filepath))
                    if(ghost_coords[0] != -1):
                        main_data_dict[key]['dark_blue_ghost1_coord_x_env_'+ str(i)] = ghost_coords[0]
                        main_data_dict[key]['dark_blue_ghost1_coord_y_env_'+ str(i)] = ghost_coords[1]
                    if(ghost_coords[2] != -1):
                        main_data_dict[key]['dark_blue_ghost2_coord_x_env_'+ str(i)] = ghost_coords[2]
                        main_data_dict[key]['dark_blue_ghost2_coord_y_env_'+ str(i)] = ghost_coords[3]
                    if(ghost_coords[4] != -1):
                        main_data_dict[key]['dark_blue_ghost3_coord_x_env_'+ str(i)] = ghost_coords[4]
                        main_data_dict[key]['dark_blue_ghost3_coord_y_env_'+ str(i)] = ghost_coords[5]
                    if(ghost_coords[6] != -1):
                        main_data_dict[key]['dark_blue_ghost4_coord_x_env_'+ str(i)] = ghost_coords[6]
                        main_data_dict[key]['dark_blue_ghost4_coord_y_env_'+ str(i)] = ghost_coords[7]
            # print("orddict ", main_data_dict[key])
            key += 1

def find_life_game_info_dqn():
        total_life = total_game = steps_life = steps_game = 1
        prev_life = 3
        episode_reward = 0
        total_reward = 0
        game_reward = 0
        print("in util func")
        print("dict_orig ", dict_orig)
        num_rows = len(dict_orig["state"])
        print("numrows", num_rows)
        for key in range (num_rows):
            print("key ", key)  
            if(key < 2):
                main_data_dict[key]['step_reward'] = dict_orig['cumulative_episode_reward'][key]
            else:
                if(dict_orig['lives'][key] == 0):
                    main_data_dict[key]['step_reward'] = 0
                else:
                    main_data_dict[key]['step_reward'] = dict_orig['cumulative_episode_reward'][key] - \
                        dict_orig['cumulative_episode_reward'][key-1]
                
            episode_reward += main_data_dict[key]['step_reward'] 
            total_reward += main_data_dict[key]['step_reward'] 
            if(True):
                # game over (epoch)
                main_data_dict[key]['steps_life'] = steps_life
                if(dict_orig['lives'][key] == 0):
                    main_data_dict[key]['game_reward'] = game_reward
                    # reset values
                    total_game += 1
                    steps_game = steps_life = 0
                    game_reward = 0
                    episode_reward = 0
                main_data_dict[key]['steps_game'] = steps_game
                main_data_dict[key]['total_game'] = total_game

                main_data_dict[key]['total_reward'] = total_reward

                if(key != num_rows-1 and dict_orig['lives'][key] != dict_orig['lives'][key+1]
                    and dict_orig['lives'][key] != 0):

                    # not sure if this is correct
                    main_data_dict[key]['total_life'] = total_life
                    main_data_dict[key]['episode_reward'] = episode_reward
                    game_reward += episode_reward
                    total_life += 1
                    steps_life = 0
                    episode_reward = 0
                # normal step
                prev_life = dict_orig['lives'][key]
                steps_life += 1
                steps_game += 1

# csv_input = pd.read_csv(directory + '/df_og.csv')
# value = csv_input.to_dict()
# print("value content ", value["lives_env_0"][0])
# print("value ", value)
# print("csv ", type(csv_input))
# print("col ", csv_input['step_reward_env_0'])
find_item_locations_pacman()
# find_life_game_info()
find_life_game_info_dqn()
val = make_dataframes(df_list)
print("dataframe result ", val)
# print("df list ", df_list)
# print("df list type", type(df_list))
# print("val of dataframes" ,val)
# print("val " , type(val))
for v in val:
    # print("v ", v)
    csv_input[v] = val[v]
# print("pacman coord x env 0", val["pacman_coord_x_env_0"])
# csv_input["test col"] = val["pacman_coord_x_env_0"]
# print("here ", df_list['pacman_coord_x_env_0'])
# numpy_data = np.array([1,2,3,4,5,6,7,8,9,10])
# df = pd.DataFrame(data=numpy_data, columns=["column1"])
# csv_input['New_col'] = df["column1"]
# csv_input.to_csv('A2C_test/compare_to_mod.csv', index=False)
csv_input.to_csv(directory + "/compare_to_mod.csv", index=False)
# dataframe is a db table