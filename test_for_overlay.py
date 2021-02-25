import pandas as pd
import numpy as np
from collections import OrderedDict
import os
import colour_detection as cd

main_data_dict = OrderedDict()
# csv_input['New_col'] = df["column1"]
# csv_input.to_csv('A2C_test/mod_output.csv', index=False)
# dataframe is a db table

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

def find_item_locations_pacman():
        # subfolder = os.path.join(directory, 'screen/')
        for screen_num in range(num_envs, num_timesteps + num_envs , num_envs):
            key = screen_num
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

directory = "A2C_test"
num_envs = 2
num_timesteps = 20
df_list = []
csv_input = pd.read_csv(directory + '/df_og.csv')
# print("csv ", type(csv_input))
# print("col ", csv_input['step_reward_env_0'])
find_item_locations_pacman()
val = make_dataframes(df_list)
#print("main", main_data_dict)
# print("df list ", df_list)
# print("df list type", type(df_list))
print("val " ,val)
# print("val " , type(val))
for v in val:
    print("v ", v)
    #csv_input[v] = val[v]
print("pacman coord x env 0", val["pacman_coord_x_env_0"])
    # csv_input[v] = df_list[v]
print("here ", df_list['pacman_coord_x_env_0'])
# numpy_data = np.array([1,2,3,4,5,6,7,8,9,10])
# df = pd.DataFrame(data=numpy_data, columns=["column1"])
# csv_input['New_col'] = df["column1"]
#csv_input.to_csv('A2C_test/mod_output_with_coords.csv', index=False)
# dataframe is a db table