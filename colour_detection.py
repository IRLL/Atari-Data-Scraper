import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def find_element_centroid_pacman(img, colour, coord):
    y,x = np.where(np.all(img == colour, axis=2))
    pairs = []
    for i in range(len(x)):
        pairs.append([x[i],y[i]])
   
    if(len(x) != 0 and len(y) != 0):
        # calculate centroid
        coordx, coordy = np.mean(pairs, axis = 0)
        coord[0] = round(coordx)
        coord[1] = round(coordy)

# Duplicate colours are found in the game board for pong, so extra bounding needs to be added
def find_element_centroid_pong(img, colour, coord):
    y,x = np.where(np.all(img == colour, axis=2))
    pairs = []
    
    for i in range(len(x)):
        # restricts the bounds of the play environment
        if(y[i] >= 33.5 and y[i] <= 193.5):
            pairs.append([x[i],y[i]])
    if(len(pairs) != 0):
        # calculate centroid
        coordx, coordy = np.mean(pairs, axis = 0)
        coord[0] = round(coordx)
        coord[1] = round(coordy)


def find_distances(coordA, coordB, dist):
    dist[0] = abs(coordA[0] - coordB[0]) + abs(coordA[1] - coordB[1])

def find_blue_ghosts(img):
    image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_blue = np.array([60, 100, 188])
    upper_blue = np.array([150,255, 255])
    mask = cv.inRange(image, lower_blue, upper_blue)
    return np.count_nonzero(mask) > 0

def check_pills():
    for i in range(4):
        if(abs(pacman_coord[0] - pill_locs[i][0]) <= 3 and abs(pacman_coord[1] - pill_locs[i][1]) <= 3):
            pill_eaten[i] = True
        pill_dist[i] = abs(pacman_coord[0] - pill_locs[i][0]) + abs(pacman_coord[1] - pill_locs[i][1]) 

# Declare colours for Pacman. OpenCV uses BGR not RGB
pacman_colour = [74, 164, 210]
pink_ghost_colour = [179, 89, 198]
red_ghost_colour = [72, 72, 200]
# called blue ghost sometimes
green_ghost_colour = [153, 184, 84]
orange_ghost_colour = [48, 122, 180]
dark_blue_ghost = [194, 114, 66]

# Declare and initialize coordinates for Pacman
pacman_coord = [0, 0]
pink_ghost_coord = [0, 0]
red_ghost_coord = [0, 0]
green_ghost_coord = [0, 0]
orange_ghost_coord = [0, 0]

# Declare distances for Pacman
to_pink_ghost = [0]
to_red_ghost = [0]
to_green_ghost = [0]
to_orange_ghost = [0]

# Declare pill info for Pacman
power_pill_top_left = [19.5, 18]
power_pill_btm_left = [19.5, 150]
power_pill_top_right = [300.5, 18]
power_pill_btm_right = [300.5, 150]
pill_locs = []
pill_locs.append(power_pill_top_left)
pill_locs.append(power_pill_top_right)
pill_locs.append(power_pill_btm_right)
pill_locs.append(power_pill_btm_left)
# pills 1,2,3,4
pill_eaten = [False, False, False, False]
# top left, top right, btm right, btm left
pill_dist = [0,0,0,0]


def find_pacman_coords(im):
    img = cv.imread(im)
    find_element_centroid_pacman(img, pacman_colour, pacman_coord)
    find_element_centroid_pacman(img, pink_ghost_colour, pink_ghost_coord)
    find_distances(pink_ghost_coord, pacman_coord, to_pink_ghost)
    find_element_centroid_pacman(img, red_ghost_colour, red_ghost_coord)
    find_distances(red_ghost_coord, pacman_coord, to_red_ghost)
    find_element_centroid_pacman(img, green_ghost_colour, green_ghost_coord)
    find_distances(green_ghost_coord, pacman_coord, to_green_ghost)
    find_element_centroid_pacman(img, orange_ghost_colour, orange_ghost_coord)
    find_distances(orange_ghost_coord, pacman_coord, to_orange_ghost)

    check_pills()

    hasBlueGhost = find_blue_ghosts(img)

    return pacman_coord, pink_ghost_coord, red_ghost_coord, green_ghost_coord, orange_ghost_coord, to_pink_ghost[0], to_red_ghost[0], to_green_ghost[0], to_orange_ghost[0], pill_eaten, pill_dist, hasBlueGhost

# Declare coords and distances for Pong
ball_colour = [236, 236, 236]
green_paddle_colour = [92, 186, 92]
brown_paddle_colour = [74, 130, 213]
ball_coord = [0,0]
green_paddle_coord = [0,0]
brown_paddle_coord = [0,0]
dist_ball_green_paddle = [0]

def find_pong_coords(im):
    img = cv.imread(im)
    find_element_centroid_pong(img, ball_colour, ball_coord)
    find_element_centroid_pong(img, green_paddle_colour, green_paddle_coord)
    find_element_centroid_pong(img, brown_paddle_colour, brown_paddle_coord)
    find_distances(green_paddle_coord, ball_coord, dist_ball_green_paddle)
    return ball_coord, green_paddle_coord, brown_paddle_coord, dist_ball_green_paddle[0]

# ADDING ADDITIONAL ATARI ENVS
# TODO: Declare the colours of items you want to track in BGR array 
# item_colour = [BBB, GGG, RRR]
# TODO: Declare arrays for coordinates of item and distance to item
# item_coord = [0,0]
# dist_to_item = [0]

# Sample function:
# def find_env_coords(im):
#     find_item_centroid_env(...)
#     find_item_dist(...)
#     return item_coord, dist_to_item






