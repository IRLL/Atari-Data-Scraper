"""
    Contains various image utility functions.
    Most importantly:
     *add_saliency_to_image to overlay frames with saliency maps
     *and generate_video to create videos from given frames.
     # Borrowed from: https://rdmilligan.wordpress.com/2015/01/24/detecting-objects-in-pac-man-mark-ii/
     NOTE: if we find that the code is taking too long to execute, and thus having a negative effect on background subtraction, we can simply adjust the sample rate
"""

import numpy as np

class Character(object):
           
    # initialise the character
    def __init__(self, name, lower_colour, upper_colour, enabled):
        self.name = name
        self.lower_colour = lower_colour
        self.upper_colour = upper_colour
        self.enabled = enabled
        self.current_coord = np.array([0,0])
        self.previous_coord = np.array([0,0])
        self.direction = np.array([0,0])
 
    # check for colour match
    def is_colour_match(self, colour):
        for i in range(3):
            if colour[i] < self.lower_colour[i] or colour[i] > self.upper_colour[i]:
                return False
        return True
 
    # set coordinates
    def set_coordinates(self, coordinates):
        self.previous_coord = np.array(self.current_coord)
        self.current_coord = np.array(coordinates)
 
    # get direction
    def get_direction(self):
        x_direction = self.current_coord[0] - self.previous_coord[0]
        y_direction = self.current_coord[1] - self.previous_coord[1]
        self.direction = np.array([x_direction, y_direction])
        return self.direction
