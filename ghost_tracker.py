"""
    Contains various image utility functions.
    Most importantly:
     *add_saliency_to_image to overlay frames with saliency maps
     *and generate_video to create videos from given frames.
     # Borowed from: https://rdmilligan.wordpress.com/2015/01/24/detecting-objects-in-pac-man-mark-ii/
     
     #BackgroundSubtractorMOG() it's at cv2.bgsegm.BackgroundSubtractorMOG(),to use you must install opencv-contrib-python
"""

import cv2
#import ImageGrab
import numpy as np
from characters import Character
import math

class GhostTracker():
    # constants
    HISTORY = 12
    SAMPLE_RATE = 1
    PIXEL_OFFSET = 20
    TEXT_OFFSET = 10
    # set up characters with arrays of BGR (blue, green, red)
#    pacman = Character("Ms. Pac-Man", [74, 148, 205], [86, 152, 212], True)
#    red_ghost = Character("Red", [67, 67, 190], [77, 77, 210], True)
#    pink_ghost = Character("Pink", [175, 84, 195], [183, 94, 205], True)
#    blue_ghost = Character("Blue", [140, 179, 80], [155, 187, 89], True)
#    orange_ghost = Character("Orange", [48, 115, 173], [73, 122, 211], True)
    pacman = Character("Ms. Pac-Man", [208, 160, 72], [212, 166, 76], True)
    red_ghost = Character("Red", [195, 70, 70], [207, 75, 75], True)
    pink_ghost = Character("Pink", [195, 84, 175], [205, 91, 181], True)
    blue_ghost = Character("Blue", [81, 179, 150], [88, 187, 155], True)
    orange_ghost = Character("Orange", [177, 120, 45], [184, 125, 50], True)
    
    #NEED TO GET THESE AND STORE PERMANENT?? WHAT ABOUT NEW MAZE???
#    power_pill = Character("PowerPill", [227, 110, 110], [229, 112, 112], True)
#    power_pill1 = Character("PowerPill1", [227, 110, 110], [229, 112, 112], True)
#    power_pill2 = Character("PowerPill2", [227, 110, 110], [229, 112, 112], True)
#    power_pill3 = Character("PowerPill3", [227, 110, 110], [229, 112, 112], True)
    
    #NEED TO GET THESE AND STORE PERMANENT?? WHAT ABOUT NEW MAZE???
    bg = Character("Blue Ghost 1", [60, 100, 188], [70, 120, 200], True)
    bg1 = Character("Blue Ghost 2", [60, 100, 188], [70, 120, 200], True)
    bg2 = Character("Blue Ghost 3", [60, 100, 188], [70, 120, 200], True)
    bg3 = Character("Blue Ghost 4", [60, 100, 188], [70, 120, 200], True)
    
    # Store in an array for easy access
    CHARACTERS = [pacman, red_ghost, pink_ghost, blue_ghost, orange_ghost]
    BGS = [bg, bg1, bg2, bg3]
#    POWER_PILLS = [power_pill, power_pill1, power_pill2, power_pill3]
    # set up background subtraction -- foreground of img is extracted for further processing
    FGBG = cv2.createBackgroundSubtractorMOG2()
    # set variables
    FLOW = (0,0)
    SAMPLE_COUNTER = 0
    SIMILARITY_THRESHOLD = 0.75
#    POWER_PILL_LOCS = []
    BG_LOCS = []
    
    # initialise the tracker
    def __init__(self, history=12, sample_rate=1, pixel_offset=20, text_offset=10, flow=(0,0), sample_counter=0):
        self.HISTORY = history
        self.SAMPLE_RATE = sample_rate
        self.PIXEL_OFFSET = pixel_offset
        self.TEXT_OFFSET = text_offset
        self.FLOW = flow
        self.SAMPLE_COUNTER = sample_counter
        return
        
    def close_coords(self, coord1, coord2, tol = 5):
        if (math.isclose(coord1[0],coord2[0], rel_tol=tol, abs_tol=0.0)):
            if (math.isclose(coord1[1],coord2[1], rel_tol=tol, abs_tol=0.0)):
                return True
        return False
    
    def checkPP(self, p_coord):
        for index, pill in enumerate(self.POWER_PILLS):
        # call fxn to see if coordinates are close
            close = close_coords(p_coord, pill.current_coord)
#            print("Close is: " + str(close))
            if (close and pill.enabled == True):
                pill.enabled = False
                pill.current_coord = (0,0)
                pill.previous_coord = (0,0)
                
    # grab screenshot
#    def grab_screenshot(self):
#        # May need to pass in frame of four added shots?
#        screenshot = ImageGrab.grab(bbox=(0,50,1300,900))
#        return cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)
     
    # get contours from image
    def get_contours(self, image):
        edges = cv2.Canny(image, 100, 200)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
     
    # get contour centroid
    def get_contour_centroid(self, contour):
        try:
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            return (cx, cy)
        except:
            return (0, 0)
     
    # draw contour detail on image
    def draw_contour(self, contour, image, coord, character):
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
        cv2.putText(image, "{} {}".format(character.name, coord), (coord[0] + self.TEXT_OFFSET, coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0))
     
    # get flow of characters
    def get_flow(self, flow, character):
        if character.name == "Ms. Pac-Man":
            return flow
     
        direction = character.get_direction()
        return (flow[0] + direction[0], flow[1] + direction[1])
     
    # draw flow of characters
    def draw_flow(self, flow, image):
        negative_threshold = -20
        positive_threshold = 20
     
        # draw East or West bar
        if(flow[0] < negative_threshold):
            cv2.rectangle(image,(0,0),(50,850),(0,255,0),-1)
        elif(flow[0] > positive_threshold):
            cv2.rectangle(image,(1250,0),(1300,850),(0,255,0),-1)
             
        # draw North or South bar
        if(flow[1] < negative_threshold):
            cv2.rectangle(image,(0,0),(1300,50),(0,255,0),-1)
        elif(flow[1] > positive_threshold):
            cv2.rectangle(image,(0,800),(1300,850),(0,255,0),-1)
     
        # print flow
        cv2.putText(image, "x {} : y {}".format(flow[0], flow[1]), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        return
     
    def wheresPacman(self, frame):
        bg_coords = [-1] * 8
        starting_index = [0,0,2,4,6]
        bg_counter = 0
        
        
        # apply mask 
        fgmask = self.FGBG.apply(frame, learningRate=0.5/self.HISTORY)
        # convert colour why from gray?
        fgoutput = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
             
        height, width = frame.shape[:2]
#        print("Image dimensions, H x W")
#        print(height, width)
        
        # get contours for objects in foreground
        contours = self.get_contours(fgmask)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        # find characters in foreground objects
        for contour in contours:

            # get centre coordinates of object
            coord = self.get_contour_centroid(contour)
            
            offset = 3
            # extract colour from object
            colour0 = frame[coord[1]][coord[0]]
            try:
                colour1 = frame[coord[1]][coord[0] - offset]
            except:
                colour1 = [0,0,0]
            try:
                colour2 = frame[coord[1]][coord[0] + offset]
            except:
                colour2 = [0,0,0]
            try:
                colour3 = frame[coord[1] - offset][coord[0]]
            except:
                colour3 = [0,0,0]
            try:
                colour4 = frame[coord[1] + offset][coord[0]]
            except:
                colour4 = [0,0,0]
            # print("**************")
            # print("Found color0: " + str(colour0) + " and " + str(colour1) + " and " + str(colour2) + " and " + str(colour3) + " and " + str(colour4))

            # loop all characters and attempt to match colour
            for character in self.CHARACTERS:
                if character.enabled and (character.is_colour_match(colour0) or character.is_colour_match(colour1) or character.is_colour_match(colour2) or character.is_colour_match(colour3) or character.is_colour_match(colour4)):
                        # print("For character: ")
                        # print(character.name)
                        character.set_coordinates(coord)
                        # flow is motion of image between two frames
                        self.FLOW = self.get_flow(self.FLOW, character)
                        if (character.name == "Orange" or character.name == "Blue" or character.name == "Pink" or character.name == "Red"):
                            # set BG_LOCS to 0
                            self.BG_LOCS = []
#                       print("Found color0: " + str(colour0) + " and " + str(colour1) + " and " + str(colour2) + " and " + str(colour3) + " and " + str(colour4))
                        #print("Coordinate: ")
                        # print(coord)
                        self.draw_contour(contour, fgoutput, coord, character)
                        character.direction = character.get_direction()
                        character.enabled = False
                        break
#            # loop all power pills and blue ghosts and attempt to match colour
#            print("PILL list is this long: " + str(len(self.POWER_PILL_LOCS)))
#            print(self.POWER_PILL_LOCS)
#            if len(self.POWER_PILL_LOCS) < 4:
#                for pill_loc in self.POWER_PILL_LOCS:
#                    if (self.close_coords(coord, pill_loc, tol = 50)):
#                        break
#                for pill in self.POWER_PILLS:
#                    print("Power pill: " + str(pill.name))
#                    print(self.POWER_PILL_LOCS)
#                    print(pill.current_coord)
#                    self.draw_contour(contour, fgoutput, pill.current_coord, pill)
#                    if pill.enabled and (pill.is_colour_match(colour0) and pill.is_colour_match(colour4)) and (not coord in self.POWER_PILL_LOCS):
#                        print("For character: ")
#                        print(pill.name)
#                        pill.set_coordinates(coord)
#                        print("For character: ")
#                        print(pill.name)
#                        print("Found color0: " + str(colour0) + " and " + str(colour1) + " and " + str(colour2) + " and " + str(colour3) + " and " + str(colour4))
#                        print("Coordinate: ")
#                        print(coord)
#                        self.POWER_PILL_LOCS.append(coord)
#    #                    self.draw_contour(contour, fgoutput, coord, pill)
#                        break
            # if ghosts are blue...
            if (self.BGS[0].is_colour_match(colour0) or self.BGS[0].is_colour_match(colour1) or self.BGS[0].is_colour_match(colour2) or self.BGS[0].is_colour_match(colour3) or self.BGS[0].is_colour_match(colour4)) and (bg_counter<4):
                bg = self.BGS[bg_counter]
                self.draw_contour(contour, fgoutput, bg.current_coord, bg)
                bg.set_coordinates(coord)
                self.BG_LOCS.append(np.array(coord))
                self.draw_contour(contour, fgoutput, coord, bg)
                self.BG_LOCS.append(coord)
                # print("Found color0: " + str(colour0) + " and " + str(colour1) + " and " + str(colour2) + " and " + str(colour3) + " and " + str(colour4))
                # print("Found ghost " + str(bg.name))
                # print("at: " + str(bg.current_coord))
                # print("here: ", str(bg.name)[-1].strip())
                s = str(bg.name)[-1].strip()
                # print("s", s.strip(), "done")
                num = starting_index[int(s)] # 2 [0,1,2,3,4,5,6,7]
                                            #      1   2   3   4
                # print("num ", num)
                bg_coords[num] = bg.current_coord[0]
                bg_coords[num+1] = bg.current_coord[1]
                bg.direction = bg.get_direction()
                bg_counter = bg_counter + 1
#        print("BROKE OUT")
        # save image to disk
        cv2.imwrite('pacman.jpg', fgoutput)
        # re-enable characters
        for character in self.CHARACTERS:
            character.enabled = True
            
        # re-enable blue ghosts
        for bg in self.BGS:
            bg.enabled = True

        # set variables
        self.FLOW = (0,0)
        self.SAMPLE_COUNTER += 1
        # return self.CHARACTERS, self.BGS
        return bg_coords
