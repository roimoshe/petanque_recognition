from . import houghCircles
from . import util
import numpy as np
import cv2

def add_winner_title(img, title):
    height, width, depth = img.shape

    BLACK = (255,255,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1.3
    font_color = BLACK
    font_thickness = 2
    x,y = 10,int(w-w/3)+20
    img_text = cv2.putText(img, title, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

    return img_text

def paste_legend(img, leads, team1_remain_balls, team2_remain_balls):
    l_img = img.copy()
    s_img = cv2.imread("images/legend/t{}t{}l{}.png".format(team1_remain_balls, team2_remain_balls, leads))
    h, w = s_img.shape[:2]
    s_img = cv2.resize(s_img, (int(w/2), int(h/2)))
    y_offset=0
    w_l = l_img.shape[1]
    h_l = l_img.shape[0]
    x_offset=int(w_l-w_l/3)
    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
    for i in range(h_l):
        for j in range(w_l):
            if sum(l_img[i][j]) == 0:
               l_img[i][j] = img[i][j]
    if team1_remain_balls == 0 and team2_remain_balls == 0:
        l_img = add_winner_title(l_img, "Winner Is Team {} !!".format(leads))
    return l_img

def add_legend(img, balls, cochonnet_obj, verbose):
    team1_remain_balls = 3
    team2_remain_balls = 3
    for curr_ball in balls:
        if curr_ball.team_num == 0:
            team1_remain_balls-=1
        else:
            team2_remain_balls-=1
        curr_ball.distance = np.linalg.norm(np.array(curr_ball.center_world_position) - np.array(cochonnet_obj.center_world_pos))
    balls.sort(key=lambda x: x.distance)
    leads = 1 + balls[0].team_num
    return paste_legend(img, leads, team1_remain_balls, team2_remain_balls)