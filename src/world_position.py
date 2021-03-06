from . import houghCircles
from . import util
import numpy as np
import cv2

def find_y_position(h, theta, dy2, f, is_below_half):
    factor = 1
    if is_below_half:
        factor = -1
    alpha = np.arctan(dy2/f) * factor
    y2 = (dy2*h*np.sin((np.pi/2)+alpha)) / (f*np.sin(theta)*np.sin(theta-alpha))
    return y2

def find_x_position(h, theta, f, mid_x, mid_y, x_image, y_image):
    is_below_half = int(y_image > mid_y)
    factor = 1
    if is_below_half:
        factor = -1
    dx = x_image - mid_x
    dy = abs(mid_y - y_image)
    alpha = np.arctan(dy/f) * factor
    r = h / (np.sin(theta + alpha))
    x_world = (dx * r) / (np.sqrt(dy**2 + f**2))
    return x_world

def draw_positions(img, balls, world_cochonnet):
    # create white rectangle
    h,  w = img.shape[:2]
    for i in range(0, int(h/3)):
        for j in range(int(w-w/3), w):
            img[i][j] = [100, 100, 100]
    
    cimg = img.copy()
    for i, curr_ball in enumerate(balls):
        # draw the outer circle
        if curr_ball.team_num:
            color = (0,255,0)
        else:
            color = (255, 0 ,0)
        x = int(w-w/6 + 5 * curr_ball.center_world_position[1])
        y = int(h/6 + 5 * curr_ball.center_world_position[0])
        cimg = cv2.circle(cimg, (x, y), 10, color, -1)
    
    # cochonnet
    x_co = int(w-w/6 + 5 * world_cochonnet[1])
    y_co = int(h/6 + 5 * world_cochonnet[0])
    cimg = cv2.circle(cimg, (x_co, y_co), 4, (0, 0, 255), 2)
    return cimg 

def find_balls_world_position(img, balls, cochonnet_obj, cochonnet,  verbose):
    _THETA = 63

    THETA = ((90-_THETA)/180)*np.pi
    H = 50
    F = 2200
    mid_y = img.shape[0]/2
    mid_x = img.shape[1]/2

    x_image_cochonnet = cochonnet[0]
    y_image_cochonnet = cochonnet[1]

    x_world_cochonnet = find_x_position(H, THETA, F, mid_x, mid_y, x_image_cochonnet, y_image_cochonnet)
    y_world_cochonnet = (-1 * int((y_image_cochonnet     < mid_y)) + 1 * int((y_image_cochonnet > mid_y))) * find_y_position(H, THETA, abs(mid_y-y_image_cochonnet), F, (y_image_cochonnet > mid_y) )
    world_cochonnet = [y_world_cochonnet,x_world_cochonnet]
    cochonnet_obj.center_image_pos = [x_image_cochonnet, y_image_cochonnet]
    cochonnet_obj.center_world_pos = [y_world_cochonnet, x_world_cochonnet]

    for ball in balls:
        x_image = ball.center[0]
        y_image = ball.center[1]

        x_world = find_x_position(H, THETA, F, mid_x, mid_y, x_image, y_image)
        y_world = (-1 * int((y_image     < mid_y)) + 1 * int((y_image > mid_y))) * find_y_position(H, THETA, abs(mid_y-y_image), F, (y_image > mid_y) )
        ball.center_world_position = [y_world,x_world]
        if verbose:
            print("x_image = ", x_image, "y_image = ", y_image, ", x_world = ", x_world, ", y_world = ", y_world)

    return draw_positions(img, balls, world_cochonnet)