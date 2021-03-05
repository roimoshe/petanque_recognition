from . import houghCircles
from . import util
import numpy as np
import cv2

def position():
    # # image1
    # up_y = 560
    # middle_y = 965
    # down_y = 1345
    # h = 54
    # theta = 90-58 #degrees
    # ratio_world = 2

    # image2
    up_y = 620
    middle_y = 695
    down_y = 1300
    h = 50
    theta = 90-73 #degrees
    ratio_world = 0.25

    theta = (theta/180)*np.pi # radians
    f=np.array([0.1 * i for i in range(15000,30000)])
    # f=1766.1000000000001 # image1
    # f=555.2 # image2

    # # find pixels
    img = cv2.imread("images/day1/ball_posotion/bp3_h50_theta73_x4.jpeg")
    # mask = np.ones(img.shape[:2],np.uint8)
    # x = 1000
    # mask[up_y-10:up_y+10,x:x+50] = 0
    # img_masked = cv2.bitwise_and(img,img,mask = mask)
    # save_photo('build/ball_position.jpg',img_masked, True)
    half_y = img.shape[0]/2
    up_world_y     = (-1 * int((up_y     < half_y)) + 1 * int((up_y     > half_y))) * find_y_position(h, theta, abs(half_y-up_y),     f, (up_y     > half_y) )
    middle_world_y = (-1 * int((middle_y < half_y)) + 1 * int((middle_y > half_y))) * find_y_position(h, theta, abs(half_y-middle_y), f, (middle_y > half_y) )
    down_world_y   = (-1 * int((down_y   < half_y)) + 1 * int((down_y   > half_y))) * find_y_position(h, theta, abs(half_y-down_y),   f, (down_y   > half_y) )
    ratio = (up_world_y-middle_world_y)/(middle_world_y-down_world_y)
    print(min(np.abs(ratio/ratio_world - 1)), f[np.argmin(np.abs(ratio/ratio_world - 1))])
    plt.plot(f,np.abs(ratio/ratio_world - 1), 'r')
    # show the plot
    plt.show()
    print("up ", up_world_y)
    print("middle ", middle_world_y)
    print("down ", down_world_y)
    print("ratio ", ratio)


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
    # circles_center = []
    # circles_radius = []

    for ball in balls:
        x_image = ball.center[0]
        y_image = ball.center[1]

        x_world = find_x_position(H, THETA, F, mid_x, mid_y, x_image, y_image)
        y_world = (-1 * int((y_image     < mid_y)) + 1 * int((y_image > mid_y))) * find_y_position(H, THETA, abs(mid_y-y_image), F, (y_image > mid_y) )
        ball.center_world_position = [y_world,x_world]
        if verbose:
            print("x_image = ", x_image, "y_image = ", y_image, ", x_world = ", x_world, ", y_world = ", y_world)

    #     circles_center.append([mid_x + 10*x_world, mid_y + 10*y_world])
    #     circles_radius.append(50)
    # circles_center.append([mid_x + 10*x_world_cochonnet, mid_y + 10*y_world_cochonnet])
    # circles_radius.append(10)
    # top_view_img = np.zeros(img.shape,np.uint8)
    # fig = util.plotCircles(top_view_img, circles_center, circles_radius)
    # fig.savefig("build/top_view.png", dpi=400)
    return draw_positions(img, balls, world_cochonnet)