# commands:
# photo: python3 main.py -i images/day2/photos/setup2/image16.jpeg -c -F setup2 -v
# video: python3 main.py -i 4 -c -F video -f 4
import cv2
import sys
from src import *
import argparse
import shutil
import os
import errno
import numpy as np

# general params
BLUR_SIZE = 500

N_CLUSTERS = 3
EDGE_DETECTOR_BLUR_SIZE = 5
# format spesific params
HOUGH_THRESHOLD_VIDEO = 0.338
HOUGH_THRESHOLD_PHOTO = 0.33
HOUGH_THRESHOLD_SETUP2 = 0.18 # 0.19 wont work on image25/setup2
HOUGH_RADIUS_RANGE_VIDEO = [12,18]
HOUGH_RADIUS_RANGE_PHOTO = [28,40]
HOUGH_RADIUS_RANGE_SETUP2 = [42,70]
MEDIAN_BLUR_SIZE_DAY1 = 11
MEDIAN_BLUR_SIZE_DAY2 = 17
BURNING_SIZE_DAY1 = 100
BURNING_SIZE_DAY2 = 20

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class Step:
  def __init__(self, function, name):
    self.function = function
    self.name = name
class Parameters:
  def __init__(self, hough_threshold, hough_radius_range, burning_size, blur_size, n_clusters, edge_detector_blur_size, median_blur_size):
    self.hough_threshold         = hough_threshold
    self.hough_radius_range      = hough_radius_range
    self.burning_size            = burning_size
    self.blur_size               = blur_size
    self.n_clusters              = n_clusters
    self.edge_detector_blur_size = edge_detector_blur_size
    self.median_blur_size = median_blur_size

def bilateral_and_blur_step(img, verbose):
    img = cv2.bilateralFilter(np.float32(img),15,800,800)
    kernel = np.ones((5,5),np.float32)/25
    return cv2.filter2D(img,-1,kernel)
    return img

def blur_step(img, verbose, params):
    BLUR_WEIGHT = params.blur_size
    ksize = (BLUR_WEIGHT, BLUR_WEIGHT)
    return cv2.blur(img, ksize, cv2.BORDER_DEFAULT)

def kmeans_step(img, verbose, params):
    return kmeans.kmeans(img, cv2.COLOR_BGR2LAB, params.n_clusters, "max", original_image, verbose)

def remove_background_step(img, verbose, params):
    return kmeans.remove_background("max", original_image, verbose)

def edge_detector_step(img, verbose, params):
    global kmeans_img
    kmeans_img = img
    return util.edgeDetector(img, params.edge_detector_blur_size, params.median_blur_size, verbose)

def burn_blob_frame_step(img, verbose, params):
    return util.burn_blob_frame_step(img, params.burning_size, verbose)

def hough_circles_step(img, verbose, params):
    return houghCircles.hough_circles(original_image, kmeans_img, img, params.hough_threshold, params.hough_radius_range, verbose, balls, cochonnet, image_num)
# TODO: can move team detection out of last step
def world_position_step(img, verbose, params):
    return world_position.find_balls_world_position(img, balls, cochonnet_obj, cochonnet, verbose)

def add_legend_step(img, verbose, params):
    return legend.add_legend(img, balls, cochonnet_obj, verbose)

def hough_lines_step(img, verbose, params):
    return lines.houghLines(img, original_image, verbose)


def train():
    util.pca()
    print("empty train")

main_plan = [Step(blur_step, "blur_step"),
             Step(kmeans_step, "kmeans_step"),
             Step(remove_background_step, "remove_background_step"),
             Step(burn_blob_frame_step, "burn_blob_frame_step"),
             Step(edge_detector_step, "edge_detector_step"),
             Step(hough_circles_step, "hough_circles_step"),
             Step(world_position_step, "world_position_step"),
             Step(add_legend_step, "add_legend_step")]
main_plan_with_burn = [ Step(blur_step, "blur_step"),
              Step(kmeans_step, "kmeans_step"),
              Step(burn_blob_frame_step, "burn_blob_frame_step"),
              Step(edge_detector_step, "edge_detector_step"),
              Step(hough_circles_step, "hough_circles_step"),
              Step(world_position_step, "world_position_step") ]
plans     = [ main_plan, main_plan_with_burn ] 

def main(arguments):
    try:
        os.makedirs('build')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    global original_image
    global balls
    global cochonnet
    global cochonnet_obj
    global image_num
    parser = argparse.ArgumentParser(description="Petanque recognition")
    parser.add_argument("-s","--step", type=int, help="Input step number to start from", default=1)
    parser.add_argument("-e","--end_step", type=int, help="Input step number to end in", default=len(main_plan))
    parser.add_argument("-i","--image_path", type=str, help="Input image number")
    parser.add_argument("-r","--run_num", type=int, help="Input run number", default=1)
    parser.add_argument("-p","--plan_num", type=int, help="Input plan number", default=0)
    parser.add_argument("-f","--frame", type=int, help="Input frame number", default=-1)
    parser.add_argument("-q","--quick", action='store_true', help="quick run, without saving any step")
    parser.add_argument("-v","--verbose", action='store_true', help="verbosity level")
    parser.add_argument("-t","--train", action='store_true', help="train mode")
    parser.add_argument("-c","--clean", action='store_true', help="clean build")
    parser.add_argument("-u","--user", action='store_true', help="user execution")
    parser.add_argument("-N","--no_previous_step", action='store_true', help="run from step 'step' with the original photo")
    parser.add_argument("-F","--image_format", type=str, help="image format - video/photo/day2", default="photo")
    args = parser.parse_args(arguments)
    
    photo_params = Parameters(HOUGH_THRESHOLD_PHOTO, HOUGH_RADIUS_RANGE_PHOTO, BURNING_SIZE_DAY1, BLUR_SIZE, N_CLUSTERS, EDGE_DETECTOR_BLUR_SIZE, MEDIAN_BLUR_SIZE_DAY1)
    video_params = Parameters(HOUGH_THRESHOLD_VIDEO, HOUGH_RADIUS_RANGE_VIDEO, BURNING_SIZE_DAY1, BLUR_SIZE, N_CLUSTERS, EDGE_DETECTOR_BLUR_SIZE, MEDIAN_BLUR_SIZE_DAY1)
    setup2_params  = Parameters(HOUGH_THRESHOLD_SETUP2, HOUGH_RADIUS_RANGE_SETUP2, BURNING_SIZE_DAY2, BLUR_SIZE, N_CLUSTERS, EDGE_DETECTOR_BLUR_SIZE, MEDIAN_BLUR_SIZE_DAY2) # image18 works good

    if args.user:
        image_num = 1
        args.run_num = image_num
        args.image_path = "images/final/image{}.jpeg".format(image_num)
        args.clean = True
        args.image_format = "setup2"
        args.plan_num = 0

        
    img_path = args.image_path

    if args.image_format == "photo":
        params = photo_params
    elif args.image_format == "video":
        params = video_params
    elif args.image_format == "day2":
        params = day2_params
    elif args.image_format == "setup2":
        params = setup2_params
    else:
        print("image_format: ", args.image_format," not supported")
    if args.clean:
        for root, dirs, files in os.walk('./build/'):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        print("remove ./build content")
    
    if args.train:
        return train()
    if args.plan_num >= len(plans):
        print("plan_num: ", args.plan_num," not supported")
        return 1
    if args.step < 1 or args.step > len(main_plan):
        print("step: ", args.step," not supported")
        return 3
    
    if args.user:
        print(color.BOLD + "start process image {}".format(image_num) + color.END)

    while True:
        cochonnet_obj = houghCircles.Cochonnet(None, None)
        balls = []
        cochonnet = []

        img = cv2.imread(img_path)
        original_image = img.copy()
        if not args.quick:
            util.save_photo('build/{}_image{}_{}_{}.jpg'.format(args.image_format,args.run_num, 0, "original"),img, True)


        for i in range(args.step - 1, min(args.end_step, len(plans[args.plan_num]))):
            next_step = plans[args.plan_num][i]
            print("execute {}..".format(next_step.name))
            img = next_step.function(img, args.verbose, params)
            if not args.quick:
                util.save_photo('build/{}_image{}_{}_{}.jpg'.format(args.image_format,args.run_num, i+1, next_step.name),img, True)
        if args.quick:
            util.save_photo('build/image{}_step{}.jpg'.format(args.run_num, i+1),img, True)
        image_num+=1
        if (not args.user) or (image_num > 6):
            break
        img_path = "images/final/image{}.jpeg".format(image_num)
        args.run_num = image_num
        print(color.BOLD + "start process image {}".format(image_num) + color.END)
    
    return 0

if __name__ == '__main__':
    result = main(sys.argv[1:])
    sys.exit(result)
