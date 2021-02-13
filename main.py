import cv2
import sys
from src import *
import argparse


def step1(img):
    return kmeans.kmeans(img, cv2.COLOR_BGR2LAB)

def step2(img):
    return lines.houghLines(img)

def step3(img):
    return util.hough(img)

steps = [step1, step2, step3]

def main(arguments):
    parser = argparse.ArgumentParser(description="Petanque recognition")
    parser.add_argument("-s","--step", type=int, help="Input step number to start from", default=1)
    parser.add_argument("-i","--image_num", type=int, help="Input image number", default=1)
    args = parser.parse_args(arguments)

    if args.image_num < 0 or args.step > 17:
        print("image_num: ", args.image_num," not supported")
        return 1
    if args.step < 1 or args.step > len(steps):
        print("step: ", args.step," not supported")
        return 2
    elif args.step  == 1:
        img_path = 'images/image{}.jpg'.format(args.image_num)
        img = cv2.imread(img_path)
        util.save_photo('build/image{}.jpg'.format(args.image_num),img, True)
    else:
        img = cv2.imread('build/image{}_step{}.jpg'.format(args.image_num, args.step - 1))
    print("image_num: ", args.image_num,", start step: ", args.step)

    for i in range(args.step - 1, len(steps)):
        img = steps[i](img)
        util.save_photo('build/image{}_step{}.jpg'.format(args.image_num, i+1),img, True)
    return 0

if __name__ == '__main__':
    result = main(sys.argv[1:])
    sys.exit(result)
