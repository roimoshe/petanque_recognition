import cv2
import sys
from src import *

def main():
    for i in range(5,18):
        img_path = 'images/image{}.jpg'.format(i)
        img = cv2.imread(img_path)
        util.save_photo('build/image{}_step1.jpg'.format(i),img, True)
        img=kmeans.kmeans(img, cv2.COLOR_BGR2LAB)
        util.save_photo('build/image{}_step2.jpg'.format(i),img, True)
        img=lines.houghLines(img)
        util.save_photo('build/image{}_step3.jpg'.format(i),img, True)
        fig = util.hough(img)
        fig.savefig("build/image{}_step4.png".format(i), dpi=400)
    return 0

if __name__ == '__main__':
    result = main()
    sys.exit(result)
