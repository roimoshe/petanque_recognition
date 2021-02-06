import cv2
import sys
from src import *

def main():
    img_path = 'images/image5.jpg'
    img = cv2.imread(img_path)
    cv2.imwrite('tmp/step1.jpg',img)
    img=kmeans.kmeans(img, cv2.COLOR_BGR2LAB)
    cv2.imwrite('tmp/step2.jpg',img)
    img=lines.houghLines(img)
    cv2.imwrite('tmp/step3.jpg',img)
    util.hough(img)

    return 0

if __name__ == '__main__':
    result = main()
    sys.exit(result)
