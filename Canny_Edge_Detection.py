import cv2
import math
import multiprocessing

image = cv2.imread('image.jpg')

def GrayScale(img):
    img = img.copy()
    for x, row in enumerate(img):
        for y, pixel in enumerate(row):
            intensity = int(sum(img[x][y])/3)
            for z, subpixel in enumerate(pixel):
                img[x][y][z] = intensity
    return img

def meanBlur(img, size):
    blurImg = img.copy()
    size2 = (size*2)-1
    for x, row in enumerate(img):
        for y, pixel in enumerate(row):
            for z, subpixel in enumerate(pixel):
                avrageVal = 0
                cals = 0
                for i in range(0, size2**2):
                    try:
                        avrageVal += int(img[x+i%size2-size+1][y+math.floor(i/size2)-size+1][z])
                        cals += 1
                    except IndexError:
                        pass
                    continue
                avrageVal = avrageVal / cals
                blurImg[x][y][z] = avrageVal
    return blurImg

def gaussianBlurPascal(img, size):
    blurImg = img.copy()
    size2 = (size*2)-1
    stddiv = pascal(size2)

    for x, row in enumerate(img):
        for y, pixel in enumerate(row):
            for z, subpixel in enumerate(pixel):
                avrageVal = 0
                cals = 0
                for i in range(0, size2**2):
                    xi = i%size2
                    yi = math.floor(i/size2)
                    weight = stddiv[xi] * stddiv[yi]
                    try:
                        avrageVal += int(img[x+xi-size+1][y+yi-size+1][z])*weight
                        cals += weight
                    except IndexError:
                        pass
                    continue
                avrageVal = avrageVal / cals
                blurImg[x][y][z] = avrageVal
    return blurImg

def edgeDetectionVector():
    array = [0]
    return array

def edgeDetectionColor():
    array = [0]
    return array

def edgeDetectionGrey():
    array = [0]
    return array


def pascal(n):
  Row = [1]
  for i in range(n):
    Row.append(Row[i] * (n-i) / (i+1))
  return Row





if __name__ == '__main__':
    #p = Process(target=f, args=('bob',))
    #p.start()
    #p.join()
    image = GrayScale(image)

    #cv2.imshow('image0', image)
    #cv2.imshow('image1', meanBlur(image, 3))
    cv2.imshow('image2', gaussianBlurPascal(image, 3))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


