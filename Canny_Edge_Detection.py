import cv2
import math
import multiprocessing
from enum import Enum

class EdgeHandling(Enum):
    extend = 1
    wrap = 2
    mirror = 3
    crop = 4
    kernelCrop = 4

class KernelType(Enum):
    pascal = 1
    mean = 2
    sobely = 3
    sobelx = 4

def GrayScale(img):
    img = img.copy()
    for x, row in enumerate(img):
        for y, pixel in enumerate(row):
            intensity = int(sum(img[x][y]) / 3)
            for z, subpixel in enumerate(pixel):
                img[x][y][z] = intensity
    return img

def Convolution(image, kernel, edgeHandling):
    modImg = image.copy()
    kernelRadius = int(math.floor(len(kernel) / 2))

    for j, row in enumerate(image):
        for k, pixel in enumerate(row):
            for l, subpixel in enumerate(pixel):
                accumulator = 0
                cals = 0
                for m, kernelRow in enumerate(kernel):
                    for n, element in enumerate(kernelRow):
                        mi = j + m - kernelRadius
                        ni = k + n - kernelRadius
                        if (mi >= 0 <= ni and mi < len(image) and ni < len(image[j]) and element != 0):
                            accumulator += int(image[mi][ni][l]) * element
                            cals += abs(element)
                modImg[j][k][l] = accumulator / cals
    return modImg

def KernelGenerator(*args):
    if (len(args) == 0):
        raise Exception('Missing first argument of type KernelType')
    elif (len(args) >= 3):
        raise Exception('Too many arugments')
    elif (len(args) == 1):
        kernelType = args[0]
        size = 3
        if (type(kernelType) != KernelType):
            raise Exception('First arguemnt needs to be of type KernelType')
    elif (len(args) == 2):
        kernelType = args[0]
        size = args[1]
        if (type(size) != int):
            raise Exception('Second argument needs to be of type int')
        elif (type(kernelType) != KernelType):
            raise Exception('First arguemnt needs to be of type KernelType')
        
    if (kernelType == KernelType.pascal):
        size -= 1
        row = [1]
        for i in range(size):
            row.append(row[i] * (size - i) / (i + 1))
        return MetrixFromRow(row)
    elif (kernelType == KernelType.mean):
        return MetrixFromRow([1 for x in range(size)])
    elif (kernelType == KernelType.sobelx):
        return [[1,0,-1],
                [2,0,-2],
                [1,0,-1]]
    elif (kernelType == KernelType.sobely):
        return [[1, 2, 1],
                [0, 0, 0],
                [-1,-2,-1]]
        

def MetrixFromRow(row):
    metrix = [[0 for x in range(len(row))] for y in range(len(row))]
    for i, metrixRow in enumerate(metrix):
        for j, element in enumerate(metrixRow):
            metrix[i][j] = row[j] * row[i]
    return metrix


if __name__ == '__main__':
    image = cv2.imread('image2.jpg')
    cv2.imshow('image0', image)
    image = GrayScale(image)
    cv2.imshow('image1', image)
    image = Convolution(image, KernelGenerator(KernelType.pascal, 11), EdgeHandling.kernelCrop)
    cv2.imshow('image2', image)
    image = Convolution(image, KernelGenerator(KernelType.sobelx), EdgeHandling.kernelCrop)
    cv2.imshow('image3', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()