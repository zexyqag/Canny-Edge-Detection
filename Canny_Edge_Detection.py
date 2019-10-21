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
                            cals += element
                modImg[j][k][l] = accumulator / cals
    return modImg


def KernelGenerator(size, kernelType):
    if (kernelType == KernelType.pascal):
        size -= 1
        row = [1]
        for i in range(size):
            row.append(row[i] * (size - i) / (i + 1))
        return MetrixFromRow(row)
    elif (kernelType == KernelType.mean):
        return MetrixFromRow([1 for x in range(size)])
        

def MetrixFromRow(row):
    metrix = [[0 for x in range(len(row))] for y in range(len(row))]
    for i, metrixRow in enumerate(metrix):
        for j, element in enumerate(metrixRow):
            metrix[i][j] = row[j] * row[i]
    return metrix


if __name__ == '__main__':
    image = cv2.imread('image.jpg')
    cv2.imshow('image0', image)
    cv2.imshow('image1', Convolution(image, KernelGenerator(7,KernelType.pascal), EdgeHandling.kernelCrop))
    cv2.imshow('image2', Convolution(image, KernelGenerator(7,KernelType.mean), EdgeHandling.kernelCrop))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
