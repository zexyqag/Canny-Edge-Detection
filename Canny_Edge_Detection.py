import cv2
import multiprocessing
import numpy as np
from enum import Enum

class KernelType(Enum):
    pascal = 1
    mean = 2
    sobely = 3
    sobelx = 4
    prewitty = 5
    prewittx = 6
    emboss = 7
    outline = 8
    sobely45 = 9
    sobelx45 = 10
    sharpen = 11
    sobel5x5y = 12
    sobel5x5x = 13

def GrayScale(img):
    img = img.copy()
    for x, row in enumerate(img):
        for y, pixel in enumerate(row):
            intensity = int(sum(img[x][y]) / 3)
            for z, subpixel in enumerate(pixel):
                img[x][y][z] = intensity
    return img

def Convolution(image, kernel, normalization):
    modImg = np.array(image.copy(), dtype = int)
    kernelRadius = len(kernel) // 2

    for j, row in enumerate(image):
        for k, pixel in enumerate(row):
            for l, subpixel in enumerate(pixel):
                accumulator = 0
                weightSum = 0
                for m, kernelRow in enumerate(kernel):
                    for n, element in enumerate(kernelRow):
                        mi = j + m - kernelRadius
                        ni = k + n - kernelRadius
                        if (mi >= 0 <= ni and mi < len(image) and ni < len(image[j]) and element != 0):
                            accumulator += int(image[mi][ni][l]) * element
                            weightSum += abs(element)
                modImg[j][k][l] = (accumulator / weightSum if normalization else accumulator)
    return (np.array(modImg.copy(), dtype = np.uint8) if normalization else modImg)

def KernelGenerator(*args):

    #arguments handling
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
        
        #Kernel handling
    if (kernelType == KernelType.pascal):
        size -= 1
        row = [1]
        for i in range(size):
            row.append(row[i] * (size - i) / (i + 1))
        return MetrixFromRow(row)
    elif (kernelType == KernelType.mean):
        return MetrixFromRow([1 for x in range(size)])
    elif (kernelType == KernelType.sobely):
        return [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
    elif (kernelType == KernelType.sobelx):
        return [[-1,-2,-1],
                [ 0, 0, 0],
                [ 1, 2, 1]]
    elif (kernelType == KernelType.prewitty):
        return [[-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]]
    elif (kernelType == KernelType.prewittx):
        return [[-1,-1,-1],
                [ 0, 0, 0],
                [ 1, 1, 1]]
    elif (kernelType == KernelType.emboss):
        return [[-2,-1, 0],
                [-1, 1, 1],
                [ 0, 1, 2]]
    elif (kernelType == KernelType.outline):
        return [[-1,-1,-1],
                [-1, 8,-1],
                [-1,-1,-1]]
    elif (kernelType == KernelType.sobelx45):
        return [[ 0, 1, 2],
                [-1, 0, 1],
                [-2,-1, 0]]
    elif (kernelType == KernelType.sobely45):
        return [[-2,-1, 0],
                [-1, 0, 1],
                [ 0, 1, 2]]
    elif (kernelType == KernelType.sharpen):
        return [[0, -1, 0],
                [-1, 5, -1],
                [ 0, -1, 0]]
    elif (kernelType == KernelType.sobel5x5y):
        return [[ 2, 2, 4, 2, 2],
                [ 1, 1, 2, 1, 1],
                [ 0, 0, 0, 0, 0],
                [-1,-1,-2,-1,-1],
                [-2,-2,-4,-2,-2]]
    elif (kernelType == KernelType.sobel5x5x):
        return [[ 2, 1, 0,-1,-2],
                [ 2, 1, 0,-1,-2],
                [ 4, 2, 0,-2,-4],
                [ 2, 1, 0,-1,-2],
                [ 2, 1, 0,-1,-2]]
        
def MetrixFromRow(row):
    metrix = [[0 for x in range(len(row))] for y in range(len(row))]
    for i, metrixRow in enumerate(metrix):
        for j, element in enumerate(metrixRow):
            metrix[i][j] = row[j] * row[i]
    return metrix

def GradientMagnitude(imagex, imagey):
    modImgx = np.array(imagex.copy(), dtype = np.int)
    modImgy = np.array(imagey.copy(), dtype = np.int)   
    return np.sqrt(modImgx ** 2 + modImgy ** 2)

def ImageNormalization(image):
    modImg = image.copy()
    modImg = modImg + np.absolute(np.amin(modImg))
    modImg = modImg * (255 / np.amax(modImg))
    return np.array(modImg, dtype = np.uint8)

def ImageThresholding(image, threshold):
    modImg = image.copy()
    modImg[modImg >= threshold] = 255
    modImg[modImg < threshold] = 0
    return modImg


if __name__ == '__main__':
    #Get image
    image = cv2.imread('image2.jpg')
    cv2.imshow('image', image)

    #Gray scale image
    imageGray = GrayScale(image)
    cv2.imshow('imageGray', imageGray)

    #Blur image
    imageBlured = Convolution(imageGray, KernelGenerator(KernelType.pascal, 3), True)
    cv2.imshow('imageBlured', imageBlured)

    #Detect edges image
    imageSobelx = Convolution(imageBlured, KernelGenerator(KernelType.sobel5x5x), False)
    cv2.imshow('imageSobelx', ImageNormalization(imageSobelx))
    imageSobely = Convolution(imageBlured, KernelGenerator(KernelType.sobel5x5y), False)
    cv2.imshow('imageSobely', ImageNormalization(imageSobely))

    #Get Magnetude
    imageMagnetude = ImageNormalization(GradientMagnitude(imageSobelx, imageSobely))
    cv2.imshow('imageMagnetude', imageMagnetude)

    #Threshold image
    imageThreshold = ImageThresholding(imageMagnetude, 33)
    cv2.imshow('imageThreshold', imageThreshold)



    #Image Outline
    imageOutline = Convolution(imageBlured, KernelGenerator(KernelType.outline), False)
    cv2.imshow('imageOutline', ImageNormalization(imageOutline))

    #Image Emboss
    imageEmboss = ImageNormalization(Convolution(imageBlured, KernelGenerator(KernelType.emboss), False))
    cv2.imshow('imageEmboss', imageEmboss)

    #Image sharpen
    imageSharpen = Convolution(image, KernelGenerator(KernelType.sharpen), True)
    cv2.imshow('imageEmboss', imageEmboss)


    #Outline Magnetude
    imageOutlineMagnetude = ImageNormalization(GradientMagnitude(imageOutline, imageOutline))
    cv2.imshow('imageOutlineMagnetude', imageOutlineMagnetude)

    cv2.waitKey(0)
    cv2.destroyAllWindows()