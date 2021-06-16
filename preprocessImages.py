import random

import cv2
import numpy as np
import os
import csv

root_dir = 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/train/'
img_dir = 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/images/'
csv_dir = 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/Codex1.csv'
new_csv_dir = 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/Codex_train_new.csv'
images = os.listdir(root_dir)
sumNewImages = 2


def add_noise(img, row, col):
    # Getting the dimensions of the image

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


with open(csv_dir, newline='') as csvfile:
    original_data = list(csv.reader(csvfile))

csvHeader = ['ID', 'Origin', 'Angle']
csvData = []
csvData.append(csvHeader)

imageId = 0
for index, image in enumerate(images):
    img = cv2.imread(root_dir + image)
    originalImgId = image.split('.')[0]
    originImg = ''
    originalAngle = 0
    for row in original_data:
        # if current rows 2nd value is equal to input, print that row
        if originalImgId == row[0]:
            originImg = row[1]
            originalAngle = row[2]
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    for i in range(0, sumNewImages):
        imageColumn = []
        randomAngle = random.randrange(0, 360)
        imageColumn.append(imageId)
        imageColumn.append(originImg)
        imageColumn.append(randomAngle)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, randomAngle, scale)
        rotatedImg = cv2.warpAffine(img, M, (w, h))
        randomFilter = random.randrange(1, 5)

        # Case 5: Apply no filter
        finalImg = rotatedImg
        if randomFilter == 1:
            # Case 1: Blur Image with 5x5 kernel Boxfilter
            print("Boxblur filter")
            finalImg = cv2.blur(rotatedImg, (5, 5))
        elif randomFilter == 2:
            print("Sharpen Kernel")
            # Case 2: Sharpen Kernel
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            finalImg = cv2.filter2D(rotatedImg, -1, kernel)
        elif randomFilter == 3:
            print("Median Blur")
            # Case 3: Median Blur
            finalImg = cv2.medianBlur(rotatedImg, 5)
        elif randomFilter == 4:
            print("Salt and Pepper Noise")
            # Case 4: Salt and Pepper noise
            finalImg = add_noise(rotatedImg, h, w)
        csvData.append(imageColumn)
        cv2.imwrite(img_dir + str(imageId) + ".jpg", finalImg)
        imageId = imageId + 1
    # Save original image
    imageColumn = []
    imageColumn.append(imageId)
    imageColumn.append(originImg)
    imageColumn.append(originalAngle)
    csvData.append(imageColumn)
    cv2.imwrite(img_dir + str(imageId) + ".jpg", img)
    imageId = imageId + 1

f = open(new_csv_dir, 'w')
with f:
    writer = csv.writer(f)
    writer.writerows(csvData)
