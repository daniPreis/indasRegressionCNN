import math
import os
from numpy.linalg import lstsq

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import csv

csv_dir = 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/Codex1.csv'
new_csv_dir = 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/trained_images.csv'


def calcLineInMiddle(points):
    coefficients = np.polyfit(points[1], points[0], 1)
    return coefficients


if __name__ == "__main__":
    with open(csv_dir, newline='') as csvfile:
        original_data = list(csv.reader(csvfile))

    csvHeader = ['fileName', 'Type']
    csvData = []
    csvData.append(csvHeader)

    image_test_path = 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/train'
    list_of_images = os.listdir(image_test_path)

    for index, filename in enumerate(list_of_images):
        image = cv2.imread(image_test_path + '/' + str(filename))
        grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([grayimg], [0], None, [256], [0, 256])

        y, x, _ = plt.hist(hist)
        countVals = 0
        i = 254
        indices = np.where(grayimg == [y.max()])
        while len(indices) < 6400 and i > 0:
            i = i - 1
            # indices = np.concatenate((indices, ), axis=0)

        if len(indices[0]) > 0 and len(indices[1]) > 0:
            coordinates = zip(indices[0], indices[1])
            a = np.array(list(coordinates))
            ca = np.cov(a, y=None, rowvar=0, bias=1)
            if ca.shape != ():
                v, vect = np.linalg.eig(ca)

                tvect = np.transpose(vect)

                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111)
                ax.scatter(a[:, 0], a[:, 1])

                # use the inverse of the eigenvectors as a rotation matrix and
                # rotate the points so they align with the x and y axes
                ar = np.dot(a, np.linalg.inv(tvect))

                # get the minimum and maximum x and y
                mina = np.min(ar, axis=0)
                maxa = np.max(ar, axis=0)

                diff = (maxa - mina) * 0.5

                # the center is just half way between the min and max xy
                center = mina + diff

                # get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
                corners = np.array(
                    [center + [-diff[0], -diff[1]], center + [diff[0], -diff[1]], center + [diff[0], diff[1]],
                     center + [-diff[0], diff[1]], center + [-diff[0], -diff[1]]])
                leng = corners[1][0] - corners[0][0]
                hei = corners[3][1] - corners[0][1]
                imageColumn = []
                imageColumn.append(filename)
                imageClass = 0
                corners = np.dot(corners, tvect)
                isTopReallyTop = False

                if leng / hei > 0.7 and leng / hei < 1.3:
                    print("Große Niete ", filename)
                    print("Wirklich? ", pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[index, 1] == '3_1.jpg' or
                          pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[
                              index, 1] == '3_2.jpg', pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[
                              index, 1])
                    imageColumn.append('3')
                    imageColumn.append(pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[index, 1] == '3_1.jpg' or
                          pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[
                              index, 1] == '3_2.jpg')
                    imageClass = 3
                    if leng > hei:
                        lineInMid = calcLineInMiddle(
                            [[int((corners[0][0] + corners[3][0]) / 2), int((corners[0][1] + corners[3][1]) / 2)],
                             [int((corners[1][0] + corners[2][0]) / 2), int((corners[1][1] + corners[2][1]) / 2)]])
                    else:
                        lineInMid = calcLineInMiddle(
                            [[int((corners[0][0] + corners[1][0]) / 2), int((corners[0][1] + corners[1][1]) / 2)],
                             [int((corners[3][0] + corners[2][0]) / 2), int((corners[3][1] + corners[2][1]) / 2)]])
                    for i in enumerate(a):
                        lineY = int(lineInMid[0] * i[1][0] + lineInMid[1])
                        if i[1][1] > lineY:
                            countTop = countTop + 1
                    if countTop / len(a) > 0.69:
                        isTopReallyTop = True
                else:
                    countTop = 0
                    if leng > hei:
                        lineInMid = calcLineInMiddle(
                            [[int((corners[0][0] + corners[1][0]) / 2), int((corners[0][1] + corners[1][1]) / 2)],
                             [int((corners[3][0] + corners[2][0]) / 2), int((corners[3][1] + corners[2][1]) / 2)]])
                    else:
                        lineInMid = calcLineInMiddle(
                            [[int((corners[0][0] + corners[3][0]) / 2), int((corners[0][1] + corners[3][1]) / 2)],
                             [int((corners[1][0] + corners[2][0]) / 2), int((corners[1][1] + corners[2][1]) / 2)]])
                    for i in enumerate(a):
                        lineY = int(lineInMid[0] * i[1][0] + lineInMid[1])
                        if i[1][1] > lineY:
                            countTop = countTop + 1
                    print("Top Ratio", countTop, len(a))
                    if countTop / len(a) > 0.65 or countTop / len(a) < 0.35:
                        if countTop / len(a) > 0.7:
                            isTopReallyTop = True
                        print("Nagel ", filename)
                        print("Wirklich?",
                              pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[index, 1] == '5_1.jpg' or
                              pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[
                                  index, 1] == '5_2.jpg' or pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[
                                  index, 1] == '5_3.jpg',
                              pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[index, 1])
                        imageColumn.append('1')
                        imageColumn.append(pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[index, 1] == '5_1.jpg' or
                              pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[
                                  index, 1] == '5_2.jpg' or pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[
                                  index, 1] == '5_3.jpg')
                        imageClass = 1
                    else:
                        print("Kleine Niete ", filename)
                        print("Wirklich? ",
                              pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[index, 1] == '1_1.jpg' or
                              pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[
                                  index, 1] == '1_2.jpg', pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[index, 1])
                        if leng > hei:
                            lineInMid = calcLineInMiddle(
                                [[int((corners[0][0] + corners[3][0]) / 2), int((corners[0][1] + corners[3][1]) / 2)],
                                 [int((corners[1][0] + corners[2][0]) / 2), int((corners[1][1] + corners[2][1]) / 2)]])
                        else:
                            lineInMid = calcLineInMiddle(
                                [[int((corners[0][0] + corners[1][0]) / 2), int((corners[0][1] + corners[1][1]) / 2)],
                                 [int((corners[3][0] + corners[2][0]) / 2), int((corners[3][1] + corners[2][1]) / 2)]])
                        for i in enumerate(a):
                            lineY = int(lineInMid[0] * i[1][0] + lineInMid[1])
                            if i[1][1] > lineY:
                                countTop = countTop + 1
                        if countTop / len(a) > 0.69:
                            isTopReallyTop = True
                        imageColumn.append('2')
                        imageColumn.append(pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[index, 1] == '1_1.jpg' or
                              pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[
                                  index, 1] == '1_2.jpg')
                        imageClass = 2

                # use the the eigenvectors as a rotation matrix and
                # rotate the corners and the center back
                # corners = np.dot(corners, tvect)
                center = np.dot(center, tvect)
                csvData.append(imageColumn)
                ax.scatter([center[0]], [center[1]])
                ax.plot(corners[:, 0], corners[:, 1], '-')
                plt.axis('equal')
                # plt.show()
                if imageClass == 1:
                    if isTopReallyTop:
                        angleInDegrees = math.degrees(
                            math.atan2(corners[1][1] - corners[2][1], corners[1][0] - corners[2][0]))
                    else:
                        angleInDegrees = math.degrees(
                            math.atan2(corners[0][1] - corners[3][1], corners[0][0] - corners[3][0]))
                else:
                    if isTopReallyTop:
                        angleInDegrees = math.degrees(
                            math.atan2(corners[2][1] - corners[3][1], corners[2][0] - corners[3][0]))
                    else:
                        angleInDegrees = math.degrees(
                            math.atan2(corners[1][1] - corners[0][1], corners[1][0] - corners[0][0]))

                rot_mat = cv2.getRotationMatrix2D(center, 360 - angleInDegrees, 1.0)

                imageColumn.append(int(angleInDegrees))
                imageColumn.append(pd.read_csv(csv_dir).sort_values(by=['ID']).iloc[index, 2])

                result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
                plt.imshow(result)
                plt.show()
                csvData.append(imageColumn)
                print("-----------------------------------------------------------------------------------------------------------------------")
    f = open(new_csv_dir, 'w')
    with f:
        writer = csv.writer(f)
        writer.writerows(csvData)
