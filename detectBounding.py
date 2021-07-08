import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import csv

csv_dir = 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/Codex1.csv'
new_csv_dir = 'D:/Dokumente/Uni/Master/Indas/BMW - Material-20210512/trained_images.csv'

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
        print(y.max())
        indices = np.where(grayimg == [ y.max()])
        if len(indices[0]) > 0 and len(indices[1]) > 0:
            coordinates = zip(indices[0], indices[1])
            a = np.array(list(coordinates))
            ca = np.cov(a, y=None, rowvar=0, bias=1)
            print('SHAPE', ca.shape)
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
                if leng / hei < 0.75:
                    print("Nagel ")
                    print("Wirklich? ", pd.read_csv(csv_dir).iloc[index, 1] == '5_1.jpg' or pd.read_csv(csv_dir).iloc[
                        index, 1] == '5_2.jpg' or pd.read_csv(csv_dir).iloc[index, 1] == '5_3.jpg',
                          pd.read_csv(csv_dir).iloc[index, 1])
                    imageColumn.append('1')
                elif leng / hei > 1.25:
                    print("Kleine Niete ")
                    print("Wirklich? ", pd.read_csv(csv_dir).iloc[index, 1] == '1_1.jpg' or pd.read_csv(csv_dir).iloc[
                        index, 1] == '1_2.jpg', pd.read_csv(csv_dir).iloc[index, 1])
                    imageColumn.append('2')
                else:
                    print("Große Niete ")
                    print("Wirklich? ", pd.read_csv(csv_dir).iloc[index, 1] == '3_1.jpg' or pd.read_csv(csv_dir).iloc[
                        index, 1] == '3_2.jpg', pd.read_csv(csv_dir).iloc[
                              index, 1])
                    imageColumn.append('3')
                print("PRINT 1", leng / hei)
                # use the the eigenvectors as a rotation matrix and
                # rotate the corners and the centerback
                corners = np.dot(corners, tvect)
                center = np.dot(center, tvect)
                csvData.append(imageColumn)
                # ax.scatter([center[0]], [center[1]])
                # ax.plot(corners[:, 0], corners[:, 1], '-')

                # plt.axis('equal')
                # plt.show()
    f = open(new_csv_dir, 'w')
    with f:
        writer = csv.writer(f)
        writer.writerows(csvData)