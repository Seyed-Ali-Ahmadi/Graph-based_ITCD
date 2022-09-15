import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import os
import csv


def get_min_idx(array):
    minimum = 10
    r = None
    c = None
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            if array[row, col] < minimum:
                minimum = array[row, col]
                r = row
                c = col
    return minimum, r, c


gt = np.loadtxt(r"\3d_ground_truth_data.txt")  # GT3D
tree_gt = KDTree(gt[:, 1:])  # GT3D

pathname = './location_of_output_files/'
dirs = os.listdir(pathname)
dirs.remove('CHM.tif')

for folder in dirs:
    path = pathname + folder + '/'
    points = np.loadtxt(path + 'centroids.csv', delimiter=',')
    tree_points = KDTree(points[:, 2:4])
    sdm = tree_gt.sparse_distance_matrix(tree_points, 3)  # pairwise distances less than 3m
    SDM = np.zeros((gt.shape[0], points.shape[0]))

    for i in range(SDM.shape[0]):
        for j in range(SDM.shape[1]):
            SDM[i, j] = sdm[i, j]
            if SDM[i, j] == 0.0:
                SDM[i, j] = 10

    Min = 0
    SDM_mins = np.zeros_like(SDM)
    Iter = 0
    while (Min != 10) or (Min < 10):
        Min, r, c = get_min_idx(SDM)
        # print(Iter, Min, r, c)
        if (r is not None) and (c is not None):
            SDM_mins[r, c] = Min
            SDM[r, :] = 11
            SDM[:, c] = 11
            Iter += 1

    # finalmat = np.zeros((gt.shape[0], 6))  # GT
    finalmat = np.zeros((gt.shape[0], 8))  # GT3D
    plt.figure(figsize=(8, 8)), plt.tight_layout()
    plt.xticks([]), plt.yticks([])
    # plt.plot(gt[:, 0], gt[:, 1], 'ok')
    plt.plot(gt[:, 1], gt[:, 2], 'ok')  # GT3D
    plt.title('kNN map')
    for i in range(gt.shape[0]):
        # finalmat[i, 0] = gt[i, 0]  # Put X-GT
        # finalmat[i, 1] = gt[i, 1]  # Put Y-GT
        finalmat[i, 0] = gt[i, 1]  # Put X-GT3D
        finalmat[i, 1] = gt[i, 2]  # Put Y-GT3D
        finalmat[i, 7] = gt[i, 0]  # Put Y-GT3D
        maximum = np.max(SDM_mins[i, :])
        j = np.argmax(SDM_mins[i, :])
        if maximum == 0:
            finalmat[i, 2:] = -999.0  # Those GTs without assignments
            # plt.plot(gt[i, 0], gt[i, 1], 'or')
            plt.plot(gt[i, 1], gt[i, 2], 'or')  # GT3D
        else:
            finalmat[i, 2] = points[j, 2]  # Put x-Centers
            finalmat[i, 3] = points[j, 3]  # Put y-Centers
            finalmat[i, 4] = j  # Put index
            finalmat[i, 5] = maximum  # Put distance to nearest GT
            finalmat[i, 6] = points[j, 4]   # Put z-Centers  # GT3D
            plt.plot(points[j, 2], points[j, 3], '*b')
    plt.savefig(path + 'kNN Map.png', dpi=300)
    plt.show(block=False), plt.pause(3), plt.close()

    TP = 0  # True Positive detections (trees found as trees)
    FN = 0  # False Negative detections (trees found as no-trees)
    FP = 0  # False Positive detections (no-trees found as trees)
    rmse = 0    # RMSE of horizontal differences
    rmse_z = 0  # RMSE of height differences  # GT3D

    # plt.figure()
    # plt.xlabel('True positive detections')
    # plt.ylabel('Distance to GT')
    for i in range(finalmat.shape[0]):
        if finalmat[i, 4] != -999.0:
            TP += 1
            rmse += (finalmat[i, 0] - finalmat[i, 2]) ** 2 + (finalmat[i, 1] - finalmat[i, 3]) ** 2
            rmse_z += (finalmat[i, 6] - finalmat[i, 7]) ** 2  # GT3D
            # plt.plot(i, finalmat[i, 5], 'r.')
            # plt.axvline(x=i)
        else:
            FN += 1

    rmse = np.sqrt(rmse / TP)
    rmse_z = np.sqrt(rmse_z / TP)
    FP = points.shape[0] - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1score = 2 * (precision * recall) / (precision + recall)
    print(folder, TP, FN, FP, round(precision, 2), round(recall, 2),
          round(f1score, 2), round(rmse, 2), round(rmse_z, 2))  # GT3D
    # plt.show()


# ttops4_(10)_249 123 82 126 0.49 0.6 0.54 1.56 2.49   # 3m
# 15_0.8 130 75 117 0.53 0.63 0.58 1.51 1.5      ----> 3m

# ttops4_(10)_249 150 55 99 0.6 0.73 0.66 2.19 3.38    # 5m
# 15_0.8 152 53 95 0.62 0.74 0.67 2.04 2.31      ----> 5m

