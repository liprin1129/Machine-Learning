from os import listdir
from os import walk
from os import path

import csv
import numpy as np


def get_csv_file_paths(dir_path):
# Get only csv file path
    csv_files = []
    # r=root, d=directories, f = files
    for r, d, f in walk(dir_path):
        for file in f:
            if '.csv' in file:
                csv_files.append(path.join(r, file))

    return csv_files


def read_a_csv_file_and_return_a_numpy_float_array(csv_file_path):
    # Read CSV
    #print(csv_file_path)
    coord_lists = []
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for row in csv_reader:
            coord_lists.append(row)

    return np.array(coord_lists, dtype=np.float64)


def compute_landmark_distances_numpy_array(indv_coord_np):
    centre_point = 33
    indv_coord_np = indv_coord_np.astype(dtype=np.float64)
    
    indv_abs_distance_np = np.empty([indv_coord_np.shape[0], int(indv_coord_np.shape[1]/2)], dtype=np.float64)

    for idx, coord_list in enumerate(indv_coord_np):
        #print(np.shape(x_coord_row), np.shape(x_coord_row))
            
        x_coord_np = coord_list[::2]
        y_coord_np = coord_list[1::2]
        #print(x_coord_np.shape, y_coord_np.shape)

        #print(x_coord_np, y_coord_np)
        #print(x_coord_np[centre_point], y_coord_np[centre_point])
        
        x_diff_np = x_coord_np - x_coord_np[centre_point]
        y_diff_np = y_coord_np - y_coord_np[centre_point]
        #print(x_diff_np[0], y_diff_np[0])

        x_coord_np = np.power(x_diff_np, 2)
        y_coord_np = np.power(y_diff_np, 2)
        #print(x_coord_np[0], y_coord_np[0])

        distance_np = np.sqrt(x_coord_np + y_coord_np)
        indv_abs_distance_np[idx, :] = distance_np
        break
        
    return np.round(indv_abs_distance_np.astype(np.float64), 4)


def compute_mean_and_var(indv_coord_np):
# Caculate mean and variance
    coord_np = indv_coord_np.astype(np.float64)
    mean_var_np = np.empty([coord_np.shape[1]*2])

    mean = coord_np.mean(axis=0)
    var = coord_np.var(axis=0)
    mean_var_np[::2] = mean
    mean_var_np[1::2] = var
    #mean_var_np = np.hstack([mean, var])

    return np.round(mean_var_np.astype(np.float64), 4)

def wirte_a_maen_and_var_csv_file(csv_name, csv_file_paths_list):
# Write a csv file for the calculated means and variances

    tot_means_and_vars_list = []
    for csv_file_path in csv_file_paths_list:
        # Get coordinates
        coords_np = read_a_csv_file_and_return_a_numpy_float_array(csv_file_path)
        #print(coords_np.shape)
        
        # Get distances
        distances_np = compute_landmark_distances_numpy_array(coords_np)
        #print(distances_np.shape)

        # Merge coordinates and distances
        features_np = np.hstack([coords_np, distances_np])
        #print(features_np.shape)

        # Get means and variances
        indv_mean_and_var_np = compute_mean_and_var(features_np)
        #print(indv_mean_and_var_np.shape)

        tot_means_and_vars_list.append(indv_mean_and_var_np)

    tot_means_and_vars_np = np.round(np.array(tot_means_and_vars_list, dtype=np.float64), 4)
    print(tot_means_and_vars_np.shape)

    with open(csv_name, 'w') as write_file:
        writer = csv.writer(write_file)
        header = []
        for i in range(int(tot_means_and_vars_np.shape[1]/2)):
            header.append("mean:{0}".format(i+1))
            header.append("variance:{0}".format(i+1))
        writer.writerow(header)

        for coord in tot_means_and_vars_np:
            writer.writerow(coord)


if __name__=="__main__":
    
    dir_path = "/DATASETs/Face/Face-SJC/Temp-Detection-Check-Data"

    csv_file_paths_list = get_csv_file_paths(dir_path)
    #print(csv_file_paths_list)

    wirte_a_maen_and_var_csv_file("/DATASETs/Face/Face-SJC/Temp-Detection-Check-Data/face-master.csv", csv_file_paths_list)