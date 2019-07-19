from os import listdir
from os import walk
from os import path

import csv
import numpy as np


def compute_mean_and_var(indv_coord_lists):
# Caculate mean and variance
    coord_np = np.array(indv_coord_lists).astype(np.float)

    mean = coord_np.mean(axis=0)
    var = coord_np.var(axis=0)
    mean_var_list = np.hstack([mean, var])

    return mean_var_list
    
    #tot_mean_var_list.append(mean_var_list)

def compute_landmark_distances_numpy_array(indv_coord_list):
    centre_point = 33
    indv_coord_np = np.array(indv_coord_list, dtype=np.float)

    for coord_list in indv_coord_np:
        #print(np.shape(x_coord_row), np.shape(x_coord_row))
            
        x_coord_np = coord_list[::2]
        y_coord_np = coord_list[1::2]

        x_coord_np = np.power(x_coord_np - x_coord_np[centre_point], 2)
        y_coord_np = np.power(y_coord_np - y_coord_np[centre_point], 2)

        distance_np = np.sqrt(x_coord_np + y_coord_np)
        break
        
    return distance_np


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
    print(csv_file_path)
    coord_lists = []
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for row in csv_reader:
            coord_lists.append(row)

    return np.array(coord_lists, dtype=np.float)

if __name__=="__main__":
    
    dir_path = "/DATASETs/Face/Face-SJC/Temp-Detection-Check-Data"

    csv_file_paths_list = get_csv_file_paths(dir_path)
    #print(csv_file_paths)
    coords_np = read_a_csv_file_and_return_a_numpy_float_array(csv_file_paths_list[0])
    #print(coord_np.shape)
    distances_np = compute_landmark_distances_numpy_array(coords_np)
    print(distances_np)