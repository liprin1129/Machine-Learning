import numpy as np
#from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import pyplot as plt
import json
import glob

if __name__=='__main__':
    """
    root_dir = "../Datasets/"

    json_files = [jf for jf in glob.glob(root_dir + "/3D-Coordinates/Sitting/*.json")]

    # Sitting data
    x_sitting_coord_list = []
    y_sitting_coord_list = []
    z_sitting_coord_list = []
    for json_file in json_files:
        with open(json_file, 'r') as jf:
            json_data = json.load(jf)
            keys = list(json_data.keys())
            '''x_sitting_coord_list.append(json_data['Right Elbow'][0])
            y_sitting_coord_list.append(json_data['Right Elbow'][1])
            z_sitting_coord_list.append(json_data['Right Elbow'][2])            
            '''
            for key in keys[:-1]:
                x_sitting_coord_list.append(json_data[key][0])
                y_sitting_coord_list.append(json_data[key][1])
                z_sitting_coord_list.append(json_data[key][2])

    json_files = [jf for jf in glob.glob(root_dir + "/3D-Coordinates/Standing/*.json")]
    """
    # Standing data
    json_files = ["./key.json"]
    x_standing_coord_list = []
    y_standing_coord_list = []
    z_standing_coord_list = []
    for json_file in json_files:
        with open(json_file, 'r') as jf:
            json_data = json.load(jf)
            keys = list(json_data.keys())
            '''
            x_standing_coord_list.append(json_data['Right Elbow'][0])
            y_standing_coord_list.append(json_data['Right Elbow'][1])
            z_standing_coord_list.append(json_data['Right Elbow'][2])            
            '''
            for key in keys[:-1]:
                #if idx not in [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
                if key not in ["Nose", "LEye", "REye", "LEar", "REar", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"]:
                    x_standing_coord_list.append(json_data[key][0])
                    y_standing_coord_list.append(json_data[key][1])
                    z_standing_coord_list.append(json_data[key][2])

    # Plotting
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    #ax.scatter(x_sitting_coord_list, y_sitting_coord_list, z_sitting_coord_list, marker='o')
    ax.set_xlim(1550, 3550)
    ax.set_ylim(0,1000)
    ax.scatter(x_standing_coord_list, y_standing_coord_list, z_standing_coord_list, marker='^')

    plt.show()
    '''
    for x in json_data:
        print("%s: %d" % (x, json_data[x]))
    '''
