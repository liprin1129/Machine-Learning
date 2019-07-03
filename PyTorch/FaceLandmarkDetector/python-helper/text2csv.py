from os import listdir
from os.path import join
import csv

dir_path = "/DATASETs/Face/Landmarks/300W-Dataset/300W/Data/"
file_list = listdir(dir_path)

# Read files in a directory
img_file = []
pts_file = []
for f in file_list:
    if f[-3:] == "png":
        img_file.append(f)
    elif f[-3:] == "pts":
        pts_file.append(f)

tot_coord_list = []
for pts_f in pts_file:
    coord_list = [pts_f[:-3]+"png"]
    with open(join(dir_path, pts_f), 'r') as opened_file:
        str_list = opened_file.readlines()[2:][1:-1] # extracts only coordinate strings
        for coord_str in str_list:
            coord = coord_str.split()
            coord_list.append(float(coord[0]))
            coord_list.append(float(coord[1]))

    tot_coord_list.append(coord_list)

# Write a csv file
with open("face_landmarks.csv", 'w') as write_file:
    writer = csv.writer(write_file)
    header = ["image_name"]
    for i in range(68):
        header.append("part_{0}_x".format(i))
        header.append("part_{0}_y".format(i))
    writer.writerow(header)

    for coord in tot_coord_list:
        writer.writerow(coord)