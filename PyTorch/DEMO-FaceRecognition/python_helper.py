import os
import os.path
import shutil

from random import sample
import numpy as np
import re

root_dir = "/DATASETs/Face/Face_SJC/Original_Data/"


def read_dir(dir_name):
    directories = []

    for x in os.listdir(os.path.join(root_dir, dir_name)):
        if os.path.isdir(os.path.join(root_dir, dir_name, x)):
            directories.append(os.path.join(root_dir, dir_name, x))
            #os.mkdir(os.path.join(root_dir, 'valid', x))

    return directories

#print(directories)


def move_files():
    for dir_path in read_dir('train'):
        files = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, x)) and x[-3:] == 'png']
        #print(files)
        rand_sample = sample(files, 20)

        for s in rand_sample:
            new_loc = s.replace('train', 'valid')
            shutil.move(s, new_loc)


def rename_files(dir_name):
    class_dir = read_dir(dir_name)
    
    print(class_dir)
    for c in class_dir:
        img = os.listdir(c)
        img.sort(key=lambda f: int(re.sub('\D', '', f)))
        
        for i, x in enumerate(img):
            #print(i, x)
            old_name = os.path.join(c, x)
            new_name = os.path.join(c, str(i)+'.png')

            #print(old_name, new_name, '\n')
            os.rename(old_name, new_name)
        #break

rename_files('train') 
rename_files('valid')