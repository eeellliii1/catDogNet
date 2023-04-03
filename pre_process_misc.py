import os
from posixpath import dirname
import sys
import cv2

def import_from_source(dir_name):
    imgs = []
    img_names = []

    # Get the filenames of all images in dir
    pic_ls = os.listdir(dir_name)

    for pic_name in pic_ls:
        pixel_inf = cv2.imread(dir_name + '/' + pic_name)
        imgs.append(pixel_inf)
        img_names.append(pic_name)

    return imgs, img_names


def process(imgs):
    proc_imgs = []
    for img in imgs:
        proc_imgs.append(cv2.resize(img, (100, 100)))

    return proc_imgs
    

def write_to_dest(dir_name, imgs, img_names):
    for i in range(len(imgs)):
        dest_path = dir_name + '/' + img_names[i]

        cv2.imwrite(dest_path, imgs[i])


###################################
# NOW DEPRICATED as make_nn.py and classify_nn.py automatically resize images
#                pre_process_misc.py can be used to view the resized images for quality verification and demonstration
#
# Command in format of: <source_dir> <destination_dir>

args = sys.argv

source_dir = args[1]
destination_dir = args[2]

imgs, img_names = import_from_source(source_dir)
proc_imgs = process(imgs)
write_to_dest(destination_dir, proc_imgs, img_names)