#### SCRIPT

from library import *

import numpy as np
from skimage import io, img_as_float
import random


PATCH_SIZE = 5
FILE_NAME1 = 'Img001_diffuse_smallgray.png'
FILE_NAME2 = 'Img002_diffuse_smallgray.png'
FILE_NAME9 = 'Img002_diffuse_smallgray.png'

img1 = img_as_float(io.imread(FILE_NAME1))
img2 = img_as_float(io.imread(FILE_NAME2))
img9 = img_as_float(io.imread(FILE_NAME9))

# img1_blobs = [[77,34,1], [20,20,1], [40,40,1], [60,60,1], [122,121,1]]
# img2_blobs = [[40,40,2], [60,60,2], [20,20,2], [121, 222,2], [150, 250,2]]
# random.shuffle(img2_blobs)
# print(match(img1, img2, img1_blobs, img2_blobs, PATCH_SIZE))
# print(match2(img1, img2, img1_blobs, img2_blobs, PATCH_SIZE))

img1_blobs = compute_log(img1, PATCH_SIZE)
img2_blobs = compute_log(img2, PATCH_SIZE)

# matches1_2 = match(img1, img2, img1_blobs, img2_blobs, PATCH_SIZE)
# matches1_2 = match2(img1, img2, img1_blobs, img2_blobs, PATCH_SIZE)
# print(matches1_2)

matches3 = match3(img1, img2, img1_blobs, img2_blobs, PATCH_SIZE)

# print(img1_blobs)

# viz1(img1, img1_blobs)

"""
viz1(img1, img1_blobs)


#looking for blobs in img1 and img9

img2_blobs = compute_log(img2, PATCH_SIZE)
#img9_blobs = compute_log(img9, PATCH_SIZE)

#looking for matches

matches1_2 = match(img1, img2, img1_blobs, img2_blobs, PATCH_SIZE)

viz2(img1, img1_blobs, img2, img2_blobs, matches1_2)"""