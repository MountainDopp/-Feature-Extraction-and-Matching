from skimage import io, img_as_float
from library import *


min_sigma = [1, 2] 
max_sigma = [10, 10] 
num_sigma = [4, 4]
threshold = [0.2, 0.2]
PATCH_SIZE = [5, 5]

FILE_NAME1 = '../Img001_diffuse_smallgray.png'
FILE_NAME2 = '../Img002_diffuse_smallgray.png'
FILE_NAME9 = '../Img009_diffuse_smallgray.png'

img1 = img_as_float(io.imread(FILE_NAME1))
img2 = img_as_float(io.imread(FILE_NAME2))
img9 = img_as_float(io.imread(FILE_NAME9))

min_sigma = [0.5, 20, 2, 2, 2] 
max_sigma = [0.5, 20, 10, 10, 10] 
num_sigma = [1, 1, 4, 4, 4]
threshold = [0.2, 0.2, 0.3, 0.1, 0.01]
PATCH_SIZE = [5, 5, 5, 5, 5]


FILE_NAME1 = '../Img001_diffuse_smallgray.png'
FILE_NAME2 = '../Img002_diffuse_smallgray.png'
FILE_NAME9 = '../Img009_diffuse_smallgray.png'

img1 = img_as_float(io.imread(FILE_NAME1))
img2 = img_as_float(io.imread(FILE_NAME2))
img9 = img_as_float(io.imread(FILE_NAME9))

# for mis, mas, nus, thres, patch in zip(min_sigma, max_sigma, num_sigma, threshold, PATCH_SIZE):
# 	img1_blobs = compute_log(img1, patch, mis, mas, nus, thres)
# 	viz1(img1, img1_blobs, patch, thres, mis, mas, nus, name='exp')

# for mis, mas, nus, thres, patch in zip(min_sigma, max_sigma, num_sigma, threshold, PATCH_SIZE):
  
#   img1_blobs = compute_log(img1, patch, mis, mas, nus, thres)
#   img2_blobs = compute_log(img2, patch, mis, mas, nus, thres)
#   img9_blobs = compute_log(img9, patch, mis, mas, nus, thres)
#   viz1(img1, img1_blobs, patch, thres, mis, mas, nus, name='img1')
#   viz1(img2, img2_blobs, patch, thres, mis, mas, nus, name='img2')
#   viz1(img9, img9_blobs, patch, thres, mis, mas, nus, name='img9')


min_sigma = 2
max_sigma = 10
num_sigma = 4
threshold = 0.2
PATCH_SIZE = [5, 13]

for patch in PATCH_SIZE:
	img1_blobs = compute_log(img1, patch, min_sigma, max_sigma, num_sigma, threshold)
	img2_blobs = compute_log(img2, patch, min_sigma, max_sigma, num_sigma, threshold)
	img9_blobs = compute_log(img9, patch, min_sigma, max_sigma, num_sigma, threshold)

	#match3 fun
	matches1_2 = match3(img1, img2, img1_blobs, img2_blobs, patch)
	matches1_9 = match3(img1, img9, img1_blobs, img9_blobs, patch)

	# print('Statistics for match between image 1 and image 2, with Patch size=' + str(patch))
	# print(statistics(img1_blobs, img2_blobs, matches1_2))
	# print()
	# print('Statistics for match between image 1 and image 9, with Patch size=' + str(patch))
	# print(statistics(img1_blobs, img9_blobs, matches1_9))
	# print()
	viz2(img1, img1_blobs, img2, img2_blobs, matches1_2, patch, threshold, min_sigma, max_sigma, num_sigma, '1_2_match3')
	viz2(img1, img1_blobs, img9, img9_blobs, matches1_9, patch, threshold, min_sigma, max_sigma, num_sigma, '1_9_match3')


	#match2 fun img 1 and img 2
	match_john12 = match2(img1, img2, img1_blobs, img2_blobs, patch)
	"""
	print()
	print('Statistics for match between image 1 and image 9 with different match function, with Patch size=' + str(patch))
	print(statistics(img1_blobs, img2_blobs, match_john))
	print()
"""
	#match 2 fun img1 and img 9
	match_john19 = match2(img1, img9, img1_blobs, img9_blobs, patch)

	viz2(img1, img1_blobs, img2, img2_blobs, match_john12, patch, threshold, min_sigma, max_sigma, num_sigma, '1_2_match2')
	viz2(img1, img1_blobs, img9, img9_blobs, match_john19, patch, threshold, min_sigma, max_sigma, num_sigma, '1_9_match2')
	"""
	print()
	print('Statistics for match between image 1 and image 9 with different match function, with Patch size=' + str(patch))
	print(statistics(img1_blobs, img9_blobs, match_john))
	print()"""


