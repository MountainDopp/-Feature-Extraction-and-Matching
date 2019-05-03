from skimage import io, img_as_float
from library import *


min_sigma = [1, 2] 
max_sigma = [10, 10] 
num_sigma = [4, 4]
threshold = [0.2, 0.2]
PATCH_SIZE = [5, 5]

FILE_NAME1 = 'Img001_diffuse_smallgray-1.png'
FILE_NAME2 = 'Img009_diffuse_smallgray.png'
FILE_NAME9 = 'Img002_diffuse_smallgray.png'

img1 = img_as_float(io.imread(FILE_NAME1))
img2 = img_as_float(io.imread(FILE_NAME2))
img9 = img_as_float(io.imread(FILE_NAME9))

for mis, mas, nus, thres, patch in zip(min_sigma, max_sigma, num_sigma, threshold, PATCH_SIZE):
  
  img1_blobs = compute_log(img1, patch, mis, mas, nus, thres)
  viz1(img1, img1_blobs, patch, thres, mis, mas, nus)


min_sigma = 2
max_sigma = 10 
num_sigma = 4
threshold = 0.2
PATCH_SIZE = [5, 7, 9, 13]

for patch in PATCH_SIZE:
	img1_blobs = compute_log(img1, patch, min_sigma, max_sigma, num_sigma, threshold)
	img2_blobs = compute_log(img2, patch, min_sigma, max_sigma, num_sigma, threshold)
	img9_blobs = compute_log(img9, patch, min_sigma, max_sigma, num_sigma, threshold)

	matches1_2 = match3(img1, img2, img1_blobs, img2_blobs, patch)
	matches1_9 = match3(img1, img9, img1_blobs, img9_blobs, patch)

	print('Statistics for match between image 1 and image 2, with Patch size=' + str(patch))
	print(statistics(img1_blobs, img2_blobs, matches1_2))
	print()
	print('Statistics for match between image 1 and image 9, with Patch size=' + str(patch))
	print(statistics(img1_blobs, img9_blobs, matches1_9))
	print()
	viz2(img1, img1_blobs, img2, img2_blobs, matches1_2, patch, threshold, min_sigma, max_sigma, num_sigma, '1_2')
	viz2(img1, img1_blobs, img9, img9_blobs, matches1_9, patch, threshold, min_sigma, max_sigma, num_sigma, '1_9')

	if patch == 5 or patch == 13:
		match_john = match2(img1, img9, img1_blobs, img9_blobs, patch)
		viz2(img1, img1_blobs, img9, img9_blobs, match_john, patch, threshold, min_sigma, max_sigma, num_sigma, 'Different match_function 1 to 9')
		print()
		print('Statistics for match between image 1 and image 9 with different match function, with Patch size=' + str(patch))
		print(statistics(img1_blobs, img9_blobs, match_john))
		print()


