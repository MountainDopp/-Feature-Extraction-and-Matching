import numpy as np
import pandas as pd
import math
from skimage import io
from skimage.feature import blob_log
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from itertools import product



def normalize(differences, range=(0,1.0)):
	#linear rescaling
	
	max_val = max(differences)
	min_val = min(differences)

	return np.multiply(np.subtract(range[1], range[0]), np.divide( np.subtract(differences, min_val), np.subtract(max_val, min_val)))


def compute_log(img, PATCH_SIZE, min_sigma=1, max_sigma=8, num_sigma=1, treshold=0.2):
#Laplacian of Gaussian blob detection

  #detect all blobs 
  all_blobs = blob_log(img, min_sigma, max_sigma, num_sigma, treshold, overlap=0)
   
  #container to hold blobs that are within the range 
  blobs_within_range = []
	
  for items in all_blobs: #iterating through all blobs 
	#if x, y position minus/plus patch size exceeds image border, the blob will be discarded 
	#otherwise blob is added to blobs_within_range
	if items[1] + PATCH_SIZE > img.shape[1] or items[1] - PATCH_SIZE <= 0:
	  pass
	elif items[0] + PATCH_SIZE > img.shape[0] or items[0] - PATCH_SIZE <= 0:
	  pass
	else: 
	  blobs_within_range.append(items)
	
	#converting back to np.array to keep formatting of returned blobs in blob_log function
	blobs_within_range = np.array(blobs_within_range)

	return blobs_within_range

def make_patch(coordinate, PATCH_SIZE, img):
	# coordinate = [x, y, sigma]
	# PATCH_SIZE = int

	if PATCH_SIZE%2 == 0:
		raise ValueError('PATCH_SIZE must be an odd value')

	patch = np.zeros((PATCH_SIZE,PATCH_SIZE))
	for (i,j), _ in np.ndenumerate(patch):
		patch[i,j] = img[int(coordinate[0]-(PATCH_SIZE+1)/2) + i, int(coordinate[1]-(PATCH_SIZE+1)/2) + j]
	return patch


def viz1(img, interest_points, color='r'):
  """visualisation of the input image interest points

  img: input image 
  interest_points: list of interest points extracted from the blob_log function
  color: color of drawn circles, default red"""

	x = [a[1] for a in interest_points] #blob detection x axis
	y = [a[0] for a in interest_points] #blob detection y axis
	s = [a[2] for a in interest_points] #blob detected at sigma 
  
	plt.imshow(img, cmap='gray') #adding the input image to plot 
	for x, y, s in zip(x, y, s):
		plt.scatter(x, y, alpha=1, facecolors='none', edgecolors='r', s=s**2) #plotting the input interest points
	plt.axis('off')
	#plt.show() #showing the image // can be changed to saving the image locally 

	return


def count_difference(patch1, patch2):
  """
  sum of squared differences between values in patch1, patch2
  """

	return np.sum(np.square(patch1 - patch2))


def remove_array(L,arr):
	# taken from here: https://stackoverflow.com/questions/3157374/how-do-you-remove-a-numpy-array-from-a-list-of-numpy-arrays

	ind = 0
	size = len(L)
	while ind != size and not np.array_equal(L[ind],arr):
		ind += 1
	if ind != size:
		L.pop(ind)
	else:
		raise ValueError('array not found in list.')


def match(img1, img2, coordinates1, coordinates2, PATCH_SIZE):
"""
looks for the best match available for all of the interest points from coordinates1
TODO it maybe it should take a parameter 'override' so if its 
"""
# for convienince: making sure that the coordinates1 and 2 are lists not np.arrays
	coordinates1 = list(coordinates1)
	coordinates2 = list(coordinates2)

# in coordinates1 and image1 we want to have the one that has less or equal number of features
	swap = False #just to know if we have swapped
	if len(coordinates1) > len(coordinates2):
		# swap variables
		coordinates1, coordinates2 = coordinates2, coordinates1
		img1, img2, = img2, img1
		swap = True

	matches = []

	for feature1 in coordinates1:
		patch1 = make_patch(feature1, PATCH_SIZE, img1)
		best_match = [[-1, -1, -1], math.inf] # creating a temporary variable, with inf difference, so it will be replaced with the first feature2 from coordinates2

		for feature2 in coordinates2:
			patch2 = make_patch(feature2, PATCH_SIZE, img2)
			diff = count_difference(patch1, patch2)
			if diff < best_match[1]:    # if we find a better match we want to asign it to best_match var
				best_match = [feature2, diff]


# add the match to the match list and remove matched feature from img2 from coordinates2 so it's no longer avaliable
		if best_match[1] <= threshold:
			if swap: # need to check if the images aren't swapped and if so the output has to be 'unswapped'
				matches.append([best_match[0], feature1, best_match[1]])
			else:
				matches.append([feature1, best_match[0], best_match[1]])
		
		#OBS BE AWARE this is not working yet: 
			# because it's a list of list method .remove() doesn't work
			remove_array(coordinates2, best_match[0])
	return matches


def match2(img1, img2, coordinates1, coordinates2, PATCH_SIZE):
	"""
	calculates all differences between features in two images and looking for the best matches 
	filters the matches with difference lower or equal to threshold
	 """

	possible_matches = pd.DataFrame(columns=['feature1', 'feature2', 'diff'])

	# iteration through all the possible pairs of features from img1 and img2
	for (feature1, feature2) in product(coordinates1, coordinates2):
		patch1 = make_patch(feature1, PATCH_SIZE, img1)
		patch2 = make_patch(feature2, PATCH_SIZE, img2)
		diff = count_difference(patch1, patch2)
		# print(feature1, feature2, diff)
		# next line is weird, don't ask
		possible_matches = possible_matches.append({'feature1': feature1, 'feature2': feature2, 'diff': diff}, ignore_index=True)


	# sorting the possible_matches according to their difference and resetin the index
	# inplace=True means that changes will be performed in-place, so they will change the variable and there is no need to reassign
	possible_matches.sort_values(['diff'], inplace=True)
	possible_matches.reset_index(inplace=True, drop=True)
	# print("SORTED")
	# print(possible_matches)

	# adding only the best matches
	matches = []
	while not possible_matches.empty:
		best_match = possible_matches.loc[0]
		matches.append(list(best_match))
		# creating a list of indicies to be dropped
		trash = []

		#finding and deleting rows with features that are already used and reseting the index
		for i, (f1, f2, d) in enumerate(possible_matches.values):
		   
			if np.all(f1 == best_match[0]) or np.all(f2 == best_match[1]):
				trash.append(i)
	
		for i in trash:
			possible_matches.drop(i, inplace=True)
		possible_matches.reset_index(inplace=True, drop=True)
		# print(possible_matches)

	return matches


def match3(img1, img2, coordinates1, coordinates2, PATCH_SIZE, threshold=0.7):
	"""
	returns a list of coordinates that are their best matches both left-to-right and right-to-left
	so: [(x1, y1), (x2,y2), diff] will be returned iff:
	best_match((x1,y1)) = (x2,y2) and best_match((x2,y2)) = (x1,y1)
	"""

	#creating patches for all points from img1 and img2
	coord1_patches = [make_patch(coordinate, PATCH_SIZE, img1) for coordinate in coordinates1]
	coord2_patches = [make_patch(coordinate, PATCH_SIZE, img2) for coordinate in coordinates2]

	# creating a matrix with dissimilarity measures for all pairs
	all_matches = np.zeros((len(coordinates1), len(coordinates2)))

	for (x, y), _ in np.ndenumerate(all_matches):
		all_matches[x,y] = count_difference(coord1_patches[x], coord2_patches[y])

	#looking for best left-to-right and right-to-left matches
	matches = []
	#left-to-right
	for i, coord1 in enumerate(coordinates1):
		best_ltr_match = np.argmin(all_matches[i, :]) #best left-to-right match for coord1
		best_rtl_match = np.argmin(all_matches[:, best_ltr_match]) #best match for a best match
		if (i == best_rtl_match): #hurray, there is a super match

			matches.append([coord1, coordinates2[best_ltr_match], all_matches[i, best_ltr_match]])
	
	return matches


def statistics(img1_blobs, img2_blobs, matches):
	"""Used to provide statistics of the feature matching
	img1_blobs: blobs detected in img1
	img2_blobs: blobs detected in img2
	matches: the matched interest points from img1 to img2"""
	statistics = {}

	statistics['#Interest Points in img1'] = len(img1_blobs)
	statistics['#Interest Points in img2'] = len(img2_blobs)
	statistics['Accepted Matches'] = len(matches)
	dissimilarity = [match[2] for match in matches]
	statistics['Mean of accepted matches'] = sum(dissimilarity)/len(dissimilarity)
	statistics['SD of accepted matches'] = np.std(dissimilarity)
	return statistics

def viz2(img1, interest_points1, img2, interest_points2, matches, PATCH_SIZE, threshold, min_sigma, max_sigma, num_sigma):
	"""visualisation of the feature matches of the two input images 
  
  img1: image1
  interest_points1: list of blobs for img1
  img2: path to image2 
  interest_points2: list of blobs for img2 
  matches: list of matches 
  PATCH_SIZE: size of patch used 
  threshold: threshold used for detection of blob
  min_sigma: minimum sigma used for detection of blob
  max_sigma: maximum sigma used for detection of blob
  num_sigma: interval in sigma used for detection of blob
  """
  

	fig = plt.figure(figsize=(10,5))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)

  #adding the two images to axes 
	ax1.imshow(img1, cmap='gray')
	ax2.imshow(img2, cmap='gray')

	positionimg1 = ax1.get_position()
	new_pos = [positionimg1.x0+0.09, positionimg1.y0+0.025, \
		positionimg1.width / 1.1, positionimg1.height / 1.1] 
	ax1.set_position(new_pos)

	x1 = [a[1] for a in interest_points1] #blob detection x axis
	y1 = [a[0] for a in interest_points1] #blob detection y axis
	s1 = [a[2] for a in interest_points1] #blob detected at sigma 
  
	x2 = [a[1] for a in interest_points2] #blob detection x axis
	y2 = [a[0] for a in interest_points2] #blob detection y axis
	s2 = [a[2] for a in interest_points2] #blob detected at sigma 
  
	differences = [a[2] for a in matches]


	weighted_differences = normalize(differences)

  #iterating through the input list of matches
	for coordinates, difference in zip(matches, weighted_differences):
		cord_a = (coordinates[0][1], coordinates[0][0]) #extracting coordinates for interest point in img1
		cord_b = (coordinates[1][1], coordinates[1][0]) #extracting coordinates for interest point in img2
		if difference <=0.33:
			color = "green"
		elif difference > 0.33 and difference <= 0.66:
			color = "yellow"
		else:
			color = "red"

	#defining the path from cord_a to cord_b
		con = ConnectionPatch(xyA=cord_a, xyB=cord_b, coordsA="data", coordsB="data",
							  axesA=ax2, axesB=ax1, color=color) #arrowstyle='->')
	#adding line to axes2 
		ax2.add_artist(con)

  #showing the image // can be changed to saving the image locally 
	for x, y, s in zip(x1, y1, s1):
		ax1.scatter(x, y, alpha=1, facecolors='none', edgecolors='r', s=s**2) #plotting the input interest points for img1
	for x, y, s in zip(x2, y2, s2):
		ax2.scatter(x, y, alpha=1, facecolors='none', edgecolors='r', s=s**2) #plotting the input interest points for img2
	ax1.axis('off')
	ax2.axis('off')
	title = 'Patch Size=' + str(PATCH_SIZE) + ', Threshold=' + str(threshold) + ', min sigma=' + \
	str(min_sigma) + ', max sigma=' + str(max_sigma) + ', num sigma=' + str(num_sigma)
	plt.title(title, x=+0.1)
	#plt.show()
	plt.savefig(title+'.png')


	return

