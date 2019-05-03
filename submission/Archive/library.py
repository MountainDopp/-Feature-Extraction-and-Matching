import math
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from itertools import product
from matplotlib import image
from matplotlib.patches import ConnectionPatch

from skimage import io
from skimage.feature import blob_log


def compute_log(img, PATCH_SIZE, min_sigma=1, max_sigma=10, num_sigma=1, threshold=0.2):
#Laplacian of Gaussian blob detection

    all_blobs = blob_log(img, min_sigma, max_sigma, num_sigma, threshold, overlap=0)
    
    #creating list for holding blobs that are within the range when applying the patch 
    blobs_within_range = []
    
    #if the patch_size plus the coordinate exceeds the image border the blob will be discarded.
    for items in all_blobs: 
        if items[1] + PATCH_SIZE > img.shape[1] or items[1] - PATCH_SIZE <= 0:
            pass
        elif items[0] + PATCH_SIZE > img.shape[0] or items[0] - PATCH_SIZE <= 0:
            pass
        else:
            blobs_within_range.append(items)
    blobs_within_range = np.array(blobs_within_range) #converting back to np.array

    return blobs_within_range


def make_patch(coordinate, PATCH_SIZE, img):
# coordinate = [x, y, sigma]
# PATCH_SIZE must be in 2k+1

    if PATCH_SIZE%2 == 0:
        raise ValueError('PATCH_SIZE must be an odd value')

    patch = np.zeros((PATCH_SIZE,PATCH_SIZE))
    for (i,j), _ in np.ndenumerate(patch):
        patch[i,j] = img[int(coordinate[0]-(PATCH_SIZE+1)/2) + i, int(coordinate[1]-(PATCH_SIZE+1)/2) + j]

    return patch


def viz1(img, interest_points, PATCH_SIZE, threshold, min_sigma, max_sigma, num_sigma, name=''):
    """
    Plotting interest points on the given image.
    """

    x = [a[1] for a in interest_points] #blob detection x axis
    y = [a[0] for a in interest_points] #blob detection y axis
    s = [a[2] for a in interest_points] #blob detected at sigma 

    title = name+'Blob detection at Threshold=' + str(threshold) + ', min sigma=' + \
    str(min_sigma) + ', max sigma=' + str(max_sigma) + ', num sigma=' + str(num_sigma)
    plt.imshow(img, cmap='gray') #adding the input image to plot 
    for x, y, s in zip(x, y, s):
        plt.scatter(x, y, alpha=1, facecolors='none', edgecolors='r', s=s**2) #plotting the input interest points
    plt.axis('off')
    plt.savefig('imagecontainer/'+title+'.png')
    plt.clf()

    return


def count_difference(patch1, patch2, PATCH_SIZE):
    # difference between two patches. L-2 norm

    return np.divide(np.sum(np.square(patch1 - patch2)), np.power(PATCH_SIZE, 2))



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


def match2(img1, img2, coordinates1, coordinates2, PATCH_SIZE):
    """
    calculates all differences between features in two images and looking for the best matches
     """

    possible_matches = pd.DataFrame(columns=['feature1', 'feature2', 'diff'])

    # iteration through all the possible pairs of features from img1 and img2
    for (feature1, feature2) in product(coordinates1, coordinates2):
        patch1 = make_patch(feature1, PATCH_SIZE, img1)
        patch2 = make_patch(feature2, PATCH_SIZE, img2)
        diff = count_difference(patch1, patch2, PATCH_SIZE)
        # next line is weird, don't ask
        possible_matches = possible_matches.append({'feature1': feature1, 'feature2': feature2, 'diff': diff}, ignore_index=True)

    # sorting the possible_matches according to their difference and resetin the index
    # inplace=True means that changes will be performed in-place, so they will change the variable and there is no need to reassign
    possible_matches.sort_values(['diff'], inplace=True)
    possible_matches.reset_index(inplace=True, drop=True)

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

    return matches


def match3(img1, img2, coordinates1, coordinates2, PATCH_SIZE):
    """
    returns a list of coordinates that are their best matches both left-to-right and right-to-left
    so: [(x1, y1), (x2,y2), diff] will be returned iff:
    best_match((x1,y1)) = (x2,y2) and best_match((x2,y2)) = (x1,y1)
    """

    #creating patches for all points from img1 and img2
    coord1_patches = np.array([make_patch(coordinate, PATCH_SIZE, img1) for coordinate in coordinates1])
    coord2_patches = np.array([make_patch(coordinate, PATCH_SIZE, img2) for coordinate in coordinates2])

    # creating a matrix with dissimilarity measures for all pairs
    all_matches = np.zeros((len(coordinates1), len(coordinates2)))

    for (x, y), _ in np.ndenumerate(all_matches):
        all_matches[x,y] = count_difference(coord1_patches[x], coord2_patches[y], PATCH_SIZE)

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
    """
    fuction for counting statistics: mean, SD, blob numbers and accepted points of interest
    """


    statistics = {}
    blobs1 = len(img1_blobs)
    blobs2 = len(img2_blobs)
    matched = len(matches)
    statistics['#Interest Points in img1'] = blobs1
    statistics['#Interest Points in img2'] = blobs2
    statistics['Accepted Matches'] = matched

    dissimilarity = [match[2] for match in matches]
    statistics['Mean of accepted matches'] = round(sum(dissimilarity)/len(dissimilarity), 5)
    statistics['SD of accepted matches'] = np.round(np.std(dissimilarity), decimals=5)

    if blobs1 < blobs2:
        statistics['Ratio of interest points and matches'] = round(matched/blobs1, 5)
    else:
        statistics['Ratio of interest points and matches'] = round(matched/blobs2, 5)

    return statistics


def viz2(img1, interest_points1, img2, interest_points2, matches, PATCH_SIZE, threshold, min_sigma, max_sigma, num_sigma, from_to):
    """
    Visualization of matched and not matched points on two images.
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
  
    differences = [a[2] for a in matches] #extracting the difference 

    weighted_differences = normalize(differences) #normalising the difference between 0 and 1

  #iterating through the input list of matches
    for coordinates, difference in zip(matches, weighted_differences):
        cord_a = (coordinates[0][1], coordinates[0][0]) #extracting coordinates for interest point in img1
        cord_b = (coordinates[1][1], coordinates[1][0]) #extracting coordinates for interest point in img2
        if difference <=0.33: #defining how good of a match it is
            color = "green"
        elif difference > 0.33 and difference <= 0.66:
            color = "yellow"
        else:
            color = "red"

    #defining the path from cord_a to cord_b
    # the below 2 lines of code were taken from https://stackoverflow.com/questions/17543359/drawing-lines-between-two-plots-in-matplotlib and modified
        con = ConnectionPatch(xyA=cord_b, xyB=cord_a, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color=color) #arrowstyle='->')
    #adding line to axes2 
        ax2.add_artist(con)

  #Plotting the interest points in the two images 
    for x, y, s in zip(x1, y1, s1):
        ax1.scatter(x, y, alpha=1, facecolors='none', edgecolors='r', s=s**2) #plotting the input interest points for img1
    for x, y, s in zip(x2, y2, s2):
        ax2.scatter(x, y, alpha=1, facecolors='none', edgecolors='r', s=s**2) #plotting the input interest points for img2
    ax1.axis('off')
    ax2.axis('off')
    title = 'Matchingof_' + str(from_to) + '_PatchSize=' + str(PATCH_SIZE) + ',Threshold=' + str(threshold) + ',mins=' + \
    str(min_sigma) + ',maxs=' + str(max_sigma) + ',nums=' + str(num_sigma)
    plt.title(title, x=+0.1)
    plt.savefig('imagecontainer/'+title+'.png')
    plt.clf()
    return


def normalize(differences, range=(0,1.0)):
    #linear rescaling

    max_val = max(differences)
    min_val = min(differences)

    return np.multiply(np.subtract(range[1], range[0]), np.divide( np.subtract(differences, min_val), np.subtract(max_val, min_val)))