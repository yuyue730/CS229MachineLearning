from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    height, width, colors = image.shape
    # The initialized clusters should have a size of num_clusters * C. So before randomization, we
    # make an zeros matrix with such a size.
    centroids_init = np.zeros((num_clusters, colors))

    for i in range(num_clusters):
        cur_h = random.randint(0, height - 1)
        cur_w = random.randint(0, width - 1)
        # Initialize centeriods index at i with the same color as image at pixel [cur_h, cur_w].
        centroids_init[i, :] = image[cur_h, cur_w, :]
    # *** END YOUR CODE ***

    return centroids_init

def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('update_centroids function not implemented')
    # Usually expected to converge long before `max_iter` iterations
    #   Initialize `dist` vector to keep track of distance to every centroid
    #   Loop over all centroids and store distances in `dist`
    #   Find closest centroid and update `new_centroids`
    # Update `new_centroids`
    height, width, colors = image.shape
    num_clusters = len(centroids)
    new_centroids = np.zeros((num_clusters, colors))

    for iter in range(max_iter):
        # Defines how many pixels are assigned to each cluster. This should be a cluster-sized
        # vector.
        cluster_assignments_count = np.zeros((num_clusters, 1))
        # The accumulation of each color's values at each new cluster at current iteration. When
        # each element is divided by cluster_assignments_count, we derive the new clusters.
        cluster_new_accumulation_each_color = np.zeros((num_clusters, colors))
        for cur_h in range(height):
            for cur_w in range(width):
                # all_distances_to_centroids is a vector tracking the distance between color of
                # image pixel (cur_h, cur_w) and color of each clusters in centroids
                all_distances_to_centroids = np.zeros((num_clusters, 1))
                for cluster_idx in range(num_clusters):
                    dist_vector = image[cur_h, cur_w] - centroids[cluster_idx]
                    # Take the norm of the distance vector, which is equal to
                    # sqrt(x1^2 + x2^2 + x3^2).
                    dist_norm = np.linalg.norm(dist_vector)
                    all_distances_to_centroids[cluster_idx] = dist_norm

                # Check which cluster is the closest to pixel [cur_h, cur_w].
                closest_cluster_idx = np.argmin(all_distances_to_centroids)
                cluster_assignments_count[closest_cluster_idx] += 1
                cluster_new_accumulation_each_color[closest_cluster_idx] += image[cur_h, cur_w, :]

        # After going over all pixels in the image at this iteration, we can derive the next
        # clusters (centroids)
        for cluster_idx in range(num_clusters):
            if cluster_assignments_count[cluster_idx] > 0:
                new_centroids[cluster_idx, :]\
                    = (cluster_new_accumulation_each_color[cluster_idx, :]
                       / cluster_assignments_count[cluster_idx])

        # Conditionally print iteration count in the first iteration or every print_every.
        if iter == 0 or iter % print_every == 0:
            print("Complete iteration index = {}, maximal iteration = {}".format(iter, max_iter))
    # *** END YOUR CODE ***

    return new_centroids

def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('update_image function not implemented')
    # Initialize `dist` vector to keep track of distance to every centroid
    # Loop over all centroids and store distances in `dist`
    # Find closest centroid and update pixel value in `image`
    height, width, colors = image.shape
    num_clusters = len(centroids)

    for cur_h in range(height):
        for cur_w in range(width):
            # all_distances_to_centroids is a vector tracking the distance between color of
            # image pixel (cur_h, cur_w) and color of each clusters in centroids
            all_distances_to_centroids = np.zeros((num_clusters, 1))
            for cluster_idx in range(num_clusters):
                dist_vector = image[cur_h, cur_w] - centroids[cluster_idx]
                # Take the norm of the distance vector, which is equal to
                # sqrt(x1^2 + x2^2 + x3^2).
                dist_norm = np.linalg.norm(dist_vector)
                all_distances_to_centroids[cluster_idx] = dist_norm

            # Check which cluster is the closest to pixel [cur_h, cur_w].
            closest_cluster_idx = np.argmin(all_distances_to_centroids)
            image[cur_h, cur_w, :] = centroids[closest_cluster_idx]
    # Go through all pixels to find out the closest centroid, and update to that centroid.
    # *** END YOUR CODE ***

    return image

def main(args):
    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = mpimg.imread(image_path_large).copy()
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
