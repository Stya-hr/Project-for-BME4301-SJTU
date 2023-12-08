from skimage import morphology
import numpy as np


def conditional_dilation(binary_image):
    marker = morphology.opening(binary_image)
    print('Conditional Dilation Loop ...')
    while True:
        target = marker.copy()
        marker = morphology.dilation(marker)
        marker = marker * binary_image
        if np.array_equal(target, marker):
            print('Conditional Dilation Complete !')
            return marker


def grayscale_reconstruction(gray_image):
    marker = morphology.dilation(gray_image)
    eps = 1e-4
    print('Grayscale Reconstruction Loop ...')
    while True:
        target = marker.copy()
        marker = morphology.dilation(marker)
        marker = np.minimum(marker, gray_image)
        if np.max(np.abs(target - marker)) < eps:
            print('Grayscale Reconstruction Loop Complete !')
            return marker


def OBR_func(gray_image):
    open_image = morphology.opening(gray_image)
    reconst_image = grayscale_reconstruction(open_image)
    return reconst_image


def CBR_func(gray_image):
    close_image = morphology.closing(gray_image)
    reconst_image = grayscale_reconstruction(close_image)
    return reconst_image
