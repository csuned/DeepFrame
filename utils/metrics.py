import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def compare_images(imageA, imageB, title='', makeplot=False, data_range=255):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mean_squared_error(imageA, imageB)
    s = ssim(imageA, imageB, data_range=data_range)

    if makeplot:
        # setup the figure
        fig = plt.figure(title)
        plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA, cmap = plt.cm.gray)
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB, cmap = plt.cm.gray)
        plt.axis("off")

        # show the images
        plt.show()
    else:
        return m, s

def compare_images_series(seriesA, seriesB):
    s_ssim = []
    s_mse = []
    for i in range(len(seriesA)):
        m, s = compare_images(seriesA[i], seriesB[i], makeplot=False)
        s_ssim.append(s)
        s_mse.append(m)
    return s_mse, s_ssim