import cv2
import numpy as np
import matplotlib.pyplot as plt


def k_means_segmentation(img, k=2, max_iter=10):
    # Convert image to 1D array of pixels
    pixels = img.reshape((-1, 3)).astype(np.float32)

    # Define criteria for k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)

    # Perform k-means clustering
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape labels to match original image shape
    labels = labels.reshape(img.shape[:2])

    # Create a mask for each cluster
    masks = []
    for i in range(k):
        mask = np.zeros_like(labels, np.uint8)
        mask[labels == i] = 255
        masks.append(mask)

    # Calculate the segmented image
    segmented_img = np.zeros_like(img)
    for i, center in enumerate(centers):
        segmented_img[masks[i] > 0] = center

    return segmented_img, masks, centers


if __name__ == '__main__':
    # Load image
    img = cv2.imread("maxresdefault.jpg")

    # Perform k-means segmentation with 3 clusters and 10 iterations
    segmented_img, masks, centers = k_means_segmentation(img, k=3, max_iter=10)

    # Display original and segmented images
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[1].imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Segmented")
    plt.show()
