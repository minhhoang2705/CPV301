import numpy as np
import scipy.ndimage.filters as flt
import matplotlib.pyplot as plt

# Read image
img = plt.imread('maxresdefault.jpg')
# Convert to RGB color space
img = img[:, :, :3]
# Reshape image to a 2D array of pixels and 3 color values (RGB)
pixel_values = img.reshape((-1, 3))
# Convert to float type
pixel_values = np.float32(pixel_values)

# Define kernel function for mean shift algorithm
def kernel(x):
    return np.exp(-x**2/2)

# Define bandwidth parameter for mean shift algorithm
h = 0.1

# Define stopping criteria: 10 iterations or move by less than 0.01 pixel
max_iter = 10
epsilon = 0.01

# Initialize labels array with -1 values
labels = -np.ones(pixel_values.shape[0])

# Initialize cluster index with 0
cluster_idx = 0

for i in range(pixel_values.shape[0]):
    # Skip if pixel is already labeled
    if labels[i] != -1:
        continue
    
    # Initialize mean vector with current pixel value
    mean = pixel_values[i]
    
    # Initialize iteration counter with 0
    iter_count = 0
    
    while True:
        # Compute distance between mean vector and all pixels
        dists = np.linalg.norm(pixel_values - mean, axis=1)
        
        # Find pixels within bandwidth radius of mean vector
        neighbors_idx = np.where(dists < h)[0]
        
        # Update mean vector by applying kernel function and normalizing weights 
        new_mean = np.sum(kernel(dists[neighbors_idx])[:, None] * pixel_values[neighbors_idx], axis=0) / np.sum(kernel(dists[neighbors_idx]))
        
        # Compute distance between new mean vector and old mean vector 
        diff = np.linalg.norm(new_mean - mean)
        
        # Update mean vector with new mean vector 
        mean = new_mean
        
        # Increment iteration counter 
        iter_count += 1
        
        # Check stopping criteria 
        if diff < epsilon or iter_count > max_iter:
            break
    
    # Assign cluster index to current pixel and its neighbors 
    labels[i] = cluster_idx 
    labels[neighbors_idx] = cluster_idx 
    
    # Increment cluster index 
    cluster_idx += 1

# Convert back to int type 
labels = labels.astype(int)

# Find unique cluster indices 
unique_labels = np.unique(labels)

# Assign each cluster a random color 
colors = np.random.rand(len(unique_labels), 3)

# Replace each pixel value with its cluster color 
segmented_image = colors[labels]

# Reshape back to the original image dimension 
segmented_image = segmented_image.reshape(img.shape)

plt.imshow(segmented_image)
plt.show()