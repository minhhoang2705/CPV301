import cv2
import numpy as np

def snake(img, contour, alpha=0.1, beta=0.1, gamma=0.5, kappa=2):
    """
    Implements the Snakes algorithm for active contours
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define energy function
    def energy(img, contour):
        # Calculate external energy
        ext = cv2.Canny(img, 100, 200)
        ext_contour = np.zeros_like(ext)
        cv2.drawContours(ext_contour, [contour], 0, 255, 1)
        ext_energy = np.sum(ext * ext_contour)

        # Calculate internal energy
        int_energy = 0
        for i in range(len(contour)):
            j = (i + 1) % len(contour)
            dist = np.linalg.norm(contour[i] - contour[j])
            int_energy += dist

        return alpha * ext_energy - beta * int_energy

    # Iterate until convergence
    delta = 1
    old_energy = energy(gray, contour)
    while delta > 0.1:
        # Calculate new contour
        for i in range(len(contour)):
            j = (i - 1) % len(contour)
            k = (i + 1) % len(contour)

            x = contour[i][0] + gamma * (contour[j][0] + contour[k][0] -
                                         2 * contour[i][0]) - kappa * (contour[j][1] - contour[k][1])
            y = contour[i][1] + gamma * (contour[j][1] + contour[k][1] -
                                         2 * contour[i][1]) - kappa * (contour[k][0] - contour[j][0])

            contour[i] = (int(x), int(y))

        # Check for convergence
        new_energy = energy(gray, contour)
        if new_energy == 0:
            break
        delta = abs(new_energy - old_energy) / new_energy
        old_energy = new_energy

    return contour
