import numpy as np
import matplotlib.pyplot as plt

# chatgpt example image

image = np.array([
    # Pixel (0,0): Red=255, Green=0, Blue=0 (pure red)
    # Pixel (0,1): Red=0, Green=255, Blue=0 (pure green)
    [[255,   0,   0], [  0, 255,   0]],
    # Pixel (1,0): Red=0, Green=0, Blue=255 (pure blue)
    # Pixel (1,1): Red=255, Green=255, Blue=255 (white)
    [[  0,   0, 255], [255, 255, 255]]
])

# print(image.shape)
# print(image)

plt.imshow(image)

gray = image.mean(axis=2)

plt.imshow(gray)

plt.show()


