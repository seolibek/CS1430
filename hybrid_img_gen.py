import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def plot_images(images: list, titles: list, rows: int, columns: int, fig_width=15, fig_height=7):
    fig = plt.figure(figsize=(fig_width, fig_height))
    for idx, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(rows, columns, idx + 1)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(title)

def calculate_distance(point1, point2):

    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def create_gaussian_low_pass_filter(D0, image_shape):
    filter_matrix = np.zeros(image_shape[:2])
    rows, cols = image_shape[:2]
    center = (rows / 2, cols / 2)
    for i in range(rows):
        for j in range(cols):
            filter_matrix[i, j] = np.exp(-calculate_distance((i, j), center) ** 2 / (2 * D0 ** 2))
    return filter_matrix

def create_gaussian_high_pass_filter(D0, image_shape):
    filter_matrix = np.zeros(image_shape[:2])
    rows, cols = image_shape[:2]
    center = (rows / 2, cols / 2)
    for i in range(rows):
        for j in range(cols):
            filter_matrix[i, j] = 1 - np.exp(-calculate_distance((i, j), center) ** 2 / (2 * D0 ** 2))
    return filter_matrix

def generate_hybrid_image(image_high, image_low, D0=50):
    hybrid_channels = []

    for channel_high, channel_low in zip(cv2.split(image_high), cv2.split(image_low)):
        high_freq_transform = np.fft.fft2(channel_high)
        low_freq_transform = np.fft.fft2(channel_low)

        high_freq_centered = np.fft.fftshift(high_freq_transform)
        low_freq_centered = np.fft.fftshift(low_freq_transform)

        low_pass_filter = create_gaussian_low_pass_filter(D0, image_high.shape)
        high_pass_filter = create_gaussian_high_pass_filter(D0, image_low.shape)

        low_freq_filtered = high_freq_centered * low_pass_filter
        high_freq_filtered = low_freq_centered * high_pass_filter

        low_freq_image = np.fft.ifft2(np.fft.ifftshift(low_freq_filtered))
        high_freq_image = np.fft.ifft2(np.fft.ifftshift(high_freq_filtered))

        hybrid_channel = np.abs(low_freq_image) + np.abs(high_freq_image)
        hybrid_channel_normalized = np.uint8(np.clip(hybrid_channel, 0, 255))
        hybrid_channels.append(hybrid_channel_normalized)

    hybrid_image = cv2.merge(hybrid_channels)
    return hybrid_image

def average_images(image1, image2):
    return np.uint8((image1.astype(np.float32) + image2.astype(np.float32)) / 2)

# image_high = cv2.imread('UTK_processed/83_0_0_20170120225615281.jpg.chip.jpg', cv2.IMREAD_COLOR)
# image_low = cv2.imread('UTK_processed/83_1_0_20170120230456826.jpg.chip.jpg', cv2.IMREAD_COLOR)

image_high = cv2.imread('/Users/seoli/Desktop/CS1430/CS1430/UTK_processed/18_0_4_20170103234736836.jpg.chip.jpg', cv2.IMREAD_COLOR)
image_low = cv2.imread('/Users/seoli/Desktop/CS1430/CS1430/UTK_processed/82_1_1_20170110154342872.jpg.chip.jpg', cv2.IMREAD_COLOR)

image_high_resized = cv2.resize(image_high, (128, 128))
image_low_resized = cv2.resize(image_low, (128, 128))

averaged_images = average_images(image_high_resized, image_low_resized)

hybrid_image = generate_hybrid_image(image_high_resized, image_low_resized, D0=1)

image_high_resized_rgb = cv2.cvtColor(image_high_resized, cv2.COLOR_BGR2RGB)
image_low_resized_rgb = cv2.cvtColor(image_low_resized, cv2.COLOR_BGR2RGB)
hybrid_image_rgb = cv2.cvtColor(hybrid_image, cv2.COLOR_BGR2RGB)
averaged_images_rgb = cv2.cvtColor(averaged_images, cv2.COLOR_BGR2RGB)

plot_images([image_high_resized_rgb, image_low_resized_rgb, averaged_images_rgb],
            ['High Frequency Image', 'Low Frequency Image', 'Hybrid Image'],
            rows=1, columns=3)
plt.show()