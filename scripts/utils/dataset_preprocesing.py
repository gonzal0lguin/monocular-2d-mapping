import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.cluster import KMeans

from tqdm import tqdm
import argparse
import os


def flatten_colors_kmeans(image):
    # Convert the image to the HSV color space
    image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    # Reshape the image to a 2D array of pixels
    pixels = image_hsv.reshape((-1, 3))

    # Define the number of clusters (colors)
    num_clusters = 5

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)

    # Get the cluster centers (representative colors)
    colors = kmeans.cluster_centers_.astype(np.uint8)

    # Map each pixel to the closest cluster color
    labels = kmeans.predict(pixels)
    segmented_image_hsv = colors[labels].reshape(image_hsv.shape)

    # Convert the segmented image back to the BGR color space
    segmented_image = cv.cvtColor(segmented_image_hsv, cv.COLOR_HSV2RGB)

    return segmented_image

def nearest_non_black_color(image, x, y):
    # Define a search radius for the nearest non-black pixel
    search_radius = 10

    # Get the region around the specified pixel
    region = image[max(0, x - search_radius):min(image.shape[0], x + search_radius + 1),
                   max(0, y - search_radius):min(image.shape[1], y + search_radius + 1)]

    # Find the nearest non-black pixel in the region
    non_black_pixels = region[np.any(region != [0, 0, 0], axis=-1)]
    if non_black_pixels.size == 0:
        print('No non-black pixels')
        return [0, 0, 0]
    
    color, count = np.unique(non_black_pixels, axis=0, return_counts=True)
    out_color = color[np.argmax(count)]
    # nearest_pixel = non_black_pixels[np.argmin(np.linalg.norm(non_black_pixels - image[x, y], axis=-1))]
    
    return out_color

def flatten_colors_simple(image):
    # Split the image into color channels (B, G, R)
    red, green, blue = cv.split(image)

    # Create masks based on color conditions
    red_mask = (red > 1.5 * blue) & (red > 1.5 * green)
    blue_mask = (blue > 1.5 * red) & (blue > 1.5 * green)
    green_mask = (green > 1.5 * red) & (green > 1.5 * blue)
    magenta_mask = (red > 1.5 * green) & (blue > 1.5 * green) & (abs(red - blue) < 10)
    grey_mask = (abs(red - green) < 10) & (abs(red - blue) < 10) & (abs(green - blue) < 10)

    # Create an empty image
    segmented_image = np.zeros_like(image)

    # Assign colors to the segmented regions
    segmented_image[grey_mask] = [128, 128, 128]  # Grey
    segmented_image[red_mask] = [255, 0, 0]  # Red
    segmented_image[blue_mask] = [0, 0, 255]  # Blue
    segmented_image[green_mask] = [0, 255, 0]  # Green
    segmented_image[magenta_mask] = [255, 0, 255]  # Magenta

    black_pixels = np.all(segmented_image == [0, 0, 0], axis=-1)
    for i in range(segmented_image.shape[0]):
        for j in range(segmented_image.shape[1]):
            if black_pixels[i, j]:
                segmented_image[i, j] = nearest_non_black_color(segmented_image, i, j)

    return segmented_image

def flatten_colors_threshold(image):
    # Convert the image to the HSV color space
    image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    # Define color ranges for red, green, blue, grey, and magenta
    color_ranges = {
        'red': [(0, 100, 100), (10, 255, 255)],
        'green': [(40, 40, 40), (80, 255, 255)],
        'blue': [(100, 100, 100), (140, 255, 255)],
        'grey': [(0, 0, 100), (179, 50, 200)],
        'magenta': [(140, 100, 100), (160, 255, 255)],
    }

    # Create an empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Apply color thresholding for each color range
    for color_name, (lower_bound, upper_bound) in color_ranges.items():
        lower_bound = np.array(lower_bound, dtype=np.uint8)
        upper_bound = np.array(upper_bound, dtype=np.uint8)
        color_mask = cv.inRange(image_hsv, lower_bound, upper_bound)
        mask = cv.bitwise_or(mask, color_mask)

    # Apply the mask to the original image
    segmented_image = cv.bitwise_and(image, image, mask=mask)

    return segmented_image


def test():
    imgtest = cv.imread('images/image2.png')
    imgtest = cv.cvtColor(imgtest, cv.COLOR_BGR2RGB)

    img_filt = flatten_colors_simple(imgtest)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(imgtest)
    ax[0].axis('off')

    ax[1].imshow(img_filt, vmin=0, vmax=255)
    ax[1].axis('off')

    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Image color flattener")
    parser.add_argument("-d", "--dir", type=str, default="images", help="Directory with target images")
    parser.add_argument("-o", "--out_dir", type=str, default="images_flattened", help="Directory of output images")
    parser.add_argument("-m", "--method", type=str, default="simple", help="Method of filtering [simple, kmeans, thresh]")

    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # List all files in the input folder
    image_files = [f for f in os.listdir(args.dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image
    for image_file in tqdm(image_files, desc='Processing images', unit='image'):
        # Read the image
        image_path = os.path.join(args.dir, image_file)
        img = cv.imread(image_path)

        # Apply the flattening function
        if args.method == "simple":
            segmented_img = flatten_colors_simple(img)
        elif args.method == "kmeans":
            segmented_img = flatten_colors_kmeans(img)
        elif args.method == "thresh":
            segmented_img = flatten_colors_threshold(img)
        else:
            raise ValueError(f"Method {args.method} does not exits.")
        
        # Save the result
        output_path = os.path.join(args.out_dir, image_file)
        cv.imwrite(output_path, segmented_img)
