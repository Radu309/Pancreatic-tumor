import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
image_path = 'data/Pancreas_Segmentation/dataset/images/0010/0102.npy'  # Replace with the path to your .npy file
image = np.load(image_path)

# Define the size of the crop
crop_size = 400
half_crop_size = crop_size // 2

center_x, center_y = image.shape[1] // 2, image.shape[0] // 2

# Calculate the coordinates for cropping
start_x = center_x - half_crop_size
start_y = center_y - half_crop_size
end_x = start_x + crop_size
end_y = start_y + crop_size
cropped_image = image[start_y:end_y, start_x:end_x]

# Display the image using matplotlib
plt.imshow(image, cmap='gray')  # Assuming it's a grayscale image
plt.title('Image from .npy file')
plt.axis('off')  # Hide axes
plt.show()

plt.imshow(cropped_image, cmap='gray')  # Assuming it's a grayscale image
plt.title('Image from .npy file')
plt.axis('off')  # Hide axes
plt.show()

# Print out details of the image
print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Image min value: {np.min(image)}")
print(f"Image max value: {np.max(image)}")
print(f"Image mean value: {np.mean(image)}")
