import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from PIL import Image
from sklearn.model_selection import train_test_split

def flipping(image, steer):
    """Flip the image and invert the steering angle."""
    flipped_image = cv2.flip(image, 1)
    flipped_steer = -steer
    return flipped_image, flipped_steer

def brightness(image, steer):
    """Randomly adjust brightness of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + (np.random.rand() - 0.5) * 0.4
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
    bright_image = cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_HSV2RGB)
    return bright_image, steer

class DataGenerator(Sequence):
    """
    Custom data generator for loading and preprocessing driving data.
    """
    def __init__(self, image_filenames, labels, class_labels, batch_size, augment, crop_coords=None):
        self.image_filenames = image_filenames
        self.labels = labels  # Continuous steering values
        self.class_labels = class_labels  # Discretized class labels
        self.batch_size = batch_size
        self.augment = augment
        self.crop_coords = crop_coords
    
    def __len__(self):
        return (np.floor(len(self.image_filenames) / float(self.batch_size))).astype(np.int32)
  
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y_continuous = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y_classes = self.class_labels[idx * self.batch_size : (idx+1) * self.batch_size]
        
        inputs, outputs_continuous, outputs_classes = [], [], []
        for filename, continuous, class_label in zip(batch_x, batch_y_continuous, batch_y_classes):
            try:
                img = Image.open(filename).convert("RGB")
                if self.crop_coords:
                    left, top, right, bottom = self.crop_coords
                else:
                    left, top, right, bottom = 0, 218, 448, 448
                img = img.crop((left, top, right, bottom))
                img = np.array(img)
                img_resized = cv2.resize(img, (160, 160))
                inputs.append(img_resized)
                outputs_continuous.append(continuous)
                outputs_classes.append(class_label)
                
                if 'Flip' in self.augment:
                    aug_image, aug_steer = flipping(img_resized, continuous)
                    inputs.append(aug_image)
                    outputs_continuous.append(aug_steer)
                    outputs_classes.append(class_label)
                
                if 'bright' in self.augment:
                    aug_image, aug_steer = brightness(img_resized, continuous)
                    inputs.append(aug_image)
                    outputs_continuous.append(aug_steer)
                    outputs_classes.append(class_label)
            except Exception as e:
                print(f"Skipping file {filename} due to error: {e}")
        
        inputs_final = np.array(inputs).astype("float32") / 255.0
        outputs_continuous = np.array(outputs_continuous).astype("float32")
        outputs_classes = np.array(outputs_classes).astype("int32")
        
        return inputs_final, [outputs_continuous, outputs_classes]

if __name__ == "__main__":
    labels_path = "your_labels.csv"  # Update with actual path
    images_directory = "your_image_folder"  # Update with actual path
    df = pd.read_csv(labels_path)
    
    labels = []
    image_paths = []
    steering_labels = []
    
    for index, row in df.iterrows():
        labels.append(row['Steering'])
        steering_labels.append(row['Steering_Class'])
        image_paths.append(os.path.join(images_directory, row['Image_Fname']))  # Ensure correct path
    
    batch_size = 32
    # Combine labels and class labels for consistent splitting
    combined_labels = list(zip(labels, steering_labels))
    X_train_data, X_valid_data, combined_train_labels, combined_valid_labels = train_test_split(
        image_paths, combined_labels, test_size=0.1, shuffle=True, random_state=42)
    
    # Separate continuous and class labels after the split
    y_train_continuous, y_train_classes = zip(*combined_train_labels)
    y_valid_continuous, y_valid_classes = zip(*combined_valid_labels)
    
    train_generator = DataGenerator(X_train_data, y_train_continuous, y_train_classes, batch_size, augment=["Flip", "no"])
    valid_generator = DataGenerator(X_valid_data, y_valid_continuous, y_valid_classes, batch_size, augment=["Flip", "no"])
