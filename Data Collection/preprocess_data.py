import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from PIL import Image

# Function to normalize steering values by binning and balancing
def normalize_steering_values(data, column_name='Steering', num_bins=20, max_count_per_bin=20000):
    bin_edges = np.linspace(-1, 1, num_bins + 1)
    normalized_df = pd.DataFrame()
    
    for i in range(len(bin_edges) - 1):
        bin_mask = (data[column_name] >= bin_edges[i]) & (data[column_name] < bin_edges[i + 1])
        bin_data = data[bin_mask]
        
        if len(bin_data) > max_count_per_bin:
            bin_data = bin_data.sample(n=max_count_per_bin, random_state=42)
        
        normalized_df = pd.concat([normalized_df, bin_data])
    
    return normalized_df

# Function to filter steering values within a given range
def filter_steering_values(data, column_name='Steering', min_value=-0.25, max_value=0.25):
    return data[(data[column_name] >= min_value) & (data[column_name] <= max_value)]

# Function to scale steering values back to the range [-1, 1]
def scale_steering_values(data, column_name='Steering', scale_factor=4):
    data[column_name] *= scale_factor
    data[column_name] = np.clip(data[column_name], -1, 1)  # Ensure values remain in range [-1, 1]
    return data

# Function to assign discrete class labels based on steering values
def assign_steering_classes(data, column_name='Steering', num_classes=11):
    bin_width = 2 / num_classes
    half_width = bin_width / 2
    custom_intervals = np.linspace(-1, 1, num_classes + 1)
    middle_index = num_classes // 2
    custom_intervals[middle_index] = -half_width
    custom_intervals[middle_index + 1] = half_width
    custom_labels = list(range(len(custom_intervals) - 1))
    
    data['Steering_Class'] = pd.cut(
        data[column_name], bins=custom_intervals, labels=custom_labels, include_lowest=True
    )
    return data

# Function to perform data augmentation by flipping images and reversing steering angles
def augment_data(data, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    augmented_data = []
    
    for idx, row in data.iterrows():
        original_path = row['Image_Fname']
        steering_angle = row['Steering']
        
        img_path = os.path.join(input_dir, original_path)
        img = Image.open(img_path)


        # Save the original (unflipped) image in the new directory
        unflipped_filename = f"{original_path[:-4]}_unflipped.png"
        unflipped_path = os.path.join(output_dir, unflipped_filename)
        img.save(unflipped_path)
        
        unflipped_row = row.copy()
        unflipped_row['Image_Fname'] = unflipped_filename
        augmented_data.append(unflipped_row)

        # Flip the image horizontally
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Save the flipped image in the new directory
        flipped_filename = f"{original_path[:-4]}_flipped.png"
        flipped_path = os.path.join(output_dir, flipped_filename)
        flipped_img.save(flipped_path)
        
        # Reverse the steering angle (positive to negative and vice versa)
        flipped_row = row.copy()
        flipped_row['Image_Fname'] = flipped_filename
        flipped_row['Steering'] = -steering_angle
        augmented_data.append(flipped_row)
    
    return pd.DataFrame(augmented_data)

# Function to plot and save steering distributions
def plot_steering_distribution(original_data, processed_data, column_name, output_dir, title):
    #os.makedirs(output_dir, exist_ok=True)
    save_dir = os.path.abspath(os.path.join(output_dir, os.pardir))  # Move one level up from output_dir
    os.makedirs(save_dir, exist_ok=True)

    
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.hist(processed_data[column_name], bins=20, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    
    plot_path = os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"{title} plot saved at {plot_path}")

# Main function
def preprocess_data(input_csv, output_csv, output_dir, input_dir):
    data = pd.read_csv(input_csv)
    
    # Normalize steering values
    normalized_data = normalize_steering_values(data)
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply filtering
    filtered_data = filter_steering_values(normalized_data)
    
    # Apply augmentation
    augmented_data = augment_data(filtered_data, input_dir, output_dir)
    
    # Apply scaling
    scaled_data = scale_steering_values(augmented_data)
    
    # Assign classification labels
    processed_data = assign_steering_classes(scaled_data)
    
    # Save processed dataset
    processed_data.to_csv(output_csv, index=False)
    print(f"Processed dataset saved at {output_csv}")
    
    # Plot distributions
    plot_steering_distribution(data, processed_data, 'Steering', output_dir, "Processed Steering Data")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CARLA driving dataset.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save processed CSV file.")
    parser.add_argument("--output_dir", type=str, default="preprocessed_data", help="Directory to save plots and processed data.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    args = parser.parse_args()
    
    preprocess_data(args.input_csv, args.output_csv, args.output_dir, args.input_dir)
