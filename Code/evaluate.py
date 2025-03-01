import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import pandas as pd
from data_loader import DataGenerator
import os

# Load the trained model
model_path = "final_trained_model.h5"
model = tf.keras.models.load_model(model_path)

# Load test dataset
labels_path = "your_test_labels.csv"  # Update with actual test CSV path
images_directory = "your_test_image_folder"  # Update with actual path
df_test = pd.read_csv(labels_path)

test_images = [os.path.join(images_directory, fname) for fname in df_test['Image_Fname'].values]
test_continuous = df_test['Steering'].values
test_classes = df_test['Steering_Class'].values

# Create test data generator
test_generator = DataGenerator(test_images, test_continuous, test_classes, batch_size=32, augment=[])

# Compute predictions and generate confusion matrix
def compute_confusion_matrix(generator):
    predicted_values = []
    ground_truth_values = []
    predicted_labels_value = []
    
    for i in range(len(generator)):
        X_valid, y_valid = generator[i]
        continuous_output, discrete_output = model.predict(X_valid)
        y_valid_continuous, y_valid_discrete = y_valid
        predicted_labels = np.argmax(discrete_output, axis=1)
        
        predicted_values.extend(discrete_output.flatten())
        ground_truth_values.extend(y_valid_discrete.flatten())
        predicted_labels_value.extend(predicted_labels.flatten())
    
    ground_truth_values = np.array(ground_truth_values)
    predicted_labels_value = np.array(predicted_labels_value)
    
    # Compute accuracy
    accuracy = (np.sum(ground_truth_values == predicted_labels_value) / len(ground_truth_values)) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(ground_truth_values, predicted_labels_value)
    
    # Compute F1 Score
    f1 = f1_score(ground_truth_values, predicted_labels_value, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}")
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(ground_truth_values, predicted_labels_value))
    
    # Plot standard confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Normalized Confusion Matrix")
    plt.savefig("normalized_confusion_matrix.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
if __name__ == "__main__":
    # Compute confusion matrix for the test set
    compute_confusion_matrix(test_generator)