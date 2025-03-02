import os
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import plot_model
from data_loader import DataGenerator
from model import get_resnet_model_with_two_outputs
import pandas as pd


def main(args):
    # Load Dataset
    df = pd.read_csv(args.data_csv)
    image_paths = [os.path.join(args.image_dir, fname) for fname in df['Image_Fname']]
    labels = df['Steering'].tolist()
    steering_labels = df['Steering_Class'].tolist()
    
    # Split Dataset
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, list(zip(labels, steering_labels)), 
                                                          test_size=0.1, shuffle=True, random_state=42)
    y_train_cont, y_train_class = zip(*y_train)
    y_valid_cont, y_valid_class = zip(*y_valid)

    # Data Generators
    train_generator = DataGenerator(X_train, y_train_cont, y_train_class, args.batch_size, augment=['Flip', 'bright'])
    valid_generator = DataGenerator(X_valid, y_valid_cont, y_valid_class, args.batch_size, augment=['Flip', 'bright'])
    
    # Load Model
    model = get_resnet_model_with_two_outputs()
    
    # Plot Model Structure
    plot_model(model, to_file='model_architecture.pdf', show_shapes=True, show_layer_names=True)
    
    # Learning Rate Schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=args.lr,
        decay_steps=10000,
        decay_rate=0.96
    )
    optimizer = Adam(learning_rate=lr_schedule)
    
    # Callbacks
    log_dir = os.path.join("logs", args.experiment_name)
    checkpoint_path = os.path.join("checkpoints", f"{args.experiment_name}_best.h5")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    
    # Compile Model
    model.compile(
        optimizer=optimizer,
        loss={
            'continuous_output': 'mse',
            'discrete_output': SparseCategoricalCrossentropy(from_logits=False),
        },
        loss_weights={'continuous_output': 0.5, 'discrete_output': 0.5},
        metrics={'continuous_output': ['mse'], 'discrete_output': ['accuracy']}
    )
    
    # Train Model
    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=args.epochs,
        shuffle=True,
        verbose=1,
        callbacks=callbacks
    )
    
    # Save Final Model
    final_model_path = os.path.join("pretrained_models", f"{args.experiment_name}_final.h5")
    os.makedirs("pretrained_models", exist_ok=True)
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Confidence-Aware Imitation Learning Model")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--experiment_name", type=str, default="IL_experiment", help="Name of the experiment for logging.")
    
    args = parser.parse_args()
    main(args)
