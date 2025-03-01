import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import plot_model
from data_loader import train_generator, valid_generator
from model import get_resnet_model_with_two_outputs

# Load the model
model = get_resnet_model_with_two_outputs()

# Plot and save the model architecture
plot_model(model, to_file='model_discrete.pdf', show_shapes=True, show_layer_names=True)

# Define the learning rate schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=10000,
    decay_rate=0.96
)
optimizer_mine = Adam(learning_rate=lr_schedule)

# Specify the directory where TensorBoard logs will be stored
log_dir = "IL_model_bs32_resnet_reg_with_class"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Checkpoint to save the best weights
model_ckpt_name = "IL_model_bs32_resnet_reg_with_class_50_50"
checkpoint = ModelCheckpoint(model_ckpt_name + '_best.h5',
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')

# Early stopping with restore_best_weights enabled
patience = 10
earlystop = EarlyStopping(monitor='val_loss', 
                          min_delta=0, 
                          patience=patience, 
                          verbose=1, 
                          mode='min', 
                          restore_best_weights=True)

# Combine all callbacks
callbacks = [checkpoint, earlystop, tensorboard_callback]

# Compile the model with SparseCategoricalCrossentropy for discrete_output
model.compile(
    optimizer=optimizer_mine,
    loss={
        'continuous_output': 'mse',
        'discrete_output': SparseCategoricalCrossentropy(from_logits=False),
    },
    loss_weights={
        'continuous_output': 0.5,  # Weight for continuous output
        'discrete_output': 0.5,   # Weight for discrete output
    },
    metrics={
        'continuous_output': ['mse'],
        'discrete_output': ['accuracy'],
    }
)

# Train the model
train_hist = model.fit(
    train_generator,  
    validation_data=valid_generator,
    epochs=500, 
    shuffle=True,
    verbose=1, 
    callbacks=callbacks
)

# Save the final model
model.save('final_trained_model.h5')
