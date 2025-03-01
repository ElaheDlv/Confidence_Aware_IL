from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf

def get_resnet_model_with_two_outputs():
    """
    Builds a dual-head ResNet-50-based model for continuous and discrete steering prediction.
    
    Returns:
        keras.Model: Compiled dual-head model with ResNet-50 backbone.
    """
    # Input Layer
    inputs_img = Input(shape=(160, 160, 3), name='input_image')
    
    # ResNet-50 Backbone (Pre-trained on ImageNet)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs_img, input_shape=(160, 160, 3))
    
    # Freeze initial ResNet layers (optional, for transfer learning)
    for layer in base_model.layers[:120]:
        layer.trainable = False
    for layer in base_model.layers[120:]:
        layer.trainable = True

    # Add Custom Layers for Regression
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)  # Global average pooling for reduced dimensionality
    x = Dense(256, activation='relu', name='fc1', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(rate=0.3, name='dropout1')(x)
    x = Dense(128, activation='relu', name='fc2', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(rate=0.3, name='dropout2')(x)

    # Continuous Output for Steering Value
    xc = Dense(64, activation='relu', name='fc3', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    xc = Dropout(rate=0.3, name='dropout3')(xc)
    continuous_output = Dense(1, name='continuous_output')(xc)

    # Discrete Output for Steering Classes
    num_classes = 11  # Assuming 11 classes for steering
    xd = Dense(64, activation='relu', name='fc4', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    xd = Dropout(rate=0.3, name='dropout4')(xd)
    discrete_output = Dense(num_classes, activation='softmax', name='discrete_output')(xd)

    # Define the Model with Two Outputs
    model = Model(inputs=inputs_img, outputs=[continuous_output, discrete_output])
    
    return model

if __name__ == "__main__":
    IL_model = get_resnet_model_with_two_outputs()
    IL_model.summary()
