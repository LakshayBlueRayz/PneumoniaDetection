import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def fix_model(model_path='pneumonia_model.h5'):
    # Load the existing model
    model = load_model(model_path)
    
    # Get all layers except the last one
    layers = model.layers[:-1]
    
    # Create new model
    new_model = tf.keras.Sequential()
    
    # Add all layers except last
    for layer in layers:
        new_model.add(layer)
    
    # Add new final layer with proper activation
    new_model.add(tf.keras.layers.Dense(2, activation='softmax'))
    
    # Compile model
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Copy weights from old model
    for old_layer, new_layer in zip(model.layers[:-1], new_model.layers[:-1]):
        new_layer.set_weights(old_layer.get_weights())
    
    # Save fixed model
    new_model.save('pneumonia_model_fixed.h5')
    
    # Test prediction
    test_input = np.zeros((1, 224, 224, 3))
    pred = new_model.predict(test_input)
    print("Test prediction with fixed model:", pred)
    
    return new_model

if __name__ == "__main__":
    fixed_model = fix_model()