import tensorflow as tf

from config import config

def CNN(output_shape: int = 10, scale: int = 16, compounding_dropout: float = 0, gaussian_noise: float = 0,
dropout: float = 0, weight_decay : float = 1e-4, batch_momentum : float = 0.9) -> tf.keras.Model:
    """
    A function that creates a convolutional neural network model.

    Parameters:
    output_shape (int): The number of output classes.
    scale (int): A scaling factor for the number of filters in the convolutional layers.
    compounding_dropout (float): A dropout rate to apply on the compounding layers.
    gaussian_noise (float): The standard deviation of the gaussian noise to add to the input.
    dropout (float): The dropout rate to apply on the layers.
    weight_decay (float): The weight decay (L2 regularization) applied to the model's weights.
    momentum (float): The momentum for the batch normalization layers.

    Returns:
    tf.keras.Model: The created convolutional neural network model.
    """
    inputs = tf.keras.layers.Input(shape=(config['image']['height'], config['image']['width'], 1))
    x = inputs
    if gaussian_noise != 0:
        # Add gaussian noise to input as augmentation
        x = tf.keras.layers.GaussianNoise(gaussian_noise)(x)
    
    # num_conv_layers CNN with increasing filter depth
    for i in range(config['model']["num_conv_layers"]):
        res_input = x
        x = tf.keras.layers.Conv2D((2**i) * scale, (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=batch_momentum)(x)

        x = tf.keras.layers.Conv2D((2**i) * scale, (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=batch_momentum)(x)

        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        # Average pooling with stride 2 and convolution with same feature depth to match residual connection with x
        res_input = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(res_input)
        res_input = tf.keras.layers.Conv2D((2**i) * scale, (1, 1), kernel_initializer='he_normal')(res_input)
        x = tf.keras.layers.add([res_input, x])  # Adding the residual connection

    x = tf.keras.layers.Flatten()(x)
    # Dense layer
    x = tf.keras.layers.Dense(4*scale, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(momentum=batch_momentum)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # Output layer
    outputs = tf.keras.layers.Dense(output_shape, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model



def compile_model(model: tf.keras.Model, lr: float = 3e-4):
    """
    This function is used to compile the given model with the specified learning rate, loss function and metrics.

    Parameters:
    model (tf.keras.Model): The model to be compiled
    lr (float): The learning rate for the optimizer. Default value is 3e-4.

    Returns:
    None
    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,
                               decay_steps=config['training']['ExponentialDecay']['decay_steps'],
                               decay_rate=config['training']['ExponentialDecay']['decay_rate'])
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=config['training']['Adam']['epsilon'], 
                                   beta_1 = config['training']['Adam']['beta_1'], 
                                   beta_2=config['training']['Adam']['beta_2']) 
    model.compile(optimizer=opt, loss=config['training']['loss'],
                    metrics=[config['training']["metrics"]])
    model.summary()
