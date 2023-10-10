# do not forget to code the mnist relu activation model here!!!!!! 09/10/23

import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalize the data
x_train_norm = x_train/255.0
x_test_norm = x_test/255.0

# create the model
model = tf.keras.Sequential([
    # input layer
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # hidden layers
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    # output layer
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile the model
model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# fit the model
model.fit(
    x_train_norm,
    y_train,
    epochs=40,
    validation_data=(
        x_test_norm,
        y_test
    )
)
