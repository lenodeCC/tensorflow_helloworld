import numpy as np
import tensorflow as tf
layers = tf.keras.layers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

model = tf.keras.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.fit(x_test, y_test, validation_split=0.25,
          epochs=10, batch_size=64)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)
i=102
predictions = model.predict(np.array([x_test[i]]))
plt.imshow(x_test[i],cmap="Greys")
plt.show()
print(y_test[i])