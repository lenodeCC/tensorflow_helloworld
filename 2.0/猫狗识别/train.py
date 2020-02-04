import tensorflow as tf
import preprocess

layers = tf.keras.layers

model = tf.keras.models.Sequential([
    # 如果训练慢可以把数据设置的小一些
    layers.Conv2D(32, (3, 3), activation="relu",
                  input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),  # 最大池化

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),

    # 全链接层
    layers.Flatten(),
    layers.Dense(512, activation="relu"),

    # 二分类用Sigmoid
    layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              metrics=[tf.keras.metrics.BinaryAccuracy()]
              )


history = model.fit_generator(
    preprocess.trainGenerator,
    steps_per_epoch=100,  # 2000 images = batch_size * steps
    epochs=20,
    validation_data=preprocess.validationGenerator,
    validation_steps=50,  # 1000 images = batch_size * steps
    verbose=2)
