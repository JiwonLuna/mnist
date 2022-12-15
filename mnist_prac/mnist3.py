import tensorflow as tf
import matplotlib.pyplot as plt

Mnist = tf.keras.datasets.mnist
(x_train, t_train), (x_test, t_test) = Mnist.load_data()

for i in range(9):
    plt.subplot(3,3, i+1)
    plt.tight_layout()
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray", interpolation= "none")
    plt.title("digit: {}".format(t_train[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()

x_train.astype('float32')
x_test.astype('float32')
x_train, x_test = x_train/255, x_test/255
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
print(x_train.shape)
print(t_train.shape)

num_category = 10
t_train = tf.keras.utils.to_categorical(t_train, num_category)
t_test = tf.keras.utils.to_categorical(t_test, num_category)
print(t_train[0])

# CNN Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same", input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Conv2D(128,3,activation="relu", padding = "same"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_category, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 128
num_epoch = 10
History = model.fit(x_train, t_train, batch_size = batch_size,
    epochs=num_epoch, validation_data=(x_test, t_test))

score = model.evaluate(x_test, t_test, verbose = 0)
print("Test loss = ", score[0])
print("Test accuracy = ", score[1])

plt.plot(History.history['accuracy'], label = "train")
plt.plot(History.history['val_accuracy'], label = "test")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(History.history['loss'], label="train")
plt.plot(History.history['val_loss'], label="test")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()