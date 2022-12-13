import tensorflow as tf
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras import models
from keras.layers import Dense, Input, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pylab as plt
plt.style.use('seaborn-white')

# Load DataSet
tf.random.set_seed(111)

(x_train_full, y_train_full), (x_test, y_test) = load_data(path='mnis.npz')

x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size = 0.3, random_state = 111)

# Data check
num_x_train = (x_train.shape[0])
num_x_val = (x_val.shape[0])
num_x_test = (x_test.shape[0])

print("learning data : {}\t label : {}".format(x_train_full.shape,
y_train_full.shape))
print("learning data : {}\t label : {}".format(x_train.shape, y_train.shape))
print("val data : {}\t label : {}".format(x_val.shape, y_val.shape))
print("test data : {}\t label : {}".format(x_test.shape, y_test.shape))

num_sample = 5

random_idxs = np.random.randint(6000, size = num_sample)

plt.figure(figsize = (14, 8))
for i, idx in enumerate(random_idxs):
    img = x_train_full[idx, :]
    label = y_train_full[idx]

    plt.subplot(1, len(random_idxs), i+1)
    plt.imshow(img)
    plt.title("Index: {}, Label : {}".format(idx, label))

plt.grid()
plt.show()

# Data preprocessing
x_train = x_train / 255.
x_val = x_val / 255.
x_test = x_test / 255.

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# Model Construction
model = Sequential([Input(shape = (28, 28), name = 'input'),
    Flatten(input_shape = [28, 28], name = 'flatten'),
    Dense(100, activation= 'relu', name = 'dense1'),
    Dense(64, activation = 'relu', name = 'dense2'),
    Dense(32, activation = "relu", name = 'dense3'),
    Dense(10, activation = 'softmax', name = 'output')])

model.summary()

# Model compile
model.compile(loss = 'categorical_crossentropy',
    optimizer = 'sgd', metrics = ['accuracy'])

# Model learning
history = model.fit(x_train, y_train, epochs = 50, batch_size = 128,
    validation_data = (x_val, y_val))

# Learning result visualization
history.history.keys()

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)
fig = plt.figure(figsize = (12, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(epochs, loss, color = 'blue', label = 'train_loss')
ax1.plot(epochs, val_loss, color = 'red', label = 'val_loss')
ax1.set_title('Train and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid()
ax1.legend()

accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(epochs, accuracy, color = 'blue', label = 'train_accuracy')
ax2.plot(epochs, val_accuracy, color = 'red', label = 'val_accuracy')
ax2.set_title('Train and Validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.grid()
ax2.legend()

plt.show()

# model estimate(1)
model.evaluate(x_test, y_test)

# Prediction using learned model
pred_ys = model.predict(x_test)
print(pred_ys.shape)

np.set_printoptions(precision = 7)
print(pred_ys[0])

arg_pred_y = np.argmax(pred_ys, axis = 1)

plt.imshow(x_test[0])
plt.title("predicted label: {}".format(arg_pred_y[0]))
plt.show()

# model estimate(2)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
sns.set(style = 'white')

plt.figure(figsize = (8, 8))
cm = confusion_matrix(np.argmax(y_test, axis = -1), np.argmax(pred_ys, axis = -1))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.xlabel('Predicted Label')
plt.ylabel('TrueLabel')
plt.show() 

# model estimate(3)
print(classification_report(np.argmax(y_test, axis = -1),
    np.argmax(pred_ys, axis = -1)))