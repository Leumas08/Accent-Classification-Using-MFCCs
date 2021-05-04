from keras.models import load_model
import numpy as np
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import sklearn.preprocessing
import pydub
from keras.utils import to_categorical
import keras
import librosa
import librosa.display
from keras import regularizers
#from analysis import PCA_analysis
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Load in the dataset
length = 512
X_train = np.load('/home/samuel/Documents/Accent/data/X_train_moz_hop.npy').reshape(-1, 16, length, 1)
X_test2 = np.load('/home/samuel/Documents/Accent/data/X_test_moz_hop.npy').reshape(-1, 16, length, 1)
X_test = np.load('/home/samuel/Documents/Accent/data/X_val_moz_hop.npy').reshape(-1, 16, length, 1)
y_train = np.load('/home/samuel/Documents/Accent/data/y_train_moz_hop.npy')
y_test2 = np.load('/home/samuel/Documents/Accent/data/y_test_moz_hop.npy')
y_test = np.load('/home/samuel/Documents/Accent/data/y_val_moz_hop.npy')

y_train_hot = to_categorical(y_train, num_classes=3)
y_test_hot = to_categorical(y_test, num_classes=3)

list = []
for y in y_train:
    if y[0] == 0:
        list.append([1, 0, 0])
    if y[0] == 1:
        list.append([0, 1, 0])
    if y[0] == 2:
        list.append([0, 0, 1])
y_train_cat = np.array(list)

# Create Neural Network
callbacks = [TensorBoard(log_dir='./logs')]

input_shape=(16, length, 1)
num_classes=3
opt = keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9)
model = Sequential()

# First Conv layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

# Second Conv layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l1(l=0.0001)))
model.add(BatchNormalization())

# Third Conv layers
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l1(l=0.0001)))
model.add(BatchNormalization())

# FC and Soft max layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

# Train the model for only 50 epochs
history = model.fit(X_train, y_train_cat, batch_size=1024, epochs=50, verbose=1,
            validation_data=(X_test, y_test_hot), callbacks=callbacks,shuffle=True)


training_loss = history.history['loss']
test_loss = history.history['val_loss']

training_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']

# Create count of the number of epochs
epoch_count = range(1, len(training_acc) + 1)

# Visualize loss history

plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(epoch_count, training_acc, 'r--')
plt.plot(epoch_count, test_acc, 'b-')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig("test_hop.png")
plt.show();

model_name = 'final_test_model_hop_cnn_32_32_64_fc_1024_and_batchn_2_3_50epochs.h5'

model.save(model_name)
print(model.summary())

list = []
for y in y_test2:
    if y[0] == 0:
        list.append([1, 0, 0])
    if y[0] == 1:
        list.append([0, 1, 0])
    if y[0] == 2:
        list.append([0, 0, 1])
y_test_cat = np.array(list)

# Plot Results

# ROC
lw = 2
callbacks = [TensorBoard(log_dir='./logs')]

input_shape=(16, length, 1)
num_classes=3
model = keras.models.load_model(model_name)

predict = model.predict(X_test2)

# Plot linewidth.
lw = 2
n_classes = num_classes
y_test = y_test_cat
y_score = predict
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
lables = ["US", "UK", "IN"]
for i, color, l in zip(range(n_classes), colors, lables):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class %s (area = {1:0.2f})'
             ''.format(l, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC all Class Accuracy')
plt.legend(loc="lower right")
plt.show()
