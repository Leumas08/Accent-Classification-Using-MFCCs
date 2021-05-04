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
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import itertools
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from scipy import interp
from itertools import cycle
from sklearn.metrics import confusion_matrix

# This is an evalutation script for the network trained by big_cnn_3.py

# Load in the datasets
length = 512
X_train = np.load('/home/samuel/Documents/Accent/data/X_train_moz_hop.npy').reshape(-1, 16, length, 1)
X_test2 = np.load('/home/samuel/Documents/Accent/data/X_test_moz_hop.npy').reshape(-1, 16, length, 1)
X_test = np.load('/home/samuel/Documents/Accent/data/X_val_moz_hop.npy').reshape(-1, 16, length, 1)

y_train = np.load('/home/samuel/Documents/Accent/data/y_train_moz_hop.npy')
y_test2 = np.load('/home/samuel/Documents/Accent/data/y_test_moz_hop.npy')
y_test = np.load('/home/samuel/Documents/Accent/data/y_val_moz_hop.npy')

y_train_hot = to_categorical(y_train, num_classes=3)
y_test_hot = to_categorical(y_test, num_classes=3)
y_test_hot2 = to_categorical(y_test2, num_classes=3)

#y_val_hot = to_categorical(y_val, num_classes=2)
#print(X_train.shape[0])
list = []
for y in y_test2:
    if y[0] == 0:
        list.append([1, 0, 0])
    if y[0] == 1:
        list.append([0, 1, 0])
    if y[0] == 2:
        list.append([0, 0, 1])
y_test_cat = np.array(list)



#PCA_analysis(X_train,y_train)

num_classes=3

# Load in the model
model = keras.models.load_model('final_test_model_hop_cnn_32_32_64_fc_1024_and_batchn_2_3_50epochs.h5')

predict = model.predict(X_test2)

# Plot ROC
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

# Compute macro-average ROC curve and ROC area

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
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(l, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC all Class Accuracy')
plt.legend(loc="lower right")
plt.show()

# Plot Confusion Matrix
predict = model.predict_classes(X_test2)
print(predict)
print(y_test2.shape)
print(predict.shape)
cnf_matrix = confusion_matrix(y_test2, predict,labels=[0, 1, 2])
np.set_printoptions(precision=2)

# Adapted function from https://gist.github.com/daa233/e6f237a70a3586904c615334a1fea27c#file-plot_confusion_matrix-py
# Original Author: daa233
# Source: GitHub
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2],
                      title='Confusion matrix, without normalization')
