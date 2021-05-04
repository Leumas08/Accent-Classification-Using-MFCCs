import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
from clean_data import Data

from sklearn import metrics
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.ensemble import ExtraTreesClassifier
def plot_roc(y_test, y_prob, title, n_classes = 3):
    '''
    INPUT:
    y_test, the numpy array of the true class of the test data
    y_prob, the numpy array of the predicted probabilities for each class of the test data
    title, the title of the plot
    n_class, number of the classes
    OUPUT:
    This function plot the ROC curve for multiclass classifier predictions.
    The plot includes the ROC curves for micro-average, macro-average and all the classes
    '''
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    print(y_prob)
    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_prob.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    # # Compute macro-average ROC curve and ROC area
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(n_classes):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #
    # # Finally average it and compute AUC
    # mean_tpr /= n_classes
    #
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    # # Plot all ROC curves
    # plt.figure(figsize=(8, 8))
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['r', 'g', 'b', 'y'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 15)
    plt.ylabel('True Positive Rate', fontsize = 15)
    plt.title(title, fontsize = 20)
    plt.legend(loc="lower right")
    plt.savefig('roc_'+title+' FINAL.png', bbox_inches='tight')


def report_accuracy(y_test,y_predict):
    '''
    INPUT:
    y_test, the numpy array of the true class of the test data
    y_predict, the numpy array of the predicted class of the test data
    OUPUT:
    This function prints a classification report for the model.
    '''
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_predict)
    classification_report= metrics.classification_report(y_true=y_test, y_pred=y_predict)

    print('Accuracy: {:.3f}'.format(accuracy))
    print('=========================================================')
    print('Classification Report: \n{}'.format(classification_report))


def build_model():
    length = 512
    df_train = np.load('/home/samuel/Documents/Accent/data/X_train_moz_hop.npy').reshape(-1, 16, length, 1) #pd.read_csv('../data/final.csv',index_col=0)
    df_train_label = np.load('/home/samuel/Documents/Accent/data/y_train_moz_hop.npy').reshape(df_train.shape[0],)
    df_test = np.load('/home/samuel/Documents/Accent/data/X_test_moz_hop.npy').reshape(-1, 16, length, 1) #pd.read_csv('../data/final.csv',index_col=0)
    df_test_label = np.load('/home/samuel/Documents/Accent/data/y_test_moz_hop.npy')
    #data_train = Data(df_train)
    #X_train = data_train.X
    #y_train = data_train.y
    #data_test = Data(df_test)
    #X_test = data_test.X
    #y_test = data_test.y

    x_train = []
    x_test = []
    for f in df_train:
         x_train.append(f.flatten())
    #
    for f in df_test:
        x_test.append(f.flatten())
    #
    # clf = ExtraTreesClassifier(n_estimators=50)
    # clf.fit(x_train, df_train_label)
    # print(clf.feature_importances_)
    # #model = SelectFromModel(clf, prefit = True)
    # new_x2 = []
    # print(np.mean(np.array(clf.feature_importances_)))
    # features = []
    # for i in range(0,len(clf.feature_importances_)):
    #     if clf.feature_importances_[i] > .0015:
    #         new_x2.append(i)
    #         print(clf.feature_importances_[i] )
    #         print(i)
    # sum = 0
    # i = 0
    # j = 128
    # while i < 2048:
    #     sum = 0
    #     for i in range(i,j):
    #         sum += clf.feature_importances_[i]
    #     print(sum)
    #     i += 128
    #     j += 128
    # final_x2 = []
    # for i in x_train:
    #     temp = []
    #     for j in new_x2:
    #         temp.append(i[j])
    #     final_x2.append(temp)
    # final_test = []
    # for i in x_test:
    #     temp = []
    #     for j in new_x2:
    #         temp.append(i[j])
    #     final_test.append(temp)
    # print(len(new_x2))
# >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# >>> Y = np.array([1, 1, 1, 2, 2, 2])
# >>> from sklearn.naive_bayes import GaussianNB
# >>> clf = GaussianNB()
# >>> clf.fit(X, Y)

    #x_test = final_test
    #clf = DecisionTreeClassifier()
    #clf = GaussianNB()
    #clf = RandomForestClassifier()
    clf = LDA()
    clf.fit(x_train, df_train_label)
    #a = clf.fit_transform(x_train, df_train_label)
    #plt.plot(a)
    #plt.show()
    # Predict probabilities, not classes
    print(df_test_label)
    y_prob = clf.predict_proba(x_test)
    y_predict = clf.predict(x_test)
    report_accuracy(df_test_label,y_predict)
    y_test_binarize = label_binarize(df_test_label, classes=[0, 1, 2])
    plot_roc(y_test_binarize, y_prob, title = 'LDA') # update the classifier name


if __name__ =='__main__':
    build_model()
