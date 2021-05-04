from IPython.display import Image, display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import ExtraTreesClassifier

def PCA_analysis(X_train,y_train):
    #MB_matrix = np.zeros((X_train[0,:,:,:].size, X_train.shape[0]))
    new_x = []
    for i in range(X_train.shape[0]):
        new_x.append(X_train[i].flatten())
        #MB_array = X_train[i,:,:,:].flatten()  # covert 2d to 1d array
        #MB_arrayStd = (MB_array - MB_array.mean())/MB_array.std()
        #MB_matrix[:,i] = MB_arrayStd
    #print(MB_matrix.shape)

    x = StandardScaler().fit_transform(new_x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1','principal component 2'])
    target_labels = pd.DataFrame(data =y_train, columns = ['targets'])
    finalDf = pd.concat([principalDf, target_labels], axis = 1)
    #print(finalDf)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title("PCA", fontsize = 20)

    targets = [2, 1, 0]
    colors = ['b', 'g', 'r']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['targets'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(["UK","US", "IN"])
    ax.grid()
    save="PCA_FINAL.png"
    ax.figure.savefig(save)

#PCA_analysis(np.load('X_train_moz3.npy').reshape(-1, 16, 128, 1), np.load('y_train_moz3.npy'))
def PCA3_analysis(X_train,y_train):
    #MB_matrix = np.zeros((X_train[0,:,:,:].size, X_train.shape[0]))
    new_x = []
    for i in range(X_train.shape[0]):
        new_x.append(X_train[i].flatten())
        #MB_array = X_train[i,:,:,:].flatten()  # covert 2d to 1d array
        #MB_arrayStd = (MB_array - MB_array.mean())/MB_array.std()
        #MB_matrix[:,i] = MB_arrayStd
    #print(MB_matrix.shape)

    x = StandardScaler().fit_transform(new_x)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1','principal component 2','principal component 3'])
    target_labels = pd.DataFrame(data =y_train, columns = ['targets'])
    finalDf = pd.concat([principalDf, target_labels], axis = 1)
    #print(finalDf)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    #ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title("PCA", fontsize = 20)

    targets = [2, 1, 0]
    colors = ['b', 'g', 'r']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['targets'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , finalDf.loc[indicesToKeep, 'principal component 3']
                   , c = color
                   , s = 50)
    ax.legend(["UK","US", "IN"])
    ax.grid()
    save="PCA_FINAL.png"
    ax.figure.savefig(save)

def extra_trees_classifier(X_train, y):
    y = y.ravel()
    new_x = []
    for i in range(X_train.shape[0]):
        new_x.append(X_train[i].flatten())
    clf = ExtraTreesClassifier(n_estimators=50)
    clf.fit(new_x, y)
    print(clf.feature_importances_)
    #model = SelectFromModel(clf, prefit = True)
    new_x2 = []
    print(np.mean(np.array(clf.feature_importances_)))
    features = []
    for i in range(0,len(clf.feature_importances_)):
        if clf.feature_importances_[i] > .001:
            new_x2.append(i)
    final_x2 = []
    for i in new_x:
        temp = []
        for j in new_x2:
            temp.append(i[j])
        final_x2.append(temp)
    print(len(new_x2))



    x = StandardScaler().fit_transform(final_x2)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1','principal component 2'])
    target_labels = pd.DataFrame(data =y, columns = ['targets'])
    finalDf = pd.concat([principalDf, target_labels], axis = 1)
    #X_new = model.transform(new_x)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    #ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title("PCA", fontsize = 20)

    targets = [2, 1, 0]
    colors = ['b', 'g', 'r']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['targets'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.grid()
    ax.legend(["UK","US", "IN"])
    save="ETC_FINAL.png"
    ax.figure.savefig(save)
    #X_new  = SelectKBest(chi2, k=2).fit_transform(new_x, y_train)
    #lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(new_x, y)
    #model = SelectFromModel(lsvc, prefit=True)
    #X_new = model.transform(new_x)
extra_trees_classifier(np.load('/home/samuel/Documents/Accent/data/X_train_moz_hop.npy').reshape(-1, 16, 512), np.load('/home/samuel/Documents/Accent/data/y_train_moz_hop.npy'))
