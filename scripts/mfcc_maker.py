import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

x = np.load('X_train_moz.npy').reshape(-1, 16, 128)
y = np.load('y_train_moz.npy')
us = []
uk = []
ind = []
count = 0
for i in x:
    if y[count] == 0:
        us.append(i)
    if y[count] ==1:
        uk.append(i)
    if y[count] ==2:
        ind.append(i)
    count += 1

us = np.array(us)
us_av = np.average(us, axis = 0)
fig, ax = plt.subplots()

img = librosa.display.specshow(us_av[3:16], x_axis='time', ax=ax)

fig.colorbar(img, ax=ax)

ax.set(title='MFCC for US Accent')
fig.savefig("MFCC_US.jpg")

uk = np.array(uk)
uk_av = np.average(uk, axis = 0)
fig, ax = plt.subplots()

img = librosa.display.specshow(uk_av[3:16], x_axis='time', ax=ax)

fig.colorbar(img, ax=ax)

ax.set(title='MFCC for UK Accent')
fig.savefig("MFCC_UK.jpg")

ind = np.array(ind)
ind_av = np.average(ind, axis = 0)
fig, ax = plt.subplots()

img = librosa.display.specshow(ind_av[3:16], x_axis='time', ax=ax)

fig.colorbar(img, ax=ax)

ax.set(title='MFCC for Indian Accent')
fig.savefig("MFCC_IN.jpg")
