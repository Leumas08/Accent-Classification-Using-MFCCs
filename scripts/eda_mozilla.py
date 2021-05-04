import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import sklearn.preprocessing
import pydub
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import IPython.display as ipd
def clean_df(file):
    df = pd.read_csv(file, sep='\t')
    df_us = df[df['accent']=='us'].sample(12000)
    df_ind = df[df['accent']=='indian'].sample(12000)
    df_uk = df[df['accent']=='england'].sample(12000)
    df = df_us.append(df_uk)
    print(len(df))
    df = df.append(df_ind)
    print(len(df))
    # df = df.append(df_uk)
    df.drop(['client_id', 'sentence', 'up_votes', 'down_votes', 'age', 'gender'],
        axis=1, inplace=True)

    # print("Indian:"+str(len(df_ind)))
    # print("UK: "+str(len(df_uk)))
    # print("US:"+str(len(df_us)))
    return df

class Mfcc():

    def __init__(self, df, col):
        self.df = df
        self.col = col

    def mp3towav(self):
        us = self.df[self.df['accent']=='us']
        for filename in tqdm(us[self.col]):
            filename = filename.split(".mp3")[0]
            if os.path.exists("../data/mozilla_voice/clips/{}.mp3".format(filename)):
                if os.path.exists("../data/mozilla_voice/wav/{}.wav".format(filename)):
                    a = 1
                    #print("Already made: " + str(filename))
                else:
                    pydub.AudioSegment.from_mp3("../data/mozilla_voice/clips/{}.mp3".format(filename)).export("../data/mozilla_voice/wav/{}.wav".format(filename), format="wav")
        uk = self.df[self.df['accent']=='england']
        for filename in tqdm(uk[self.col]):
            filename = filename.split(".mp3")[0]
            if os.path.exists("../data/mozilla_voice/clips/{}.mp3".format(filename)):
                if os.path.exists("../data/mozilla_voice/wav/{}.wav".format(filename)):
                    a = 1
                    #print("Already made: " + str(filename))
                else:
                    pydub.AudioSegment.from_mp3("../data/mozilla_voice/clips/{}.mp3".format(filename)).export("../data/mozilla_voice/wav/{}.wav".format(filename), format="wav")
        ind = self.df[self.df['accent']=='indian']
        for filename in tqdm(ind[self.col]):
            filename = filename.split(".mp3")[0]
            if os.path.exists("../data/mozilla_voice/clips/{}.mp3".format(filename)):
                if os.path.exists("../data/mozilla_voice/wav/{}.wav".format(filename)):
                    a = 1
                    #print("Already made: " + str(filename))
                else:
                    pydub.AudioSegment.from_mp3("../data/mozilla_voice/clips/{}.mp3".format(filename)).export("../data/mozilla_voice/wav/{}.wav".format(filename), format="wav")
    # def mp3towav(self):
        # uk = self.df[self.df['accent']=='england']
        # for filename in tqdm(uk[self.col]):
        #     if os.path.exists("../data/clips/{}.mp3".format(filename)):
        #         pydub.AudioSegment.from_mp3("../data/clips/{}.mp3".format(filename)).export("../data/wav/{}.wav".format(filename), format="wav")
        #         os.remove("../data/clips/{}.mp3".format(filename))

        # engl = self.df[self.df['accent']=='england']
        # for filename in tqdm(engl[self.col]):
        #     if os.pat The energy can be of different magnitude and therefore produce quite different picture due to the different colour scaleh.exists("/home/kelsey/Desktop/accent-classification/data/en/clips/{}".format(filename)):
        #         pydub.AudioSegment.from_mp3("/home/kelsey/Desktop/accent-classification/data/en/clips/{}".format(filename)).export("/home/kelsey/Desktop/accent-classification/data/en/engl_2/{}.wav".format(filename[:-4]), format="wav")
        #         os.remove("/home/kelsey/Desktop/accent-classification/data/en/clips/{}".format(filename))
        #     else:
        #         print("DNE")
        # # us = self.df[self.df['accent']=='england']
        # # for filename in tqdm(us[self.col]):
        # #     if os.path.exists("../data/clips/{}.mp3".format(filename)):
        # #         pydub.AudioSegment.from_mp3("../data/clips/{}.mp3".format(filename)).export("../data/wav/{}.wav".format(filename), format="wav")
        # #         os.remove("../data/clips/{}.mp3".format(filename))
        # exit()
    def wavtomfcc(self, file_path):
        #print("TYPE:"+acc)
        wave, sr = librosa.load(file_path, mono=True)
        ipd.Audio(wave,rate=sr)
        mfcc = librosa.feature.mfcc(wave, sr=sr,hop_length=128, n_mfcc=13)
        return mfcc

    def create_mfcc(self):
        list_of_mfccs = []
        y=[]
        class_us=[]
        class_uk=[]
        class_in = []

        #uk = self.df[self.df['accent']=='england']
        #print(df['path'].tolist())
        for wav in tqdm(df['path'].tolist()):
            wav = wav.split(".mp3")[0]
            #print(wav)
            file_name = '../data/mozilla_voice/wav/'+wav+".wav"
            if os.path.exists(file_name):
                acc = self.df[self.df['path'] == wav+".mp3"]['accent'].item()
                #print(acc)
                if acc == 'us':
                    y.append(0)
                    class_us.append(0)
                    mfcc = self.wavtomfcc(file_name)
                    list_of_mfccs.append(mfcc)
                if acc == 'england':
                    y.append(1)
                    class_uk.append(0)
                    mfcc = self.wavtomfcc(file_name)
                    list_of_mfccs.append(mfcc)
                if acc == 'indian':
                    y.append(2)
                    class_in.append(0)
                    mfcc = self.wavtomfcc(file_name)
                    list_of_mfccs.append(mfcc)
                # if acc == 'indian':
                #     y.append(0)
                #     mfcc = self.wavtomfcc(file_name)
                #     list_of_mfccs.append(mfcc)
        print("US Class:"+str(len(class_us)))
        print("UK Class:"+str(len(class_uk)))
        print("IN Class:"+str(len(class_in)))
        self.y = np.asarray(y)
        print(len(self.y))
        self.list_of_mfccs = list_of_mfccs
        print(len(self.list_of_mfccs))

    def resize_mfcc(self):
        self.target_size = 512
        resized_mfcc = [librosa.util.fix_length(mfcc, self.target_size, axis=1)
                         for mfcc in self.list_of_mfccs]
        resized_mfcc = [np.vstack((np.zeros((3, self.target_size)), mfcc)) for mfcc in resized_mfcc]
        self.X = resized_mfcc

    # def label_samples(self):
    #     uk = self.df[self.df['accent']=='england']
    #     y_labels = np.array(uk['accent'])
    #     #print(len(y_labels))
    #     y = np.where(y_labels=='england', 2, 0)
    #     self.y = y

    def split_data(self):
        # print((self.X).shape)
        # print((self.y).shape)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify=self.y, shuffle = True, test_size=0.15)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test, shuffle = True, test_size=0.3)
        self.X_train = np.array(X_train).reshape(-1, 16, self.target_size)
        self.X_test = np.array(X_test).reshape(-1, 16, self.target_size)
        self.X_val = np.array(X_val).reshape(-1, 16, self.target_size)
        self.y_train = np.array(y_train).reshape(-1, 1)
        self.y_test = np.array(y_test).reshape(-1,1)
        self.y_val = np.array(y_val).reshape(-1,1)

    def standardize_mfcc(self):
        train_mean = self.X_train.mean()
        train_std = self.X_train.std()
        self.X_train_std = (self.X_train-train_mean)/train_std
        self.X_test_std = (self.X_test-train_mean)/train_std
        self.X_val_std = (self.X_val-train_mean)/train_std

    def oversample(self):
        temp = pd.DataFrame({'mfcc_id':range(self.X_train_std.shape[0]), 'accent':self.y_train.reshape(-1)})
        temp_1 = temp[temp['accent']==1]
        idx = list(temp_1['mfcc_id'])*3
        idx = idx + list(temp_1.sample(frac=.8)['mfcc_id'])
        self.X_train_std = np.vstack((self.X_train_std, (self.X_train_std[idx]).reshape(-1, 16, self.target_size)))
        self.y_train = np.vstack((self.y_train, np.ones(232).reshape(-1,1)))

    def save_mfccs(self):
        # np.save('X_train_moz.npy', self.X_train_std)
        # np.save('X_test_moz.npy', self.X_test_std)
        # #np.save('X_val_moz.npy', self.X_val_std)
        # np.save('y_train_moz.npy', self.y_train)
        # np.save('y_test_moz.npy', self.y_test)
        # #np.save('y_val_moz.npy', self.y_val)
        np.save('X_train_moz_hop.npy', self.X_train_std)
        np.save('X_test_moz_hop.npy', self.X_test_std)
        np.save('X_val_moz_hop.npy', self.X_val_std)
        np.save('y_train_moz_hop.npy', self.y_train)
        np.save('y_test_moz_hop.npy', self.y_test)
        np.save('y_val_moz_hop.npy', self.y_val)

# 354, 293, 61
if __name__ == '__main__':
    df = clean_df('../data/mozilla_voice/validated.tsv')
    mfcc = Mfcc(df, 'path')

    mfcc.mp3towav()
    #
    mfcc.create_mfcc()

    mfcc.resize_mfcc()

    #mfcc.label_samples()

    mfcc.split_data()

    mfcc.standardize_mfcc()
    # #mfcc.oversample()
    mfcc.save_mfccs()
