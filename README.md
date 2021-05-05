# Accent-Classification-Using-MFCCs

This repository was written to perform accent classification using Mozilla Firefox's Common Voice dataset: </br>
	https://commonvoice.mozilla.org/en/datasets

Dependencies: </br>
	keras </br>
	tensorflow </br>
	matplotlib </br>
	librosa	</br>
	IPython </br>
	sklearn </br>
	pydub </br>
	itertools </br>
		
To generate the MFCCs from this dataset, you need to run scripts/eda_mozilla.py
	This will select a random number of audio samples of preset classes and compile all of the neccessary MFCCs. It will then split and save npy files for training, testing, and validation. The data generation script was adapted from Rao N.'s repository which can be found here: https://github.com/nkrao220/accent-classification. The dataset used for our final results can be download here: https://drive.google.com/file/d/12ybj9jYFgZezGPe7PNwssJuV_LxG8yP7/view?usp=sharing. This dataset consist of 12,000 samples from 3 accents (US, UK, and Indian).
	
To run the training algorithm, you will need to run scripts/big_cnn_3.py
	This will import the npy dataset files and train the Neural Network explained in our final report.
	Afterward, it will display training/validation loss and accuracy followed by an ROC curve for each class. </br>
        Pre-trained weights can be downloaded here: https://drive.google.com/file/d/1FQK54vdxRB09j8lw9-4uMkiyFdDX-Cz-/view?usp=sharing. </br>

To reproduce the graphs, you can run scripts/baseline_models.py to generate either the Naive Bayes, Tree Classifier, and LDA plots. This was adapted from Jasmine He's repository which can be found here: https://github.com/Jasminehh/doodle_image_recognition_CNN. The script/big_cnn_3_test.py was used to reproduce the ROC and the confusion matrix. </br>
