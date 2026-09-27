# Acoustic Feature-Based Detection and Classification of Speech Disfluencies Using Machine Learning

This is a machine learning pipeline for detecting and classifying instances of stuttering with a self-written research paper alongside it.

## Motivation

Roughly 80 million people stutter worldwide, including me, yet in many parts of the world it remains undiagnosed and untreated, as people tend to not see it as a priority. It also relies on manual evaluation from a speech language pathologist, which can be time consuming, subjective, and may be inaccessible to many. I built this project to explore the possibilities of what machine learning can offer to make treatment more efficient and accessible to all.

## Research Question

Can acoustic features extracted from short audio clips accurately detect the presence of speech disfluencies, and can machine learning models reliably classify specific disfluency types?

## Dataset

This project uses the [UCLASS Stuttered Speech Clips (SEP-28K Format)](https://www.kaggle.com/datasets/vudominhgiang/uclass-stuttered-speech-clips-sep-28k-format) dataset found on Kaggle. More information can be found on the link.

## Methods

Initially, I extracted 13 Mel-Frequency Cepstral Coefficients (MFCCs) per clip as they are the standard feature for speech processing because of their ability to compress a spectrogram into a numeric representation. Later, I added 13 MFCC standard deviations to measure how much the means vary over time, zero crossing rate to aid in detection for stutter types such as blocks, root mean square energy to measure average loudness, and spectral centroid to further help distinguish stutter types for multi-label classification. For binary classification, I tested random forest, support vector machine (SVM), and logistic regression models. For multi-label classification, I tested random forest, increasing the feature count to 29, adding more sample clips by using SMOTE, and also a classifier chain. I attempted to use a convolutional neural network (CNN) as well.

## Results

The results of binary classification show that random forest obtained the highest accuracy and F1 score, 92.8% and 0.93, followed by SVM, 86.4% and 0.87, with logistic regression having the lowest, 77.2% and 0.79. Testing for multi-label classification revealed the F1 scores per each category: Block, Prolongation, SoundRep, WordRep, Interjection, and NoStutteredWords. Prior to shuffling, the dataset exhibited ordering bias with the initial five test samples being fluent clips. To assess the impact of shuffling, we evaluated the baseline model on both the original ordered and shuffled splits. The differences were minimal, ranging from 0.01 to 0.08 F1 score across labels, suggesting the original ordering was not severely biased but shuffling was retained as a correct methodological practice to ensure a representative and unbiased test set. The baseline random forest with 13 features led to the resulting F1 scores of 0.74, 0.37, 0.74, 0.67, 0.64, and 0.90. When implementing SMOTE to increase sample sizes for minority categories, the results were 0.35, 0.47, 0.45, 0.67, 0.56, and 0.88. Upon removing SMOTE and increasing the feature count to 29, the results went to 0.78, 0.44, 0.77, 0.74, 0.66, and 0.92. Using the 29 features with a classifier chain led to the F1 scores of 0.79, 0.46, 0.80, 0.72, 0.66, and 0.92. Multi-label classifying by using a CNN led to noticeably worse results, achieving F1 scores of 0.69, 0.25, 0.69, 0.06, 0.51, and 0.90. Across all approaches, prolongation was the most difficult to detect, with F1 scores ranging from 0.25 to 0.47.

## How to Run

### Prerequisites:

- Google account
- Kaggle account

### Steps:

1. Download the .ipynb
2. Go to [colab.reseach.google.com](https://colab.research.google.com) and upload the notebook
3. Go to [kaggle.com](https://kaggle.com), then settings, API Tokens, then create a legacy API key to download as a .json
4. Run the first 2 cells
5. When you get to the 3rd one, upload your kaggle.json key and then you will be able to run the rest of the cells

## Dependencies

### Pre-Installed in Colab:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- Pillow

### Installed in my code:

- librosa
- imbalanced-learn
- iterative-stratification
