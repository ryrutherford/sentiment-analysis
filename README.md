# Sentiment Analysis of IMDB Movie Reviews (Supervised Learning Project)

- I use scikit-learn, nltk, and NumPy to train and test different ML classifiers on 10,000+ sentences reflecting either positive or negative sentiment of a movie
- The goal of this project is to determine which model and what pre-processing (stemming, lemmatization, stop words, minimum frequency) are most effective at predicting positive or negative sentiment from a single sentence
- As of right now (September 23, 2020) I am only judging a model's effectiveness by it's accuracy, precision, recall, and f1 score but I will soon be adding other metrics (macro-average, micro-average, etc.)
- I have currently trained and tested 108 models (36 Naive-Bayes, 36 Linear SVM, 36 Logistic Regression) but will be tesing 72 more using a non linear model (yet to decide) and "dumb" model that predicts positive or negative based on a "coin flip" 