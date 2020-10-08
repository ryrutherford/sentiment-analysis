"""
This python script will train, and test ml models on sentences that convey either positive or negative sentiment
about a movie.
"""

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#helper function to apply the porter stemmer to sentences
def stem_sentence(sentence):
    porter = PorterStemmer()
    words = word_tokenize(sentence)
    stemmed_sentence=[]
    for word in words:
        stemmed_sentence.append(porter.stem(word))
        stemmed_sentence.append(" ")
    return "".join(stemmed_sentence)

#helper function to stem the stopwords list
def stem_stopwords(stop_words):
    porter = PorterStemmer()
    stemmed_stopwords=[]
    for word in stop_words:
        stemmed_stopwords.append(porter.stem(word))
    return stemmed_stopwords

#helper function to return a wordnet pos tag based on the nltk tag identified
def wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    elif nltk_tag.startswith('S'):
        return wordnet.ADJ_SAT
    else:
        return None

#a helper function to lemmatize the words in a sentence and return the lemmatized sentence
def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    pos_tagged = nltk.pos_tag(words)
    lemmatized_sentence=[]
    for word_and_pos in pos_tagged:
        tag = wordnet_pos(word_and_pos[1])
        if(tag == None):
            lemmatized_sentence.append(lemmatizer.lemmatize(word_and_pos[0]))
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word_and_pos[0], wordnet_pos(word_and_pos[1])))
        lemmatized_sentence.append(" ")
    return "".join(lemmatized_sentence)

def lemmatize_stopwords(stop_words):
    lemmatizer = WordNetLemmatizer()
    pos_tagged = nltk.pos_tag(stop_words)
    lemmatized_stopwords =[]
    for word_and_pos in pos_tagged:
        tag = wordnet_pos(word_and_pos[1])
        if(tag == None):
            lemmatized_stopwords.append(lemmatizer.lemmatize(word_and_pos[0]))
        else:
            lemmatized_stopwords.append(lemmatizer.lemmatize(word_and_pos[0], wordnet_pos(word_and_pos[1])))
    return lemmatized_stopwords

#feature_extraction will return sentences (pre processed or not) and their matching output (1 = positive, 0 = negative)
def data_processing(pre_processing=None):
    #extracting positive sentences
    with open('rt-polaritydata/rt-polarity.pos', 'r', encoding='cp1252') as positive:
        #extracting negative sentences
        with open('rt-polaritydata/rt-polarity.neg', 'r', encoding='cp1252') as negative:
            #unchanged positive sentences
            positive_sentences = positive.readlines()
            #creating the labels for the positive sentences
            positive_output = np.full(len(positive_sentences),1, dtype=int).tolist()
            #unchanged negative sentences
            negative_sentences = negative.readlines()
            #creating the labels for the negative sentences
            negative_output = np.full(len(negative_sentences), 0, dtype=int).tolist()

            #combining the positive sentences and negative sentences into one list
            sentences = positive_sentences + negative_sentences
            #combining the output labels to match the complete list of sentences
            output = np.array(positive_output + negative_output)

            #we will only do stemming or lemmatization if it's specified
            if(pre_processing == "stem"):
                #stemmed sentences
                stemmed_sentences = []
                #stemming each sentence in the list using the helper function
                for sentence in sentences:
                    stemmed_sentences.append(stem_sentence(sentence))
                return stemmed_sentences, output
            elif(pre_processing == "lemmatize"):
                #lemmatized sentences
                lemmatized_sentences = []
                #lemmatizing each sentence in the list using the helper function
                for sentence in sentences:
                    lemmatized_sentences.append(lemmatize_sentence(sentence))
                return lemmatized_sentences, output
            else:
                return sentences, output

#function to return a feature vector that can be passed to scikit-learn ml models
#data will be the sentences
#min_df specifies the minimum number of times a word must appear in the document to be included as a feature
#stop_words specifies whether the English stop words set should be used or not       
def feature_extraction(data, stop_words, min_df=5):
    #basic feature extraction (words that appear less than min_df times in the doc are excluded)
    if(stop_words == None):
        cv = CountVectorizer(encoding='cp1252', min_df=min_df)
    else:
        cv = CountVectorizer(encoding='cp1252', stop_words=stop_words, min_df=min_df)
    feature_vector = cv.fit_transform(data)
    #changing the countvector into a tfidf vector
    transformer = TfidfTransformer()
    feature_vector = transformer.fit_transform(feature_vector)
    return feature_vector.toarray()

#function to split dataset into training and test sets based on the specified train_size percentage
def split_data(features, labels, train_size):
    #will return features_train, features_test, label_train, label_test
    return train_test_split(features, labels, train_size=train_size, random_state=42)

#function to plot a confusion matrix
def plot_cfm(test_labels, pred_labels):
    cm = confusion_matrix(test_labels, pred_labels)

    ConfusionMatrixDisplay(cm).plot()
    plt.show()

#function that will train a multinomial naive-bayes model on the training data and then test the model on the test data
#looking at 4 metrics (accuracy, precision, recall, and f1) to test the effectiveness of the model
#alpha parameter determins the smoothing (default is 1)
def multinomial_naive_bayes_classifier(training_features, training_labels, test_features, test_labels, filename, confmat=False, alpha=1.0):
    #parameters to be tested in GridSearchCV
    #parameters were chosen based off research on stackoverflow (and out of interest of saving computation time)
    parameters = {
            'alpha': [1e-1, 1e-3, 1],
            'fit_prior': [True, False]
        }
    gscv = GridSearchCV(MultinomialNB(), parameters, n_jobs=-1, verbose=10 )
    gscv = gscv.fit(training_features, training_labels)
    
    best_params = gscv.best_params_

    model = MultinomialNB(alpha=best_params["alpha"], fit_prior=best_params["fit_prior"])

    #train the model with the best params
    model.fit(training_features, training_labels)

    pred_labels = model.predict(test_features)

    if(confmat == True):
        plot_cfm(test_labels, pred_labels)

    with open("MNB_alpha="+str(best_params["alpha"])+"_fit-prior="+str(best_params["fit_prior"])+"_"+filename, \
        'w', encoding='utf-8') as outfile:
        outfile.write(classification_report(test_labels, pred_labels, target_names=["Negative", "Positive"]))

#function that will train a linear-svm model on the training data and then test the model on the test data
def linear_svm_classifier(training_features, training_labels, test_features, test_labels, filename, confmat=False):
    #parameters to be tested in GridSearchCV
    #parameters were chosen based off research on stackoverflow (and out of interest of saving computation time)
    parameters = {
            'C' : [0.4, 0.55, 0.7],
            'max_iter' : [75, 100],
            'class_weight': ['balanced', None],
            'dual': [False]
        }
    gscv = GridSearchCV(svm.LinearSVC(), parameters, n_jobs=-1, verbose=10 )
    gscv = gscv.fit(training_features, training_labels)
    
    best_params = gscv.best_params_

    model = svm.LinearSVC(C=best_params["C"], max_iter=best_params["max_iter"], class_weight=best_params["class_weight"])

    #train the model with the best params
    model.fit(training_features, training_labels)

    pred_labels = model.predict(test_features)

    if(confmat == True):
        plot_cfm(test_labels, pred_labels)

    with open("SVM_C="+str(best_params["C"])+"_max-iter="+str(best_params["max_iter"])+"_class-weight="+str(best_params["class_weight"])+"_"+filename, \
        'w', encoding='utf-8') as outfile:
        outfile.write(classification_report(test_labels, pred_labels, target_names=["Negative", "Positive"]))

#function that will train a logistic regression model on the training data and then test the model on the test data
def logistic_regression_classifier(training_features, training_labels, test_features, test_labels, filename, confmat=False):
    #parameters to be tested in GridSearchCV
    parameters = {
            'C' : [0.5, 2 ,3.5],
            'tol' : [0.005, 0.01],
            'max_iter': [1000]
        }
    gscv = GridSearchCV(LogisticRegression(), parameters, n_jobs=-1, verbose=10)
    gscv = gscv.fit(training_features, training_labels)
    
    best_params = gscv.best_params_

    model = LogisticRegression(max_iter=1000, tol=best_params["tol"], C=best_params["C"])
    
    #train the model with the best params
    model.fit(training_features, training_labels)

    pred_labels = model.predict(test_features)

    if(confmat == True):
        plot_cfm(test_labels, pred_labels)

    with open("LR_C="+str(best_params["C"])+"_max-iter="+str(best_params["max_iter"])+"_tol="+str(best_params["tol"])+"_"+filename,\
        'w', encoding='utf-8') as outfile:
        outfile.write(classification_report(test_labels, pred_labels, target_names=["Negative", "Positive"]))

#function that will train a gaussian naive bayes regression model on the training data and then test the model on the test data
def gaussian_naive_bayes_classifier(training_features, training_labels, test_features, test_labels, filename, confmat=False):
    #parameters to be tested in GridSearchCV
    parameters = {
            'var_smoothing': [0.2, 0.5, 0.7]
        }
    gscv = GridSearchCV(GaussianNB(), parameters, n_jobs=-1, verbose=10 )
    gscv = gscv.fit(training_features, training_labels)
    
    best_params = gscv.best_params_

    model = GaussianNB(var_smoothing=best_params["var_smoothing"])

    #train the model with the best params
    model.fit(training_features, training_labels)

    pred_labels = model.predict(test_features)

    if(confmat == True):
        plot_cfm(test_labels, pred_labels)

    with open("GNB_var-smoothing="+str(best_params["var_smoothing"])+"_"+filename, 'w', encoding='utf-8') as outfile:
        outfile.write(classification_report(test_labels, pred_labels, target_names=["Negative", "Positive"]))

#function that will train a dummy model on the training set and then predict an outcome uniformly at random
def dummy_classifier(training_features, training_labels, test_features, test_labels, filename, confmat=False):
    #create the dummy classifier using the uniform strat
    model = DummyClassifier(strategy="uniform")

    #train the model
    model.fit(training_features, training_labels)

    #test the model
    pred_labels = model.predict(test_features)

    if(confmat == True):
        plot_cfm(test_labels, pred_labels)

    #analyze performance
    with open("DUMMY_"+filename, 'w', encoding='utf-8') as outfile:
        outfile.write(classification_report(test_labels, pred_labels, target_names=["Negative", "Positive"]))

#function that will train the specified classifier model and then test it. there will be 72 versions of the specified model trained and tested
#the goal is to determing which combination of parameters (min_df, stop words, training size, text pre processing) leads to the most effective model
def train_and_predict(classifier_type="Dummy", min_df_options=[1,5,7,10], stop_word_options=[False, True], training_size_options=[.6, .7, .8], pre_processing_options=[None, "lemmatize", "stem"], confmat=False):

    for pp_option in pre_processing_options:
        data, labels = data_processing(pp_option)
        stop_words = None
        if(pp_option == "lemmatize"):
            stop_words = lemmatize_stopwords(stopwords.words("english"))
        elif(pp_option == "stem"):
            stop_words = stem_stopwords(stopwords.words("english"))
        for training_size_option in training_size_options:
            for stop_word_option in stop_word_options:
                for min_df in min_df_options:
                    features_vector = feature_extraction(data, stop_words, min_df)
                    features_train, features_test, labels_train, labels_test = split_data(features_vector, labels, training_size_option)
                    filename = "_PP="+str(pp_option)+"_MIN-DF="+str(min_df)+"_STOP="+str(stop_word_option)+"_TRAIN="+str(int(training_size_option*100))+"%.txt"
                    if(classifier_type == "MNB"):
                        #train and predict with multinomial naive bayes
                        multinomial_naive_bayes_classifier(features_train, labels_train, features_test, labels_test, filename, confmat)
                    elif(classifier_type == "SVM"):
                        #train and predict on svm with linear kernel
                        linear_svm_classifier(features_train, labels_train, features_test, labels_test, filename, confmat)
                    elif(classifier_type == "LR"):
                        #train and predict on lr
                        logistic_regression_classifier(features_train, labels_train, features_test, labels_test, filename, confmat)
                    elif(classifier_type == "GNB"):
                        #train and predict on gnb
                        gaussian_naive_bayes_classifier(features_train, labels_train, features_test, labels_test, filename, confmat)
                    elif(classifier_type == "Dummy"):
                        #predict on dumb classifier
                        dummy_classifier(features_train, labels_train, features_test, labels_test, filename, confmat)


#Running this program as is will train and test 72 models per classifier
#Pass min_df_options=Array of Integers as a parameter to train_and_predict to specify the minimum frequency of words required for features 
#Pass stop_word_options=Array of Booleans as a parameter to train_and_predict to specify whether to use stopwords or not
#Pass training_size_options=Array of Floats between 0 and 1.0 as a parameter to train_and_predict to specify which training sizes to use
#Pass preprocessing_options=Array of Strings as a parameter to train_and_predict to specify whether to lemmatize ("lemmatize"), stem ("stem"), or leave unnormalized (None)
#Pass confmat=True as a parameter to train_and_predict to generate a confusion matrix
if __name__ == '__main__':
    train_and_predict("MNB") 
    train_and_predict("SVM")
    train_and_predict("LR")
    train_and_predict("GNB")
    train_and_predict()