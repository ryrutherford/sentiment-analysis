import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def stem_sentence(sentence):
    porter = PorterStemmer()
    words = word_tokenize(sentence)
    stemmed_sentence=[]
    for word in words:
        stemmed_sentence.append(porter.stem(word))
        stemmed_sentence.append(" ")
    return "".join(stemmed_sentence)

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
        return wordnet.NOUN

def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    pos_tagged = nltk.pos_tag(words)
    lemmatized_sentence=[]
    for word_and_pos in pos_tagged:
        lemmatized_sentence.append(lemmatizer.lemmatize(word_and_pos[0], wordnet_pos(word_and_pos[1])))
        lemmatized_sentence.append(" ")
    return "".join(lemmatized_sentence)

#feature_extraction will return sentences (pre processed or not) and their matching output (1 = positive, 0 = negative)
def data_processing(pre_processing=None):
    #extracting positive sentences
    with open('./rt-polaritydata/rt-polarity.pos', 'r', encoding='windows-1252') as positive:
        #extracting negative sentences
        with open('./rt-polaritydata/rt-polarity.neg', 'r', encoding='windows-1252') as negative:
            #unchanged positive sentences
            positive_sentences = positive.readlines()
            positive_output = np.full(len(positive_sentences),1, dtype=int).tolist()
            negative_sentences = negative.readlines()
            negative_output = np.full(len(negative_sentences), 0, dtype=int).tolist()

            sentences = positive_sentences + negative_sentences
            output = np.array(positive_output + negative_output)

            #we will only do stemming or lemmatization if it's specified
            if(pre_processing == "stem"):
                #stemmed sentences
                stemmed_sentences = []
                for sentence in sentences:
                    stemmed_sentences.append(stem_sentence(sentence))
                return stemmed_sentences, output
            elif(pre_processing == "lemmatize"):
                #lemmatized sentences
                lemmatized_sentences = []
                for sentence in sentences:
                    lemmatized_sentences.append(lemmatize_sentence(sentence))
                return lemmatized_sentences, output
            else:
                return sentences, output
        
def feature_extraction(data, min_df=5, stop_words=False):
    #basic feature extraction (words that appear less than min_df times in the doc are excluded)
    if(stop_words == False):
        cv = CountVectorizer(encoding='windows-1252', min_df=min_df)
    else:
        cv = CountVectorizer(encoding='windows-1252', stop_words=stopwords.words('english'), min_df=min_df)
    feature_vector = cv.fit_transform(data)
    """ print(feature_vector.shape)
    print(cv.get_feature_names()) """
    return feature_vector.toarray()

#splitting dataset
def split_data(features, labels, train_size):
    #will return features_train, features_test, label_train, label_test
    return train_test_split(features, labels, train_size=train_size, random_state=42)

def naive_bayes_classifier(training_features, training_labels, test_features, test_labels, filename):
    #create the Gaussian classifier
    model = GaussianNB()

    #train the model
    model.fit(training_features, training_labels)

    pred_labels = model.predict(test_features)

    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write('Naive-Bayes Accuracy: ' + str(metrics.accuracy_score(test_labels, pred_labels)) + '\n' \
            + 'Naive-Bayes Precision: ' +  str(metrics.precision_score(test_labels, pred_labels)) + '\n' \
                + 'Naive-Bayes Recall: ' + str(metrics.recall_score(test_labels, pred_labels)) + '\n' \
                    + 'Naive-Bayes F1: ' + str(metrics.f1_score(test_labels, pred_labels)))

def linear_svm_classifier(training_features, training_labels, test_features, test_labels, filename):
    #create the linear svm classifier
    model = svm.LinearSVC(dual=False)

    #train the model
    model.fit(training_features, training_labels)

    pred_labels = model.predict(test_features)

    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write('Linear-SVM Accuracy: ' + str(metrics.accuracy_score(test_labels, pred_labels)) + '\n' \
            + 'Linear-SVM Precision: ' +  str(metrics.precision_score(test_labels, pred_labels)) + '\n' \
                + 'Linear-SVM Recall: ' + str(metrics.recall_score(test_labels, pred_labels)) + '\n' \
                    + 'Linear-SVM F1: ' + str(metrics.f1_score(test_labels, pred_labels)))

def logistic_regression_classifier(training_features, training_labels, test_features, test_labels, filename):
    #create the logistic regression model
    model = LogisticRegression(dual=False, max_iter=1000)

    #train the model
    model.fit(training_features, training_labels)

    pred_labels = model.predict(test_features)

    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write('Logistic Regression Accuracy: ' + str(metrics.accuracy_score(test_labels, pred_labels)) + '\n' \
            + 'Logistic Regression Precision: ' +  str(metrics.precision_score(test_labels, pred_labels)) + '\n' \
                + 'Logistic Regression Recall: ' + str(metrics.recall_score(test_labels, pred_labels)) + '\n' \
                    + 'Logistic Regression F1: ' + str(metrics.f1_score(test_labels, pred_labels)))

def multi_layer_perceptron_classifier(training_features, training_labels, test_features, test_labels, filename):
    #create the mlp (nn) model
    model = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5,), random_state=1)

    #train the model
    model.fit(training_features, training_labels)

    pred_labels = model.predict(test_features)

    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write('Multi Layer Perceptron Accuracy: ' + str(metrics.accuracy_score(test_labels, pred_labels)) + '\n' \
            + 'Multi Layer Perceptron Precision: ' +  str(metrics.precision_score(test_labels, pred_labels)) + '\n' \
                + 'Multi Layer Perceptron Recall: ' + str(metrics.recall_score(test_labels, pred_labels)) + '\n' \
                    + 'Multi Layer Perceptron F1: ' + str(metrics.f1_score(test_labels, pred_labels)))

def train_and_predict(classifier_type=None):
    min_df_options = [5, 7, 10]
    stop_word_options = [False, True]
    training_size_options = [.6, .7, .8]
    pre_processing_options = [None, "lemmatize", "stem"]

    for pp_option in pre_processing_options:
        data, labels = data_processing(pp_option)
        for training_size_option in training_size_options:
            for stop_word_option in stop_word_options:
                for min_df in min_df_options:
                    features_vector = feature_extraction(data, min_df, stop_word_option)
                    features_train, features_test, labels_train, labels_test = split_data(features_vector, labels, training_size_option)
                    filename = "PP="+str(pp_option)+"_MIN-DF="+str(min_df)+"_STOP="+str(stop_word_option)+"_TRAIN="+str(int(training_size_option*100))+"%.txt"
                    if(classifier_type == "NB"):
                        naive_bayes_classifier(features_train, labels_train, features_test, labels_test, filename)
                    elif(classifier_type == "SVM"):
                        #train and predict on svm with linear kernel
                        linear_svm_classifier(features_train, labels_train, features_test, labels_test, filename)
                    elif(classifier_type == "LR"):
                        #train and predict on lr
                        logistic_regression_classifier(features_train, labels_train, features_test, labels_test, filename)
                    elif(classifier_type == "NN"):
                        #train and predict on nn
                        multi_layer_perceptron_classifier(features_train, labels_train, features_test, labels_test, filename)
                    else:
                        #predict on dumb classifier
                        pass

#train_and_predict("NB")
#train_and_predict("SVM")
#train_and_predict("LR")
train_and_predict("NN")