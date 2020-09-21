from sklearn.feature_extraction.text import CountVectorizer
import nltk
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

#extracting positive sentences
with open('./rt-polaritydata/rt-polarity.pos', 'r', encoding='windows-1252') as positive:
    #unchanged positive sentences
    positive_sentences = positive.readlines()

    #stemmed positive sentences
    positive_stemmed_sentences = []
    for sentence in positive_sentences:
        positive_stemmed_sentences.append(stem_sentence(sentence))
    
    #lemmatized positive sentences
    positive_lemmatized_sentences = []
    for sentence in positive_sentences:
        positive_lemmatized_sentences.append(lemmatize_sentence(sentence))
        

#extracting negative sentences
with open('./rt-polaritydata/rt-polarity.neg', 'r', encoding='windows-1252') as negative:
    #unchanged negative sentences
    negative_sentences = negative.readlines()

    #stemmed negative sentences
    negative_stemmed_sentences = []
    for sentence in negative_sentences:
        negative_stemmed_sentences.append(stem_sentence(sentence))

    #lemmatized negative sentences
    negative_lemmatized_sentences = []
    for sentence in negative_sentences:
        negative_lemmatized_sentences.append(lemmatize_sentence(sentence))

#basic feature extraction (words that appear less than min_df times in the doc are excluded)
cv = CountVectorizer(encoding='windows-1252', min_df=5)

#the matrices will always have 5331 rows (1 for each sentence)
#the number of columns (features/words) will vary depending on the tools being used
pos_matrix = cv.fit_transform(positive_sentences)
print("pos_matrix " + str(pos_matrix.shape))
neg_matrix = cv.fit_transform(negative_sentences)
print("neg_matrix " + str(neg_matrix.shape))

#basic feature extraction with stemmed sentences
pos_stemmed_matrix = cv.fit_transform(positive_stemmed_sentences)
print("pos_stemmed_matrix " + str(pos_stemmed_matrix.shape))
neg_stemmed_matrix = cv.fit_transform(negative_stemmed_sentences)
print("neg_stemmed_matrix " + str(neg_stemmed_matrix.shape))

#basic feature extraction with lemmatized sentences
pos_lemmatized_matrix = cv.fit_transform(positive_lemmatized_sentences)
print("pos_lemmatized_matrix " + str(pos_lemmatized_matrix.shape))
neg_lemmatized_matrix = cv.fit_transform(negative_lemmatized_sentences)
print("neg_lemmatized_matrix " + str(neg_lemmatized_matrix.shape))

#feature extraction with stop words (words that appear less than min_df times in the doc are excluded)
cv_stop = CountVectorizer(encoding='windows-1252', stop_words=stopwords.words('english'), min_df=5)

#feature extraction with stop words on unchanged sentences
pos_stop_matrix = cv_stop.fit_transform(positive_sentences)
print("pos_stop_matrix " + str(pos_stop_matrix.shape))
neg_stop_matrix = cv_stop.fit_transform(negative_sentences)
print("neg_stop_matrix " + str(neg_stop_matrix.shape))

#feature extraction with stop words on stemmed sentences
pos_stemmed_stop_matrix = cv_stop.fit_transform(positive_stemmed_sentences)
print("pos_stemmed_stop_matrix " + str(pos_stemmed_stop_matrix.shape))
neg_stemmed_stop_matrix = cv_stop.fit_transform(negative_stemmed_sentences)
print("neg_stemmed_stop_matrix " + str(neg_stemmed_stop_matrix.shape))

#feature extraction with stop words on lemmatized sentences
pos_lemmatized_stop_matrix = cv_stop.fit_transform(positive_lemmatized_sentences)
print("pos_lemmatized_stop_matrix " + str(pos_lemmatized_stop_matrix.shape))
neg_lemmatized_stop_matrix = cv_stop.fit_transform(negative_lemmatized_sentences)
print("neg_lemmatized_stop_matrix " + str(neg_lemmatized_stop_matrix.shape))