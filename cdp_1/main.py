import string
import re
import warnings
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from pandas import DataFrame
from matplotlib import pyplot
from gensim.models import KeyedVectors
from textblob import TextBlob


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r',encoding="utf8")
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# turn a doc into clean tokens
# turn a doc into clean tokens
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# turn a doc into clean tokens
def clean_doc_for_vocab(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc_for_vocab(doc)
    # update counts
    vocab.update(tokens)


# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


def process_docs_for_vocab(directory, vocab):
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)





# load all docs in a directory
def process_docs(directory, vocab, is_train):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents




# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()


# load and clean a dataset
def load_clean_dataset(vocab, is_train):
    # load documents
    neg = process_docs('C:\\tfsdataextract\\txt_sentoken\\neg', vocab, is_train)
    pos = process_docs('C:\\tfsdataextract\\txt_sentoken\\pos', vocab, is_train)
    docs = neg + pos
    # prepare labels
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
    return docs, labels

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded
# classify a review as negative or positive
def predict_sentiment(review, vocab, tokenizer, max_length, model):
    # clean review
    line = clean_doc(review, vocab)
    # encode and pad review
    padded = encode_docs(tokenizer, max_length, [line])
    # predict sentiment
    yhat = model.predict(padded, verbose=0)
    # retrieve predicted percentage and label
    percent_pos = yhat[0, 0]
    if round(percent_pos) == 0:
        return (1 - percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'




# define the model
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


# evaluate a neural network model
def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
    scores = list()
    n_repeats = 10
    n_words = Xtest.shape[1]
    for i in range(n_repeats):
        # define network
        model = define_model(n_words)
        # fit network
        model.fit(Xtrain, ytrain, epochs=10, verbose=0)
        # evaluate
        _, acc = model.evaluate(Xtest, ytest, verbose=0)
        scores.append(acc)
        print('%d accuracy: %s' % ((i + 1), acc))

    return scores


# prepare bag of words encoding of docs
def prepare_data(train_docs, test_docs, mode):
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)
    # encode training data set
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
    # encode training data set
    Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
    return Xtrain, Xtest




def createfiles(i):
    outF = open("W0.6a.txt", "r", encoding='utf-8').readlines()
    for line in outF:
        newf = open("C:\\tfsdataextract\\txt_sentoken\\voca\\cv0_" + str(i) + ".txt", "w", encoding='utf-8')
        i -= 1
        newf.write(line)
        newf.close()

def extractwatdatafrmcity():
    # extract data related to wate from cities
    cities_data = pd.read_csv("C:\\cdpdata\\Cities_Data_2017-2019_mb2.csv", encoding='cp1252')
    # acntname = tt['Account Name']
    is_water = cities_data['Question Number'] == '15.4'
    # frequency =tt['Response Answer'].value_counts()
    q9 = cities_data[is_water]
    q9.to_csv("C:\\tfsdataextract\\txt_sentoken\\voca\\cities_ques_9.csv")

def extract_response_rel2_water():
    city_q_data = pd.read_csv("C:\\cdpdata\\water data\\city_data_extract\\cities_ques_6_2.csv")
    for item in city_q_data['Response Answer'].drop_duplicates().items():
        if 'water' in str(item[1]):
            print(item[1])
#find unique responses for particular column in question
def find_unique_response_for_q(queno,column):
    city_q_data = pd.read_csv("C:\\cdpdata\\water data\\city_data_extract\\cities_ques_"+queno+".csv")
    for item in city_q_data[column].drop_duplicates().items():
        print(item[1])

find_unique_response_for_q('9_1','Response Answer')
#print(frequency)








#split reviews into new files
#createfiles(532)
# create vocab code
# define vocab
# vocab = Counter()
# # add all docs to vocab
# process_docs_for_vocab('C:\\tfsdataextract\\txt_sentoken\\voca', vocab)
#
# # print the size of the vocab
# print(len(vocab))
# # keep tokens with a min occurrence
# min_occurane = 2
# tokens = [k for k,c in vocab.items() if c >= min_occurane]
# print(len(tokens))
# # save tokens to a vocabulary file
# save_list(tokens, 'vocab6a.txt')
# print(vocab.most_common(400))


#train model start
# load the vocabulary and train the model
# vocab_filename = 'vocab_3.2.txt'
# vocab = load_doc(vocab_filename)
# vocab = set(vocab.split())
# # load training data
# train_docs, ytrain = load_clean_dataset(vocab, True)
# # create the tokenizer
# tokenizer = create_tokenizer(train_docs)
# # define vocabulary size
# vocab_size = len(tokenizer.word_index) + 1
# print('Vocabulary size: %d' % vocab_size)
# # calculate the maximum sequence length
# max_length = max([len(s.split()) for s in train_docs])
# print('Maximum length: %d' % max_length)
# # encode data
# Xtrain = encode_docs(tokenizer, max_length, train_docs)
# # define model
# model = define_model(vocab_size, max_length)
# # fit network
# model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# # save the model
# model.save('model.h5')
#train model end
#EVALUATE MODEL
# load the vocabulary
# vocab_filename = 'vocab_3.2.txt'
# vocab = load_doc(vocab_filename)
# vocab = set(vocab.split())
# # load all reviews
# train_docs, ytrain = load_clean_dataset(vocab, True)
# test_docs, ytest = load_clean_dataset(vocab, False)
# # create the tokenizer
# tokenizer = create_tokenizer(train_docs)
# # define vocabulary size
# vocab_size = len(tokenizer.word_index) + 1
# print('Vocabulary size: %d' % vocab_size)
# # calculate the maximum sequence length
# max_length = max([len(s.split()) for s in train_docs])
# print('Maximum length: %d' % max_length)
# # encode data
# Xtrain = encode_docs(tokenizer, max_length, train_docs)
# Xtest = encode_docs(tokenizer, max_length, test_docs)
# # load the model
# model = load_model('model.h5')
# # evaluate model on training dataset
# _, acc = model.evaluate(Xtrain, ytrain, verbose=0)
# print('Train Accuracy: %.2f' % (acc*100))
# # evaluate model on test dataset
# _, acc = model.evaluate(Xtest, ytest, verbose=0)
# print('Test Accuracy: %.2f' % (acc*100))
# #test positive text
# #text = 'Our operations are exposed if there is a drought situation that limits our ability to use water for the manufacture of our paper products.  Climate change has impacted our region and we are experience wide variations in winter snow pack and wide variations in summer precipitation.'
# #text ='Potable water is a necessary resource to our manufacturing process. Should water become extremelyscarce in regions where we manufacture our products, it could potentially have an impact on our manufacturing operations. This potential risk has been identified at a functional level within Environment, Health, Safety, and Sustainability and is considered a low potential risk.'
# #text ='The water is an important resource for the Griffith business. Without water, it is not possible to produce their products, which by the way has the water as an ingrediente, or maintain their structure in adequate conditions.'
# #text = 'Water security is really a significant impact on us'
# text = 'We produce aluminum sheet and light gauge products for use in multiple markets, which includes beverage can.  Our single largest end-use market is beverage can sheet. The beverage market is highly dependent on fresh clean water as a key raw material, within the manufacturing process and other product ingredients.  Certain of our customers in the beverage market are significant to our revenues, and we could be adversely affected by changes in the business or financial condition of these significant customers or by the loss of their business.'
# percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
# print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
# # test negative text
# #text = 'Not internally defined.'
# text = 'The risk to our operations is strictly due to interruptioor curtailment, as opposed to the cost of water. During our risk assessment, we concluded that the facilities that consume the largest percentage of water are not iwater stressed regions.'
# #text ='At JDSU, a material cost is defined as being over $500,000. Most water expenditure is below this amount and is not significant enough to create substantive change ithe business units.'
# #text ='Zero impact of water security'
# percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
# print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))

