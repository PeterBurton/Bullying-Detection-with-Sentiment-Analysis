# -*- coding: utf-8 -*-
"""
@author: Peter Burton R00038147
"""

import time
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import itertools

#Function to remove stop words from a string
def remove_stop_words(element, stop_words):
    
    word_tokens = word_tokenize(element)
    sentence = ""
    for w in word_tokens:
        if w not in stop_words:
            sentence  += w
    
    return sentence

#Function to make word grams of a given number i.e. 3
def word_grams(words, number):
    
    tokens = word_tokenize(words)
    ngram_list = list(ngrams(tokens, number))
    sentence = ""
    for word in ngram_list:
        sentence += str(word)
    
    return sentence


#Function to print confusion matrices to assess accuracy of classifiers
def plot_confusion_matrix(cm,classes,title='Confusion matrix'):
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    
def load_file():
    
    data = []
    target = []
    
    df = pd.read_csv("../datasets/Xu_Jun_Zhu_Bellmore_data/biggest_data.csv")
    df = df[pd.notnull(df["Answer.ContainCyberbullying"])]
    print(df['Answer.ContainCyberbullying'].value_counts())
    
    yes_count = 0
    for index, row in df.iterrows():
        if row['Answer.ContainCyberbullying'] == 'Yes':
            yes_count = yes_count + 1
    
    print(yes_count)
    
    no_counter = 0
    for index, row in df.iterrows():
        if row['Answer.ContainCyberbullying'] == 'Yes':
            data.append(row['Input.posttext'])
            target.append(row['Answer.ContainCyberbullying'])
        else:
            if no_counter < yes_count:
                data.append(row['Input.posttext'])
                target.append(row['Answer.ContainCyberbullying'])
                no_counter = no_counter + 1
    
    return data, target


def preprocess():
    
    # Initialise Porter Stemmer & Lemmatization
    ps = PorterStemmer()
    lem = WordNetLemmatizer()
    # Create a set to hold stopwords that we don't want from NLTK
    stop_words = set(stopwords.words('english'))
    
    data,target = load_file()
    
#==============================================================================
#     #Stemming using Porter Stemming Algorithm
#     data = [(' '.join(ps.stem(token) for token in word_tokenize(element))) for element in data]
#==============================================================================
#==============================================================================
#     #Lemmatization using WordNet Lemmatizer Algorithm
#     data = [(' '.join(lem.lemmatize(token) for token in word_tokenize(element))) for element in data]
#==============================================================================
#==============================================================================
#     #Make Data into N-grams
#     data = [word_grams(element, 3) for element in data]
#==============================================================================
    #Stop word removal
    data = [remove_stop_words(element, stop_words) for element in data]

    
    count_vectorizer = CountVectorizer(binary='true')
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)

    return tfidf_data, target


def train_eval(data,target):
    
    #Array to hold the names of the classifiers we are going to test
    names = ["KNN", "SVC", "Decision Tree", "Random Forest", "Bernoulli Naive Bayes"]
    
    #Array holding the actual classifiers
    classifiers = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    BernoulliNB()]
    
    
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.3,random_state=43)
    for name, clf in zip(names, classifiers):
        if name == "SVC":
            clf = SVC(probability=True, C=1000)
        clf.fit(data_train,target_train)
        predicted = clf.predict(data_test)
        print("=================================================================")
        print(name,"classifier results")
        print("-----------------------------------------------------------------")
        #evaluate_model(target_test,predicted)
        print(classification_report(target_test,predicted))
        print("The accuracy score is {:.2%}".format(accuracy_score(target_test,predicted)))
        cnf_matrix = confusion_matrix(target_test,predicted)
        #Plot the confusion matrix
        graph_name = (name, "Confusion Matrix")
        plot_confusion_matrix(cnf_matrix, classes=['not bullying', 'bullying'],title =graph_name)
        
        
def main():
    
    #Get a start time for the program
    start_time = time.time()
    tf_idf, target = preprocess()
    train_eval(tf_idf,target)
    
    #Get the time taken in seconds
    print("\n=================================================================")
    print("Program ran in %s seconds" % (time.time() - start_time))
    print("=================================================================\n")
    
    
main()