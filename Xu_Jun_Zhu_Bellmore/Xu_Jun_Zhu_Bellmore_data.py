# -*- coding: utf-8 -*-
"""
@author: Peter Burton R00038147
"""

import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn import model_selection
from sklearn.externals import joblib
from scipy.stats import randint
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

def plot_roc_curve(target_test, predicted, name):
    fpr, tpr, threshold = metrics.roc_curve(target_test, predicted, pos_label='Yes')
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.title(name + ' Receiver Operating Characteristic Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    
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
    #Make Data into N-grams
    data = [word_grams(element, 3) for element in data]
#==============================================================================
#     #Stop word removal
#     data = [remove_stop_words(element, stop_words) for element in data]
#==============================================================================

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    tfidf_data = tfidf.fit_transform(data)
    
#==============================================================================
#     count_vectorizer = CountVectorizer(binary='true')
#     data = count_vectorizer.fit_transform(data)
#     tfidf_data = TfidfTransformer(use_idf=True).fit_transform(data)
#==============================================================================

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
    
    
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.2,random_state=43)
    for name, clf in zip(names, classifiers):
        if name == "SVC":
            clf = SVC(probability=True, C=1000)
        clf.fit(data_train,target_train)
        predicted = clf.predict(data_test)
        predicted_probs = clf.predict_proba(data_test)[:,1]
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
        #Plot the ROC curve
        plot_roc_curve(target_test, predicted_probs, name)
    hyper_parameter_optimization(data_train,data_test,target_train,target_test)
        
def hyper_parameter_optimization(data_train,data_test,target_train,target_test):
    
    print("=================================================================")
    print("Optimizing SVC hyper parameters")
    print("-----------------------------------------------------------------")
    param_grid = [ {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma':[0.001, 0.01, 0.1, 1]} ]
    clf = GridSearchCV(SVC(probability=True), param_grid, cv=10)
    clf.fit(data_train, target_train)
    print("Best parameters found:", clf.best_params_)
    print("-----------------------------------------------------------------")
    joblib.dump(clf.best_estimator_, 'SVC.pkl')
    classifier= joblib.load('SVC.pkl')
    #Get classifiers predictions for the test set
    test_predict  = classifier.fit(data_train,target_train).predict(data_test)
    predicted_probs = classifier.fit(data_train,target_train).predict_proba(data_test)[:,1]
    print(classification_report(target_test,test_predict))
    print("The accuracy score is {:.2%}".format(accuracy_score(target_test,test_predict)))
    
    #Plot the confusion matrix
    cnf_matrix = confusion_matrix(target_test,test_predict)
    graph_name = ("SVC Confusion Matrix")
    plot_confusion_matrix(cnf_matrix, classes=['not bullying', 'bullying'],title =graph_name)
    #Plot the ROC curve
    plot_roc_curve(target_test, predicted_probs, "SVC")
    print("Parameters were: ", classifier.get_params())
    
    print("=================================================================")
    print("Optimizing Random Forest hyper parameters")
    print("-----------------------------------------------------------------")
    param_grid = [ {'n_estimators':list(range(10,190,20)), 'criterion':["gini","entropy"], 'max_features':["auto","log2","sqrt"]} ]
    clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=10)
    clf.fit(data_train, target_train)
    print("Best parameters found:", clf.best_params_)
    print("-----------------------------------------------------------------")
    joblib.dump(clf.best_estimator_, 'random_forest.pkl')
    classifier= joblib.load('random_forest.pkl')
    #Get classifiers predictions for the test set
    test_predict  = classifier.fit(data_train,target_train).predict(data_test)
    predicted_probs = classifier.fit(data_train,target_train).predict_proba(data_test)[:,1]
    print(classification_report(target_test,test_predict))
    print("The accuracy score is {:.2%}".format(accuracy_score(target_test,test_predict)))
    
    #Plot the confusion matrix
    cnf_matrix = confusion_matrix(target_test,test_predict)
    graph_name = ("Random Forest Confusion Matrix")
    plot_confusion_matrix(cnf_matrix, classes=['not bullying', 'bullying'],title =graph_name)
    #Plot the ROC curve
    plot_roc_curve(target_test, predicted_probs, "Random Forest")
    print("Parameters were: ", classifier.get_params())

    print("=================================================================")
    print("Optimizing Decision Tree hyper parameters")
    print("-----------------------------------------------------------------")
    param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
    clf = RandomizedSearchCV(DecisionTreeClassifier(), param_dist, cv=5, n_iter=100)
    clf.fit(data_train, target_train)
    print("Best parameters found:", clf.best_params_)
    print("-----------------------------------------------------------------")
    joblib.dump(clf.best_estimator_, 'd_tree.pkl')
    classifier= joblib.load('d_tree.pkl')
    #Get classifiers predictions for the test set
    test_predict  = classifier.fit(data_train,target_train).predict(data_test)
    predicted_probs = classifier.fit(data_train,target_train).predict_proba(data_test)[:,1]
    print(classification_report(target_test,test_predict))
    print("The accuracy score is {:.2%}".format(accuracy_score(target_test,test_predict)))
    #Plot the confusion matrix
    cnf_matrix = confusion_matrix(target_test,test_predict)
    graph_name = ("Decision Tree Confusion Matrix")
    plot_confusion_matrix(cnf_matrix, classes=['not bullying', 'bullying'],title =graph_name)
    #Plot the ROC curve
    plot_roc_curve(target_test, predicted_probs, "Decision Tree")
    print("Parameters were: ", classifier.get_params())

    print("=================================================================")
    print("Optimizing KNN hyper parameters")
    print("-----------------------------------------------------------------")
    param_grid = [ {'n_neighbors': list(range(1, 20, 2)), 'p':[1, 2, 3],  'weights':["uniform","distance"]} ]
    clf = GridSearchCV(KNeighborsClassifier(metric='euclidean'), param_grid, cv=10)
    clf.fit(data_train, target_train)
    print("Best parameters found:", clf.best_params_)
    print("-----------------------------------------------------------------")
    joblib.dump(clf.best_estimator_, 'KNN.pkl')
    classifier= joblib.load('KNN.pkl')
    #Get classifiers predictions for the test set
    test_predict  = classifier.fit(data_train,target_train).predict(data_test)
    predicted_probs = classifier.fit(data_train,target_train).predict_proba(data_test)[:,1]
    print(classification_report(target_test,test_predict))
    print("The accuracy score is {:.2%}".format(accuracy_score(target_test,test_predict)))
    
    #Plot the confusion matrix
    cnf_matrix = confusion_matrix(target_test,test_predict)
    graph_name = ("KNN Confusion Matrix")
    plot_confusion_matrix(cnf_matrix, classes=['not bullying', 'bullying'],title =graph_name)
    #Plot the ROC curve
    plot_roc_curve(target_test, predicted_probs, "KNN")
    print("Parameters were: ", classifier.get_params())
    
        
        
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