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
import scikitplot as skplt
import xml.etree.ElementTree as ET
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

#Function to plot the ROC curve
def plot_roc_curve(target_test, predicted, name):
    fpr, tpr, threshold = metrics.roc_curve(target_test, predicted, pos_label='Yes')
    roc_auc = metrics.auc(fpr, tpr)
#==============================================================================
#     #This is only used when obtaining raw data for the report, otherwise use the plot
#     print("---------------------------\nROC Curve\n---------------------------")
#     print("Area under the curve: ", roc_auc)
#     print("fpr: ", fpr)
#     print("tpr: ", tpr)
#     print("threshold: ", threshold)
#==============================================================================
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
    
    #read CSV into a dataframe
    consensus = pd.read_csv("../datasets/bayzick_data/human_consensus.csv")
    consensus = consensus[pd.notnull(consensus["Is Cyberbullying Present?"])]
    print(consensus['Is Cyberbullying Present?'].value_counts())
    
    #use info in consensus to find relevant xml files and pull data from them
    for index, row in consensus.iterrows():
        my_string = ""
        filename = "../datasets/bayzick_data/xml_files/" + str("%.4f" % row["File Name"]) + ".xml"
        # Open up the xml file with the corresponding filename and add body text to data array
        tree = ET.parse(filename)
        root = tree.getroot()
        
        for post in root.findall('post'):
            body = str(post.find('body').text)
            my_string = my_string + " " + body
        
        data.append(my_string)
        target.append(row['Is Cyberbullying Present?'])
        
    return data, target

def preprocess():
    
    # Initialise Porter Stemmer & Lemmatization
    ps = PorterStemmer()
    lem = WordNetLemmatizer()
    # Create a set to hold stopwords that we don't want from NLTK
    stop_words = set(stopwords.words('english'))
    
    #Load in the file/files
    data,target = load_file()
    
    #Various methods of preprocessing, comment or uncomment to get various combinations
    
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
#==============================================================================
#     #Stop word removal
#     data = [remove_stop_words(element, stop_words) for element in data]
#==============================================================================

    #Turn on TF-IDF
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    tfidf_data = tfidf.fit_transform(data)
    
#==============================================================================
#     #Turn off TF-IDF
#     count_vectorizer = CountVectorizer(binary='true')
#     data = count_vectorizer.fit_transform(data)
#     tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)
#==============================================================================

    return tfidf_data, target

def train_eval(data,target):
    
    #Array to hold the names of the classifiers we are going to test
    names = ["KNN", "SVC", "Decision Tree", "Random Forest", "Bernoulli Naive Bayes"]
    
    #Array holding the actual classifiers
    classifiers = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(random_state=107),
    RandomForestClassifier(),
    BernoulliNB()]
    
    #Split the data to have separate test data
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
        #Plot precision recall curve
        probas = clf.predict_proba(data_test)
        skplt.metrics.plot_precision_recall_curve(target_test, probas, title=name+" Precision Recall Curve", cmap="hot")
        plt.show()
        cnf_matrix = confusion_matrix(target_test,predicted)
#==============================================================================
#         #This is only used when obtaining raw data for the report, otherwise use the plot
#         print("---------------------------\nConfusion Matrix\n---------------------------")
#         print(cnf_matrix)
#==============================================================================
        #Plot the confusion matrix
        graph_name = (name, "Confusion Matrix")
        plot_confusion_matrix(cnf_matrix, classes=['not bullying', 'bullying'],title =graph_name)
        #Plot the ROC curve
        plot_roc_curve(target_test, predicted_probs, name)
        print("Parameters were: ", clf.get_params())
    optimized_hyper_parameters(data_train,data_test,target_train,target_test)
        
def optimized_hyper_parameters(data_train,data_test,target_train,target_test):
    
    print("=================================================================")
    print("Optimized SVC hyper parameters")
    print("-----------------------------------------------------------------")
    classifier= joblib.load('SVC.pkl')
    #Get classifiers predictions for the test set
    test_predict  = classifier.fit(data_train,target_train).predict(data_test)
    predicted_probs = classifier.fit(data_train,target_train).predict_proba(data_test)[:,1]
    print(classification_report(target_test,test_predict))
    print("The accuracy score is {:.2%}".format(accuracy_score(target_test,test_predict)))
    #Plot precision recall curve
    probas = classifier.predict_proba(data_test)
    skplt.metrics.plot_precision_recall_curve(target_test, probas, title="SVC Precision Recall Curve", cmap="hot")
    plt.show()
    #Plot the confusion matrix
    cnf_matrix = confusion_matrix(target_test,test_predict)
#==============================================================================
#     #This is only used when obtaining raw data for the report, otherwise use the plot
#     print("---------------------------\nConfusion Matrix\n---------------------------")
#     print(cnf_matrix)
#==============================================================================
    graph_name = ("SVC Confusion Matrix")
    plot_confusion_matrix(cnf_matrix, classes=['not bullying', 'bullying'],title =graph_name)
    #Plot the ROC curve
    plot_roc_curve(target_test, predicted_probs, "SVC")
    print("Parameters were: ", classifier.get_params())
    
    print("=================================================================")
    print("Optimized Random Forest hyper parameters")
    print("-----------------------------------------------------------------")
    classifier= joblib.load('random_forest.pkl')
    #Get classifiers predictions for the test set
    test_predict  = classifier.fit(data_train,target_train).predict(data_test)
    predicted_probs = classifier.fit(data_train,target_train).predict_proba(data_test)[:,1]
    print(classification_report(target_test,test_predict))
    print("The accuracy score is {:.2%}".format(accuracy_score(target_test,test_predict)))
    #Plot precision recall curve
    probas = classifier.predict_proba(data_test)
    skplt.metrics.plot_precision_recall_curve(target_test, probas, title="Random Forest Precision Recall Curve", cmap="hot")
    plt.show()
    #Plot the confusion matrix
    cnf_matrix = confusion_matrix(target_test,test_predict)
#==============================================================================
#     #This is only used when obtaining raw data for the report, otherwise use the plot
#     print("---------------------------\nConfusion Matrix\n---------------------------")
#     print(cnf_matrix)
#==============================================================================
    graph_name = ("Random Forest Confusion Matrix")
    plot_confusion_matrix(cnf_matrix, classes=['not bullying', 'bullying'],title =graph_name)
    #Plot the ROC curve
    plot_roc_curve(target_test, predicted_probs, "Random Forest")
    print("Parameters were: ", classifier.get_params())

    print("=================================================================")
    print("Optimized Decision Tree hyper parameters")
    print("-----------------------------------------------------------------")
    classifier= joblib.load('d_tree.pkl')
    #Get classifiers predictions for the test set
    test_predict  = classifier.fit(data_train,target_train).predict(data_test)
    predicted_probs = classifier.fit(data_train,target_train).predict_proba(data_test)[:,1]
    print(classification_report(target_test,test_predict))
    print("The accuracy score is {:.2%}".format(accuracy_score(target_test,test_predict)))
    #Plot precision recall curve
    probas = classifier.predict_proba(data_test)
    skplt.metrics.plot_precision_recall_curve(target_test, probas, title="Decision Tree Precision Recall Curve", cmap="hot")
    plt.show()
    #Plot the confusion matrix
    cnf_matrix = confusion_matrix(target_test,test_predict)
#==============================================================================
#     #This is only used when obtaining raw data for the report, otherwise use the plot
#     print("---------------------------\nConfusion Matrix\n---------------------------")
#     print(cnf_matrix)
#==============================================================================
    graph_name = ("Decision Tree Confusion Matrix")
    plot_confusion_matrix(cnf_matrix, classes=['not bullying', 'bullying'],title =graph_name)
    #Plot the ROC curve
    plot_roc_curve(target_test, predicted_probs, "Decision Tree")
    print("Parameters were: ", classifier.get_params())

    print("=================================================================")
    print("Optimized KNN hyper parameters")
    print("-----------------------------------------------------------------")
    classifier= joblib.load('KNN.pkl')
    #Get classifiers predictions for the test set
    test_predict  = classifier.fit(data_train,target_train).predict(data_test)
    predicted_probs = classifier.fit(data_train,target_train).predict_proba(data_test)[:,1]
    print(classification_report(target_test,test_predict))
    print("The accuracy score is {:.2%}".format(accuracy_score(target_test,test_predict)))
    #Plot precision recall curve
    probas = classifier.predict_proba(data_test)
    skplt.metrics.plot_precision_recall_curve(target_test, probas, title="KNN Precision Recall Curve", cmap="hot")
    plt.show()
    #Plot the confusion matrix
    cnf_matrix = confusion_matrix(target_test,test_predict)
#==============================================================================
#     #This is only used when obtaining raw data for the report, otherwise use the plot
#     print("---------------------------\nConfusion Matrix\n---------------------------")
#     print(cnf_matrix)
#==============================================================================
    graph_name = ("KNN Confusion Matrix")
    plot_confusion_matrix(cnf_matrix, classes=['not bullying', 'bullying'],title =graph_name)
    #Plot the ROC curve
    plot_roc_curve(target_test, predicted_probs, "KNN")
    print("Parameters were: ", classifier.get_params())
    

def main():
    
    #Set background colour for plots
    plt.rcParams['axes.facecolor'] = '#F9FFFD'
    #Get a start time for the program
    start_time = time.time()
    #load and preprocess datasets
    tf_idf, target = preprocess()
    #train classifiers on datasets & hyperoptimize
    train_eval(tf_idf,target)
    
    #Get the time taken in seconds
    print("\n=================================================================")
    print("Program ran in %s seconds" % (time.time() - start_time))
    print("=================================================================\n")
    
main()