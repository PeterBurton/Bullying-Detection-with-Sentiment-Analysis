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
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import itertools

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
    
    consensus = pd.read_csv("../datasets/bayzick_data/human_consensus.csv")
    consensus = consensus[pd.notnull(consensus["Is Cyberbullying Present?"])]
    print(consensus['Is Cyberbullying Present?'].value_counts())
    
    for index, row in consensus.iterrows():
        my_string = ""
        filename = "../datasets/bayzick_data/xml_files/" + str("%.4f" % row["File Name"]) + ".xml"
        # Open up the xml file with the corresponding filename and 
        tree = ET.parse(filename)
        root = tree.getroot()
        
        for post in root.findall('post'):
            body = str(post.find('body').text)
            my_string = my_string + " " + body
        
        data.append(my_string)
        target.append(row['Is Cyberbullying Present?'])
        
    return data, target

def preprocess():
    
    data,target = load_file()
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
    train_eval(tf_idf, target)
    
    
    #Get the time taken in seconds
    print("\n=================================================================")
    print("Program ran in %s seconds" % (time.time() - start_time))
    print("=================================================================\n")
    
main()