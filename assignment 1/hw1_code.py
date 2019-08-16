
# coding: utf-8

# In[240]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from math import log


# In[241]:


def load_data():
    dataset_clean_real = pd.read_csv('clean_real.txt',header= None)
    print ("Dataset clean_real Lenght:: ", len(dataset_clean_real))
    print ("Dataset clean_real Shape:: ", dataset_clean_real.shape)
    dataset_clean_fake = pd.read_csv('clean_fake.txt',header= None)
    print ("Dataset clean_fake Lenght:: ", len(dataset_clean_fake))
    print ("Dataset clean_fake Shape:: ", dataset_clean_fake.shape)
    clean_real=[]
    labels=[]
    with open('clean_real.txt','r') as f:
        for line in f:
            
            clean_real.append(line) 
            labels.append(1)
    clean_fake=[]
    with open('clean_fake.txt','r') as f:
        for line in f:
            
            clean_fake.append(line)   
            labels.append(0)
    corpus_real_fake=clean_real+clean_fake
 
    vectorizer = CountVectorizer()
    vec_real_fake = vectorizer.fit_transform(corpus_real_fake)
    
    feature_real_fake=vectorizer.get_feature_names()
    array_real_fake=(vec_real_fake.toarray()) 
    
    labels = np.array(labels)
    
    labels = np.reshape(labels,(labels.shape[0],1))
   
    data=np.concatenate((array_real_fake,labels),axis=1)
    np.random.shuffle(data)
    
    #Training data
    n = data.shape[0]
    train_X=data[0:int(0.7*n),:-1]
    train_Y=data[0:int(0.7*n),-1]
    #Validation data
    vad_X=data[int(0.7*n):int(0.85*n),:-1]
    vad_Y=data[int(0.7*n):int(0.85*n),-1]
    # Test data
    test_X=data[int(0.85*n):n,:-1]
    test_Y=data[int(0.85*n):n,-1]
    return train_X,train_Y,vad_X,vad_Y,test_X,test_Y
    
    


# In[199]:


def select_model():
    train_X,train_Y,vad_X,vad_Y,test_X,test_Y= load_data()
    #Decision Tree Classifier using 5 different values of max_depth and criterion gini coefficient
    max_depth=[15,30,45,20,50] 
    for i in range (0,5):
        clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=max_depth[i])
        clf_gini.fit(train_X,train_Y)
        predict_gini_Y = clf_gini.predict(vad_X)
        Acc=0.0
        for j in range(0,vad_Y.shape[0]):
           if predict_gini_Y[j]==vad_Y[j]:
               Acc+=1
        Acc/=vad_Y.shape[0]
        print ("Accuracy for split criteria gini coefficient and max depth",max_depth[i],"is",Acc*100,"%")
    #Decision Tree Classifier using 5 different values of max_depth and criterion gini coefficient
    for i in range (0,5):
        clf_ig = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=max_depth[i])
        clf_ig.fit(train_X,train_Y)
        predict_ig_Y= clf_ig.predict(vad_X)
        Acc=0.0
        for j in range(0,vad_Y.shape[0]):
           if predict_ig_Y[j]==vad_Y[j]:
               Acc+=1
        Acc/=vad_Y.shape[0]
        print ("Accuracy for split criteria information gain and max depth",max_depth[i],"is",Acc*100,"%")
select_model()        


# In[243]:


def display_model():
    train_X,train_Y,vad_X,vad_Y,test_X,test_Y= load_data()
    besttree = DecisionTreeClassifier(criterion = "entropy",random_state = 100,max_depth=50)
    besttree = besttree.fit(train_X,train_Y)
    export_graphviz(besttree,max_depth=2, out_file ='decision_tree.dot',class_names=['Fake','Real'])
display_model()


# In[242]:


def compute_information_gain():
    def entropy(p_i):
        H= 0
        for i in p_i:
            i = i / sum(p_i)
            if i != 0:
                H += i * log(i, 2)
            else:
                H+= 0
        H*= -1
        return H
    def informationgain(Y, x_i):
        H= 0
        for v in x_i:
            H+= sum(v) / sum(Y) * entropy(v)
        gain = entropy(Y) - H
        return gain
    # giving values to calculate informationgain for topmost split(X[1598]) in previous question
    S = [890, 1396]
    topmost = [[746, 818],[144,578]]
    print("Information gain for topmost split in the decision tree of previous part is", informationgain(S, topmost))
    
    #Other Keywords
    # giving values to calculate informationgain for topmost split(X[5143]) in different decision tree which was generated again
    S_K1=[915, 1371]
    topmost_K1 = [[648, 1258],[267, 113]]
    print("Information gain for topmost split in the decision tree which was generated again is", informationgain(S_K1, topmost_K1))
    
    # giving values to calculate informationgain for topmost split(X[5143]) in different decision tree which was generated again
    S_K2=[916, 1370]
    topmost_K2 = [[760, 799],[156, 571]]
    print("Information gain for topmost split in the decision tree which was generated again is", informationgain(S_K2, topmost_K2))

compute_information_gain()

