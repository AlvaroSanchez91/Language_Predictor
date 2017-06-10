# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 02:42:00 2017

@author: AlvaroSanchez91
"""
import pandas as pd
import nltk
import numpy as np
import json
import random
import time
import pickle

#Lets construct the clasifier with all the corpus.

#We will need the next list, becouse we have a lot of languages.
languages=['bg','cs','da','de','el','es','et','fi','fr','hu','it','lv','nl','pl','pt','ro','sk','sl','sv']
l_order_inverse=['bg','cs','da','de','el']

#DONT DO THIS
#This save the complete files, we need only one part.
for lgj in languages:
    phrases=[]
    open_file="D:\\corpus_PLN1\\{}-en.txt".format(lgj)
    save_file="D:\\corpus_PLN1\\prueba_loop_borrar\\phrases_{}.txt".format(lgj)
    for i,line in enumerate( open(open_file,encoding="utf8")):
        phrases.append(line[0:-1])
    phrases=pd.Series(phrases)
    phrases.to_csv(save_file,encoding="utf8")


#DONT DO THIS
#With this, we can save the part with the target language.   
for lgj in languages:
    phrases=[]
    open_file="D:\\corpus_PLN1\\{}-en.txt".format(lgj)
    save_file="D:\\corpus_PLN1\\prueba_loop_borrar\\phrases_{}.txt".format(lgj)
    lenght=0
    for i,line in enumerate( open(open_file,encoding="utf8")):
        pass
    lenght=i
    if lgj not in ['bg','cs','da','de','el']:
        for i,line in enumerate( open(open_file,encoding="utf8")):
            if (lenght/2 < i):
                phrases.append(line[0:-1])
        phrases=pd.Series(phrases)
        phrases.to_csv(save_file,encoding="utf8")
    else:
        for i,line in enumerate( open(open_file,encoding="utf8")):
            if (lenght/2 > i):
                phrases.append(line[0:-1])
        phrases=pd.Series(phrases)
        phrases.to_csv(save_file,encoding="utf8")

        
#DONT DO THIS
#This can be used, but it's done in the code above.
for lgj in ['bg','cs','da','de','el']:
    phrases=[]
    open_file="D:\\corpus_PLN1\\{}-en.txt".format(lgj)
    save_file="D:\\corpus_PLN1\\prueba_loop_borrar\\phrases_{}.txt".format(lgj)
    lenght=0
    for i,line in enumerate( open(open_file,encoding="utf8")):
        pass
    lenght=i
    for i,line in enumerate( open(open_file,encoding="utf8")):
        if (lenght/2 > i):
            phrases.append(line[0:-1])
    phrases=pd.Series(phrases)
    phrases.to_csv(save_file,encoding="utf8")        
    

#POSILBLE START PONIT:
#We need to construct our train and our test data.
#First, we do in each file, so we will have more or less 1000 instances for language.
pr_all=[]
for lgj in languages:
    pr_all.append(pd.read_csv("D:\\corpus_PLN1\\prueba_loop_borrar\\phrases_{}.txt".format(lgj),header=None,encoding="utf8"))

id_train_all=[random.sample(list(range(len(pr_all[i][1]))), 1000) for i in range(len(languages))]

id_test_all=[list(set(range(len(pr_all[i][1])))-set(id_train_all[i])) for i in range(len(languages))]

test_all=[list(pr_all[i][1][id_test_all[i]].dropna()) for i in range(len(languages))]

train_all=[list(pr_all[i][1][id_train_all[i]].dropna()) for i in range(len(languages))]

#Now, we construct only one file for all de train, and another one for the test.

#We also, need to save english prhases on train and test.
pr_en=phrases=pd.read_csv("D:\\corpus_PLN1\\phrases_en.txt",header=None,encoding="utf8")

id_train_en=random.sample(list(range(len(pr_en))), 1000)

id_test_en=list(set(range(len(pr_en)))-set(id_train_en))

test_en=list(pr_en[1][id_test_en].dropna())

train_en=list(pr_en[1][id_train_en].dropna())

train=[]
test=[]

for p in train_en:
    train.append([p,'en'])

for p in test_en:
    test.append([p,'en'])


for j,tr in enumerate(train_all):
    for p in tr:
        train.append([p,languages[j]])

for j,tr in enumerate(test_all):
    for p in tr:
        test.append([p,languages[j]])
        


del (pr_en)
del (pr_all)

del (id_train_en)
del (id_train_all)

del (id_test_en)
del (id_test_all)

del (test_en)

del (train_en)


#We have need it a lot of time untill now
#So we can save the train and the test with the following code.


with open("D:\\corpus_PLN1\\prueba_loop_borrar\\lista_train2.txt", 'wb') as fp:
    pickle.dump(train, fp)
    
with open("D:\\corpus_PLN1\\prueba_loop_borrar\\lista_test2.txt", 'wb') as fp:
    pickle.dump(test, fp)


#POSIBLE START POINT:
with open ("D:\\corpus_PLN1\\prueba_loop_borrar\\lista_train2.txt", 'rb') as fp:
    train = pickle.load(fp)
    
with open ("D:\\corpus_PLN1\\prueba_loop_borrar\\lista_test2.txt", 'rb') as fp:
    test = pickle.load(fp)

#We save a subset of test.  
test3=[ test[i] for i in random.sample(list(range(len(test))), 10000)]
with open("D:\\corpus_PLN1\\prueba_loop_borrar\\lista_test3.txt", 'wb') as fp:
    pickle.dump(test3, fp)
    
#The next list of languages is diferent than the first, here we inclde 'en'.
languages=['en','bg','cs','da','de','el','es','et','fi','fr','hu','it','lv','nl','pl','pt','ro','sk','sl','sv']


                
            
class Rosetta:    
    def __init__(self, words,languages, method='sum'):            
        #We save a count of words in a dictionary.
        self.words = words
        self.languages=languages
        self.method=method
             
    def predict(self, p, method=None):
        if method==None:
            method=self.method
        
        if method=='sum':
            return (self.predict_sum(p))
        if method=='prob':
            return (self.predict_prob(p))
        if method=='abs':
            return (self.predict_abs(p))        
        
    def predict_prob(self,p):
        #This method multiply the probs of each word.
        #predict_sum works better (problems with words with prob almost zero) 
        n_lgj=len(self.languages)
        w_list= nltk.word_tokenize(p)
        prob=[1 for i in range (n_lgj)]
        for w in w_list:
            if [x  for x in w if x in '.,123456789']==[]:
                for w2 in self.words:
                    if w==w2:
                        prob_w=[min(x / sum(self.words[w2]), 0.01) for x in self.words[w2]]
                        prob=[prob_w[i]*prob[i] for i in range(n_lgj)]
        return self.languages[prob.index(max(prob))]
                            
    def predict_sum(self,p):
        #This method sum the probs of each word.
        n_lgj=len(self.languages)
        w_list= nltk.word_tokenize(p)
        prob=[1 for i in range (n_lgj)]
        for w in w_list:
            if [x  for x in w if x in '.,123456789']==[]:
                for w2 in self.words:
                    if w==w2:
                        prob_w=[x / sum(self.words[w2]) for x in self.words[w2]]
                        prob=[prob_w[i]+prob[i] for i in range(n_lgj)]
        return self.languages[prob.index(max(prob))]

    def predict_abs(self,p):
        #This method works using the absolute frequency..
        n_lgj=len(self.languages)
        w_list= nltk.word_tokenize(p)
        prob=[1 for i in range (n_lgj)]
        for w in w_list:
            if [x  for x in w if x in '.,123456789']==[]:
                for w2 in self.words:
                    if w==w2:
                        prob_w=self.words[w2]
                        prob=[prob_w[i]+prob[i] for i in range(n_lgj)]
        return self.languages[prob.index(max(prob))]
    
                            
    def predict_retro(self,p, method=None):
        if method==None:
            method=self.method
        #This method sum the probs of each word.
        #Also, we tray to improve te model when we predict.
        
        #In this block we predict.
        n_lgj=len(self.languages)
        w_list= nltk.word_tokenize(p)
        prob=[1 for i in range (n_lgj)]
        pred=self.predict(p, method=method)
        pos_lgj=self.languages.index(pred)
        
        #In this block we 'train'.
        for w in w_list:
            if [x  for x in w if x in '.,123456789']==[]:
                if w not in (self.words).keys():
                    self.words [w]=[0 if i!= pos_lgj else 1 for i in range(n_lgj)]
                else:
                    self.words [w][pos_lgj]+=1
        
        return self.languages[prob.index(max(prob))] 

    def predict_by_phrases(self,t,method=None):
        #We have a text t, we do the mean the predictions of each prhase in t.
        if method==None:
            method=self.method
        p_list=nltk.sent_tokenize(t)
        ph_list=[]
        for p in p_list:
            ph_list=ph_list+[x   for x in p.split('"') if x != '']
        pred_list=[]
        for p in ph_list:
            pred_list.append(self.predict(p,method=method))
        d=nltk.FreqDist(pred_list)
        return (max(d, key=d.get),[[i,j]for i,j in zip (ph_list,pred_list)])
         
    def pseudo_train(self,data):
        n_lgj=len(self.languages)
        for l in data:
            for w in nltk.word_tokenize(l[0]):
                pos_lgj=self.languages.index(l[1])
                if w not in (self.words).keys():
                    self.words [w]=[0 if i!= pos_lgj else 1 for i in range(n_lgj)]
                else:
                    self.words [w][pos_lgj]+=1
      
    def train(self, data,a=1, method=None):
        #a is the number wich is added or substracted in the dictionary words
        start_time = time.time()
        if method==None:
            method=self.method
        n_lgj=len(self.languages)
        for l in data:
            real_ln=l[1]
            pos_lgj=self.languages.index(real_ln)
            pred_ln=self.predict(l[0], method=method)
            if real_ln != pred_ln:
                for w in nltk.word_tokenize(l[0]):
            
                    if w not in (self.words).keys():
                        self.words [w]=[0 if i!= pos_lgj else a for i in range(n_lgj)]
                    else:
                        self.words [w][pos_lgj]+= 2*a
                        for i in range(n_lgj):
                            if self.words [w][i] >= a:
                                self.words [w][i]+= -a
                            else:
                                self.words [w][i]=0
                                
            else:
                for w in nltk.word_tokenize(l[0]):
            
                    if w not in (self.words).keys():
                        self.words [w]=[0 if i!= pos_lgj else a for i in range(n_lgj)]
                    else:
                        self.words [w][pos_lgj]+= a
        print('Se han tardado ',(time.time() - start_time)/60,' minutos.')
        

    def test_mc(self, data ,method=None):
        start_time = time.time()
        if method==None:
            method=self.method
        n_lgj=len(self.languages)
        mc=[[0 for i in range(n_lgj)] for i in range(n_lgj)]
        for l in data:
            real_ln=l[1]
            pred_ln=self.predict(l[0],method=method)
            mc[self.languages.index(real_ln)][self.languages.index(pred_ln)] +=1
        print('Se han tardado ',(time.time() - start_time)/60,' minutos.')
        return (mc)
    
        
    def test_mc_by_phrases(self, data,method=None):
        start_time = time.time()
        if method==None:
            method=self.method
        start_time = time.time()
        n_lgj=len(self.languages)
        mc=[[0 for i in range(n_lgj)] for i in range(n_lgj)]
        for l in data:
            real_ln=l[1]
            pred_ln=self.predict_by_phrases(l[0], method=method)[0]
            mc[self.languages.index(real_ln)][self.languages.index(pred_ln)] +=1
        print('Se han tardado ',(time.time() - start_time)/60,' minutos.')
        return (mc)

