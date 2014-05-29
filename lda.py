#!/usr/bin/env python
#coding:utf-8

'''
function:LDA with Gibbs sampling
author:jwchen
date:2014-05-29
'''
import random,os

alpha = 0.1
beta = 0.1
K  = 10
iter_num = 50
top_words = 20


corpus_text = './data/test.csv'

class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0

class Dataset(object):
    def __init__(self):
        self.M = 0
        self.V = 0
        self.docs = []
        self.word2id = {} #(word,word_id)dict
        self.id2word = {} #(word_id,word)dict
    
class Model(object):
    def __init__(self,dataset):
        self.dataset = dataset
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iter_num = iter_num
        self.top_words = top_words

        self.p = []     #double,store the temporary variables
        self.zmn = []   #theme dis of every word n in every document
        self.nmk = []   #the count for theme k in document m
        self.nmsum = [] #sum for theme in document m 
        self.nkt = []   #the count for word t  in  theme k
        self.nksum = [] #sum for word in theme k (all theme)
        self.theta = [] #dis of document-theme
        self.phi = []   #dis of theme-word 

    def init_par(self):
        self.p = [0.0 for x in range(self.K)]
        self.nkt = [[0 for y in range(self.K)] for x in range(self.dataset.V)]
        self.nksum = [0 for x in range(self.K)]
        self.nmk = [[0 for y in range(self.K)] for x in range(self.dataset.M)]
        self.nmsum = [0 for x in range(self.dataset.M)]
        self.zmn = [ [] for x in range(self.dataset.M)]
        for x in range(self.dataset.M):
            self.zmn[x] = [0 for y in range(self.dataset.docs[x].length)]
            for y in range(self.dataset.docs[x].length):
                topic = random.randint(0,self.K-1)
                self.zmn[x][y] = topic
                '''
                nmk 文档-主题矩阵,表示文档m主题k的个数
                nmsum 文档-主题总矩阵,表示文档m的总主题数(等于词项数)
                nkt 词项-主题分布,表示此项t主题k的个数
                nksum 词项-主题分布矩阵,表示所有词项的主题数
                '''
                self.nkt[self.dataset.docs[x].words[y]][topic] += 1
                self.nksum[topic] += 1
                self.nmk[x][topic] += 1
                self.nmsum[x] += 1
        self.theta = [[0.0 for y in range(self.K)] for x in range(self.dataset.M)]
        self.phi = [[0.0 for y in range(self.dataset.V)] for x in range(self.K)]
            
    def estimate(self):
        print 'Sampling %d iterations' % self.iter_num
        for x in range(self.iter_num):
            print 'Iteration %d ...' % (x+1)
            for i in range(len(self.dataset.docs)):
                for j in range(self.dataset.docs[i].length):
                    topic = self.gibbs_sampling(i,j)
                    self.zmn[i][j] = topic
        print 'End Sampling...'
        print 'Compute theta...'
        self.compute_theta()
        print 'Compute phi...'
        self.compute_phi()
        print 'Saving model...'
        self.save_model()
        
    def gibbs_sampling(self,i,j):
        topic = self.zmn[i][j]
        wordid = self.dataset.docs[i].words[j]
        self.nkt[wordid][topic] -= 1
        self.nksum[topic] -= 1
        self.nmk[i][topic] -=1
        self.nmsum[i] -= 1

        Vbeta = self.dataset.V*self.beta
        Kalpha = self.K*self.alpha

        for k in range(self.K):
            self.p[k] = (self.nkt[wordid][k]+self.beta)/(self.nksum[k]+Vbeta)*(self.nmk[i][k]+alpha)/(self.nmsum[i]+Kalpha)
            
        for k in range(1,self.K):
            self.p[k] += self.p[k-1]
        u = random.uniform(0,self.p[self.K-1])
        for topic in range(self.K):
            if self.p[topic] > u:
                break
        self.nkt[wordid][topic] += 1
        self.nksum[topic] += 1
        self.nmk[i][topic] +=1
        self.nmsum[i] += 1
        return topic

    def compute_theta(self):
        for x in range(self.dataset.M):
            for y in range(self.K):
                self.theta[x][y] = (self.nmk[x][y]+self.alpha)/(self.nmsum[x]+self.K*self.alpha)
                
    def compute_phi(self):
        for x in range(self.K):
            for y in range(self.dataset.V):
                self.phi[x][y] = (self.nkt[y][x]+self.beta)/(self.nksum[x]+self.dataset.V*self.beta)
                
    def save_model(self):
        f = open('./data/topic_word.csv','wb')
        if self.top_words > self.dataset.V:
            self.top_words = self.dataset.V
        for x in range(self.K):
            f.write('Topic '+str(x)+'th:\n')
            topic_words = []
            for y in range(self.dataset.V):
                topic_words.append((y,self.phi[x][y]))
                
            topic_words.sort(key=lambda x:x[1],reverse = True)
            for y in range(self.top_words):
                word = self.dataset.id2word[topic_words[y][0]]
                f.write('\t'+word+'\t'+str(topic_words[y][1])+'\n')

            
def readfile():
    print 'Read corpus data'
    f = open(corpus_text,'rb')
    corpus = f.readlines()
    
    dataset = Dataset()
    wordid = 0
    for line in corpus:
        line =line.replace('\r\n','')
        words = line.split(',')
        doc = Document()
        for word in words:
            if dataset.word2id.has_key(word):
                doc.words.append(dataset.word2id[word])
            else:
                dataset.word2id[word] = wordid
                dataset.id2word[wordid] = word
                doc.words.append(wordid)
                wordid += 1
        doc.length = len(words)
        dataset.docs.append(doc)
    dataset.M = len(dataset.docs)
    dataset.V = len(dataset.word2id)
    print 'There are %d document'%dataset.M
    print 'There are %d words'%dataset.V
    return dataset

if __name__=="__main__":
    dataset = readfile()
    model = Model(dataset)
    model.init_par()
    model.estimate()
    

