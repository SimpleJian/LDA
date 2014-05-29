#!/usr/bin/env python
#coding:utf-8

'''
function:corpus preprocess
date:2014-05-29
author:jwchen
'''

import nltk,os,csv

def get_folder_list(path):
    folderlist = os.listdir(path)
    return folderlist

def get_file_list(path,folder):
    file_path = path+'/'+folder
    file_list = os.listdir(file_path)
    return file_list

    

if __name__=="__main__":
    corpus_dir = "./data/corpus"
    folder_list = get_folder_list(corpus_dir)
    g=open('./data/test.csv','wb')
    writer = csv.writer(g)
    for folder in folder_list:
        print '遍历%s目录下的文件' % folder
        file_list = get_file_list(corpus_dir,folder)
        for file in file_list:
            file_path = corpus_dir +'/'+folder+'/'+file
            f = open(file_path,'rb')
            context = f.read()
            words = nltk.word_tokenize(context)
            words = [w.lower() for w in words if w.isalpha()] 
            writer.writerow(words)
            f.close()
    g.close()
    
