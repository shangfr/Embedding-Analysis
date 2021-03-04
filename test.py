# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:47:18 2021

@author: shangfr
"""
import json
import numpy as np 
 
keyarr = np.load("flask_api/keywords/keywords.npy")  
keydic={}
keydic['keywords']=keyarr.tolist()
keyJson = json.dumps(keydic)


import jieba
words_list = jieba.cut("我来到北京背景北京北京清华大学", cut_all=False)
counts = {}             #创建计数器 --- 字典类型
for word in words_list:      #消除同意义的词和遍历计数
    counts[word] = counts.get(word,0) + 1

counts = sorted(counts.items(), key=lambda item:item[1],reverse=True)


counts = [{'name':key,'amount':values} for key,values in  counts.items()]
filename='flask_api/keywords/word_frq.json'
with open(filename,'w', encoding='utf-8') as file_obj:
    json.dump(counts,file_obj,ensure_ascii=False)
    

