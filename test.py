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
    

import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("zh_core_web_sm")

# Process whole documents
text = ("spaCy是世界上最快的工业级自然语言处理工具。 支持多种自然语言处理基本功能。官网地址spaCy主要功能包括分词、词性标注、词干化、命名实体识别、名词短语提取等等。")
doc = nlp(text)
doc.vector

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)
    
    
for token in doc:
    print(token.vector)    
    
   
    
doc = nlp('狗 猫 香蕉')

for token1 in doc:
    for token2 in doc:
        print(token1.similarity(token2))
    
    
    
    
import jiagu

#jiagu.init() # 可手动初始化，也可以动态初始化

text = '小明早上8点去学校上课。'

words = jiagu.seg(text) # 分词
print(words)

pos = jiagu.pos(words) # 词性标注
print(pos)

ner = jiagu.ner(words) # 命名实体识别
print(ner)    
    
    
    
    
    
    
    
    
    
from paddlenlp.datasets import ChnSentiCorp

train_ds, dev_ds, test_ds = ChnSentiCorp.get_datasets(['train', 'dev', 'test'])
    
    
from paddlenlp.embeddings import TokenEmbedding

wordemb = TokenEmbedding("w2v.baidu_encyclopedia.target.word-word.dim300")
print(wordemb.cosine_sim("苹果", "香蕉"))

wordemb.cosine_sim("艺术", "火车")

wordemb.cosine_sim("狗", "香蕉")
    
for token1 in ['狗','猫','香蕉']:
    for token2 in ['狗','猫','香蕉']:
        print(wordemb.cosine_sim(token1, token2))
    
vv = wordemb.search(['狗','猫','香蕉'])



vv2 = wordemb.search('狗猫香蕉')

