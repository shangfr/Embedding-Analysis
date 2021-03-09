# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:46:10 2021

@author: shangfr
"""
import json
import numpy as np
import pandas as pd
from sklearn import cluster, covariance, manifold
from jieba import analyse,cut
import spacy
from paddlenlp.embeddings import TokenEmbedding


class WordNet(object):
    '''
    sentence：  待分析文本
    embedding:  词向量模型选择
    cor：       相关性度量方式
    xy:         降维算法选择
    cluster:    聚类算法选择
    topK：      返回关键词的数量，重要性从高到低排序
    withWeight：是否同时返回每个关键词的权重
    allowPOS：  词性过滤，为空表示不过滤，若提供则仅返回符合词性要求的关键词
                默认为('ns', 'n', 'vn', 'v'),即仅提取地名、名词、动名词、动词
    
    '''
    def __init__(self, postdata):

        self.content=postdata['policytext']
        self.upload_path = postdata['file']
        self.topK = int(postdata['topK'])
        self.embedding = postdata['embedding']
        self.corr = postdata['cor']

        if self.content:
            self.content = self.content.replace(" ", "").replace("\r", "").replace("\n", "")
        else:
            if self.upload_path:
                file_name='test_content/'+self.upload_path.split('\\')[2]
            else:
                file_name='test_content/招商政策01.txt'
                
            with open(file_name, 'rb') as file:
                contentf = file.read()
            self.content = contentf.decode(encoding = "utf-8").replace(" ", "").replace("\r", "").replace("\n", "")
          
            
        self.keywords = self.word_cut(self.content,self.topK,self.stopwords)
        
    def __str__(self):
        return 'WordNet object (embedding=%s)' % self.embedding
    __repr__ = __str__

    
    @classmethod
    def load_model(cls):
    
        cls.wordemb1 = spacy.load('zh_core_web_sm')
        cls.wordemb2 = TokenEmbedding("w2v.baidu_encyclopedia.target.word-word.dim300")
    
        stopwords=[]
        for word in open('static/dict/chineseStopWords.txt','r', encoding='utf-8'):
            stopwords.append(word.strip())
        
        cls.stopwords = stopwords
        print('模型加载完成')

    @property
    def contents(self):
        """
        Return content
        """
        return self.content
    
    @staticmethod    
    def word_cut(content,topK,stopwords=[]):
        words_list = cut(content, cut_all=False)
         
        counts = {}             #创建计数器 --- 字典类型
        for word in words_list:      #消除同意义的词和遍历计数
            if word not in stopwords:
                counts[word] = counts.get(word,0) + 1
        
        counts = sorted(counts.items(), key=lambda item:item[1],reverse=True)
        counts = [{'name':x[0],'amount':x[1]} for x in  counts]
        filename='keywords/word_frq.json'
        with open(filename,'w', encoding='utf-8') as file_obj:
            json.dump(counts,file_obj,ensure_ascii=False)
    
        keywords = analyse.extract_tags(content,topK, withWeight=True, allowPOS=())#list   
        # 基于TextRank算法进行关键词抽取
        #textrank=analyse.textrank
        #keywords=textrank(text)#list
        
        # 输出抽取出的关键词
        #for keyword in keywords:
        #    print(keyword[0],keyword[1])#分别为关键词和相应的权重
        keydic={}
        keydic['keywords'] = [(x[0],x[1]*100) for x in keywords]
        
        filename='keywords/keywords.json'
        with open(filename,'w', encoding='utf-8') as file_obj:
            json.dump(keydic,file_obj,ensure_ascii=False)
            
        return keywords

    @staticmethod 
    def word2vec(keywords,embedding = 'Spacy'):      

        symbols = [x[0] for x in keywords]
        symbolSize = [round(x[1]*50,2) for x in keywords]

        # 利用语料训练模型
        X = []
        
        if embedding == 'Paddle':
            for string in symbols:
                doc_vector = WordNet.wordemb2.search(string)[0]
                X.append(doc_vector)
        else:  
            for string in symbols:
                doc_vector = WordNet.wordemb1(string).vector
                X.append(doc_vector)        
      
        X = np.array(X).T  
        X /= X.std(axis=0)
        
        return symbols,symbolSize,X

    @staticmethod 
    def x2cor(xarray,corr = 'pCor'):  
        # #############################################################################
        # Learn a graphical structure from the correlations
        edge_model = covariance.GraphicalLassoCV()
        edge_model.fit(xarray) 

        cov_correlations = edge_model.covariance_.copy()
        if corr == 'Cor':
            d = 1 / np.sqrt(np.diag(cov_correlations))
            non_zero = (np.abs(np.triu(cov_correlations, k=1)) > 0.5)        
        else:
            # Display a graph of the partial correlations
            partial_correlations = edge_model.precision_.copy()
            d = 1 / np.sqrt(np.diag(partial_correlations))
            partial_correlations *= d
            partial_correlations *= d[:, np.newaxis]
            non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)
            #values = np.abs(partial_correlations[non_zero])
        return non_zero,d,cov_correlations

    @staticmethod 
    def apCluster(symbols,cov_correlations):  
        # #############################################################################
        # Cluster using affinity propagation

        _, labels = cluster.affinity_propagation(cov_correlations,
                                                 random_state=0)
        n_labels = labels.max()
        categories = []
        names = np.array(symbols)
        for i in range(n_labels + 1):
            print('类别%i: %s' % ((i + 1), ', '.join(names[labels == i])))
            mydict = {}
            mydict["name"] = '类别%i' %(i + 1) 
            categories.append(mydict)
        
        return labels,categories

    @staticmethod 
    def xyDimension(xarray):  
        # use a large number of neighbors to capture the large-scale structure.
        node_position_model = manifold.LocallyLinearEmbedding(n_components=2, eigen_solver='dense', n_neighbors=5)        
        xy = node_position_model.fit_transform(xarray.T).T
        return xy
            
    def word2net(self):      

        symbols,symbolSize,X = self.word2vec(self.keywords,self.embedding)       

        non_zero,d,cov_correlations = self.x2cor(X,self.corr)    

        labels,categories = self.apCluster(symbols,cov_correlations)
        
        xy = self.xyDimension(X)
      
        df_nodes = pd.DataFrame({'id':[str(x) for x in range(self.topK)],
                                'name':symbols,'symbolSize':symbolSize,
                                'x':xy[0], 'y':xy[1],
                                'value':d,'category':labels})
        
        #df_nodes['id'] = df_nodes['id'].astype('str')       
        nodes = df_nodes.to_dict('records')

        start_idx, end_idx = np.where(non_zero)
        links = [{'source':str(start),'target':str(stop)} for start, stop in zip(start_idx, end_idx)]
    
        
        json_data = {}        
        json_data['categories'] = categories
        json_data['nodes'] = nodes
        json_data['links'] = links
        
        return json_data





