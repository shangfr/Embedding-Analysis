#  NLP分析工具

**最新的项目**：[embedding-projector-zh](https://github.com/shangfr/embedding-projector-zh)基于交互式Web应用程序搭建一个可以进行高维数据分析的系统。

**NLP分析工具**是一款基于NLP开源算法和模型库（jieba、spacy、paddlenlp）对文本数据进行向量化，然后通过机器学习算法（聚类、主成分分析、图网络GraphicalLasso）对文本数据词向量之间进行关联性分析的小工具。前后端开发上用到了flask_api+js+bootstrap+echarts等组件，小工具涉及参数如下：

- **sentence** ：待分析文本，可编辑或上传txt文件
- **embedding** ：词向量模型选择（开源的词向量库）
- **cor** ：相关性度量方式（协方差矩阵、精度矩阵（偏相关））
- **xy** ：降维算法选择 ，Locally linear embedding（LLE）是一种非线性降维算法，它能够使降维后的数据较好地保持原有流形结构。
- **cluster** ：聚类算法选择，AP（非约束簇）- 与kmeans相比，不需要指定k值
- **topK** ：分析关键词的数量，默认20个，重要性从高到低排序
- **withWeight** ：每个关键词的权重


![avatar](/static/picture/pic01.jpg)

![avatar](/static/picture/pic02.jpg)

![avatar](/static/picture/pic03.jpg)


### 代码块
``` python

from word_net import WordNet
# 载入词向量模型
WordNet.load_model()
# 待分析文本
content = ''

# 分词，获取关键词
keywords = WordNet.word_cut(content,10)

# 词向量化
symbols,symbolSize,X = WordNet.word2vec(keywords,embedding = 'Spacy')
# 词向量矩阵转换--相关系数矩阵---偏相关系数矩阵---获取边
non_zero,d,cov_correlations = WordNet.x2cor(X)    
# 相关系数矩阵-聚类-获取类别标签
labels,categories = WordNet.apCluster(symbols,cov_correlations)

# 词向量矩阵-降维-可视化
xy = WordNet.xyDimension(X)
  
```
