'''
用python实现朴素贝叶斯算法，并对输入的句子进行分类。
'''
import numpy as np


#设置样本数据
def load_dataset():
    '''
    该函数设置了一个样本数据。
    
    return：样本数据列表，每个样本的标签
    '''
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    return posting_list, class_vec

# 根据样本数据，建立单词列表
def create_vocablist(dataset):
    vocab=[]
    for sample in dataset:
        vocab.extend(sample)
    vocab_set=set(vocab)
    vocab_list=list(vocab_set)
    return vocab_list

#词集模型
# 构造基于单词列表的向量，该向量表示了单词列表中的每个单词
# 在输入文档中是否出现，出现为1，不出现为0
def create_vec(vocab_list,input_data):
    result_vec=[0]*len(vocab_list)
    for word in input_data:
        # 单词表中含有该单词
        if word in vocab_list:
            result_vec[vocab_list.index(word)]=1
    return result_vec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)      #change to np.ones()
    p0Denom = float(numWords); p1Denom = float(numWords)                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom          
    p0Vect = p0Num/p0Denom          
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def nb_classifier(doc_list):
    posting_list, class_vec=load_dataset()
    vocab_list=create_vocablist(posting_list)
    trainMat=[]
    for sample in posting_list:
        trainMat.append(create_vec(vocab_list,sample))
    p0Vec,p1Vec, pAbusive=trainNB0(trainMat,class_vec)
    vec2Classify=np.array(create_vec(vocab_list,doc_list))
    class_label=classifyNB(vec2Classify,p0Vec,p1Vec,pAbusive)
    return class_label


doc_list1=['you','are','so','cute'] 
print(nb_classifier(doc_list1))
doc_list2=['you','are','so','stupid']
print(nb_classifier(doc_list2))    