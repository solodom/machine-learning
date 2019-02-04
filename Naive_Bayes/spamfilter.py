import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
import re


# 对邮件的文本解析函数
def textParse(bigString):    #input is big string, #output is word list
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 根据样本数据，建立单词列表
def create_vocablist(dataset):
    vocab=[]
    for sample in dataset:
        vocab.extend(sample)
    vocab_set=set(vocab)
    vocab_list=list(vocab_set)
    return vocab_list

def create_vec(vocab_list,input_data):
    result_vec=[0]*len(vocab_list)
    for word in input_data:
        # 单词表中含有该单词
        if word in vocab_list:
            result_vec[vocab_list.index(word)]=1
    return result_vec


def spamFilterTest():
    '''
    垃圾邮件分类器，输入为垃圾邮件样本的地址和正常邮件样本的路径
    根据样本数据训练贝叶斯模型，并进行交叉验证，评估垃圾邮件分类器的性能
    '''
    textList=[]
    classList=[]
    classDict={1:'spam',0:'ham'}
    for i in range (1,26):
        spamTextCode=open('D:/pythoncode/machine-learning-in-action/Naive_Bayes/email/email/spam/'+str(i)+'.txt','rb').read()
        spamText=spamTextCode.decode('ISO-8859-1')
        spamTextParsed=textParse(spamText)
        textList.append(spamTextParsed)
        classList.append(1)
        hamTextCode=open('D:/pythoncode/machine-learning-in-action/Naive_Bayes/email/email/ham/'+str(i)+'.txt','rb').read()
        hamText=hamTextCode.decode('ISO-8859-1')
        spamTextParsed=textParse(hamText)
        textList.append(hamText)
        classList.append(0)
    vocabList=create_vocablist(textList)
    textVecList=[]
    for sample in textList:
        textVecList.append(create_vec(vocabList,sample))
    textVecArr=np.array(textVecList)
#     print(textVexArr[0])
    classVecArr=np.array(classList)
#     # 融合textVecArr,classVecArr
#     dataVecArr=np.c_[textVexArr,classVecArr]
#     print(dataVecArr[0])
    # 多项式朴素贝叶斯模型
    mnb=MultinomialNB()
    errorRateList=[]
    #依据交叉验证对数据集进行训练，验证
    kf=KFold(n_splits=5,shuffle=True)
    for trainIndicesArr,testIndicesArr in kf.split(textVecArr):
        # split 用generator返回坐标矩阵，将坐标矩阵作为索引可以将textVecArr中的相应元素提取出来
        trainVecArr=textVecArr[trainIndicesArr]
        trainClassArr=classVecArr[trainIndicesArr]
        testVecArr=textVecArr[testIndicesArr]
        testClassArr=classVecArr[testIndicesArr]
        print(trainVecArr,testVecArr,trainClassArr,testClassArr)
        #训练模型
        mnb=mnb.fit(trainVecArr,trainClassArr)
        # 预测数据
        yPreArr=mnb.predict(testVecArr)
        print(yPreArr)
        #计算错误率
        errorNum=0
        for i in range(len(yPreArr)):
            if yPreArr[i]!=testClassArr[i]:
                errorNum+=1
        errorRate=float(errorNum)/float(len(yPreArr))
        errorRateList.append(errorRate)
    return errorRateList


a=spamFilterTest()
print(a)
