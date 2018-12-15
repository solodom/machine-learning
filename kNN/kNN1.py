import numpy as np
import operator
from pylab import * # 为了在showfigure（）显示中文字符
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # numpy.tile(A,reps) tile共有2个参数，A指待输入数组，reps则决定A重复的次数。整个函数用于重复数组A来构建新的数组。
    # 构建与样本数组同型的数组
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    # sum 默认的axis=0 就是普通的相加 而当加入axis=1以后就是将一个矩阵的每一行向量相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # argsort()将数组元素从小到大排序，返回index数组，默认axis=1 按行排序，axis=0时按列排序
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # get返回指定键的值，如果值不在字典中返回默认值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        #sorted 返回一个list， classCount.items()返回 [(key,value)] list operator.itemgetter 指定按照哪一个
        #元素进行排序 reverse=True 降序排列
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    love_dictionary={'largeDoses':3,'smallDoses':2,'didntLike':1}
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=np.zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromline=line.split('\t')
        returnMat[index,:]=listFromline[0:3]
        if (listFromline[-1].isdigit()):
            classLabelVector.append(listFromline[-1])
        else:
            classLabelVector.append(love_dictionary.get(listFromline[-1]))
        index +=1
    return returnMat, classLabelVector


    # self define
## 设置datingLabels的颜色矩阵，用于为标记点上色
# labels is a labels list
def labels2collor(labels):
    collormatsize=len(labels)
    # construct the collor matrix
    collormat=np.zeros((collormatsize,3))
    # red for largeDoses,green for smallDoses,blue for didntlike
    for i in range(len(labels)):
        if labels[i]==3:
            collormat[i]=[220,20,60]
        elif labels[i]==2:
            collormat[i]=[0,255,0]
        elif labels[i]==1:
            collormat[i]=[25,25,112]
    # RGB used in matplotlib must be float, divided by 255
    collormat=collormat/255
    return collormat


#self define
# plot the figure 
def showfigure(datingDataMat,datingLabels):
    # 为了显示中文字符
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    # creat a new figure，and set the size
    fig=plt.figure(figsize=(7,12))
    figurelabels=['none','didntlike','smallDoses','largeDoses']
    # creat axes add_subplot(row_quantity,column_quantity,position) used as figure.add_subplot
    # the same as plt.subplot(row_quantity,column_quantity,position)
    ax1=fig.add_subplot(211)
    # scatter(x,y,size,color) size and color must match x and y
    # 以玩游戏所耗时间比， 每周消耗冰激凌的公升数来构建散点图
    ax1.scatter(datingDataMat[:,1],datingDataMat[:,2],8.0*np.array(datingLabels),labels2collor(datingLabels))
     # set label names
    ax1.set_xlabel('玩游戏所耗时间比')
    ax1.set_ylabel('每周消耗冰激凌的公升数')
    ax1.set_title('Figure-1')
    ax2=fig.add_subplot(212)
    ax2.scatter(datingDataMat[:,0],datingDataMat[:,1],8.0*np.array(datingLabels),labels2collor(datingLabels))
    ax2.set_xlabel('每年的飞行行程数')
    ax2.set_ylabel('玩游戏所耗时间比')
    ax2.set_title('Figure-2')    
    plt.show()



# autoNorm(dataset) dataset is an array
# 1.find the minVal and maxVal of every column of  matrix
# 2. use maxVal- minVal to construct a matrix as denominator with a shape of dataset :demat
# 3. use minVal to contruct a matrix with shape of dataset : minmat
# 4. use dataset-minmat as numerator: numat
# 5. the normmat is numat/demat
def autoNorm(dataset):
    maxval=dataset.max(0)
    minval=dataset.min(0)
    ranges=maxval-minval
    m=dataset.shape[0]
    demat=np.tile(ranges,(m,1))
    minmat=np.tile(minval,(m,1))
    numat=dataset-minmat
    #normat=np.zeros(dataset.shape)
    normat=numat/demat
    return normat,ranges, minval



# dataingClassTest(filename)
# set the hold ratio
# change the txt file into matrix: datingMat, datingLabels
# normalize the datingMat : normDatingMat
# seperate the data into two parts : trainMat, testMat
# set errCount
# use classify0 to classify the testMat with the trainMat as dataset
# print the classResult, and the real answer
# print the error ratio
def datingClassTest(filename):
    hRatio=0.1
    datingMat,datingLabels=file2matrix(filename)
    normDatingMat, ranges, minVals=autoNorm(datingMat)
    m=normDatingMat.shape[0]
    numTestVec=int(m*hRatio)
    errCount=0
    for i in range(numTestVec):
        classResult=classify0(normDatingMat[i,:],normDatingMat[numTestVec:m,:],datingLabels[numTestVec:m],3)
        print('the predicted result is {}, and the real answer is {}'.format(classResult,datingLabels[i]))
        if classResult!=datingLabels[i]:
            errCount+=1
    print('the error ratio is {}'.format(errCount/numTestVec))


def classifyPerson():
    # set resultList
    resultList = ['not at all', 'in small doses', 'in large doses']
    # input the percentage of time spent playing cideo games
    percentTats = float(input("percentage of time spent playing video games?"))
    # input flier miles
    ffMiles = float(input("frequent flier miles earned per year?"))
    # input liters of cream consumed per year
    iceCream = float(input("liters of ice cream consumed per year?"))
    # change dataset text to matrix return datingDataMat and datingLabels using file2matrix
    datingDataMat, datingLabels = file2matrix('D:/pythoncode/machine learning in action/DatingTestSet.txt')
    # normalize the datingDataMat, ranges and minVals matrix
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # change the inputs into array
    inArr = np.array([ffMiles, percentTats, iceCream, ])
    # classify the input
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    # print result
    print("You will probably like this person: %s" % resultList[classifierResult - 1])


def img2vec(filename):
    """
    chenge the text file of image to vector
    """
    return_vec=np.zeros(1,1024)
    fr=open(filename)
    for i in range(32):
        line_str=fr.readline()
        for j in range (32):
            return_vec[0,32*i+j]=int(line_str(j))
    return return_vec


def handwriting_classtest():
    """
    use trainningDigits as train, use testDigits to test
    """
    hw_labels=[]
    train_filelist=listdir('trainingDigits')
    tdigits_num=len(train_filelist)
    train_mat=np.zeros((tdigits_num,1024))
    for i in range (tdigits_num):
        filename_str=train_filelist[i]
        file_str=filename_str.split('.')[0]
        digit_str=file_str.split('_')[0]
        hw_labels.append(digit_str)
        train_mat[i,:]=img2vec('trainingDigits/'+filename_str)
    test_filelist=listdir('testDigits')
    error_cont=0
    testDigits_num=len(test_filelist)
    for i in range(testDigits_num):
        filename_str=test_filelist[i]
        file_str=filename_str.split('.')[0]
        digit_class=file_str.split('_')[0]
        test_vec=img2vec('testDigits/'+filename_str)
        result_class=classify0(test_vec,train_mat,hw_labels,3)
        if digit_class!=result_class: error_cont+=1
    print('the number of error is {}'.format(error_cont))
    print('the ratio of number is {}'.format(float(error_cont)/testDigits_num))

    

