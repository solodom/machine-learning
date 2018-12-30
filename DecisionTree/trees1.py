import operator
from math import log


def dataentro_cal(dataset):
    """
    this function calculate the entropy of the dataset

    Args:
    dataset：a nparray with every row as a fea_vec

    Return:
    the entropy of the dataset
    """
    total_num=len(dataset)
    data_labels={}
    for fea_vec in dataset:  #  以axis=0轴 对array进行迭代
        data_labels[fea_vec[-1]]=data_labels.get(fea_vec[-1],0)+1
    entropy=0.0
    for label in data_labels.keys():
        label_prob=float(data_labels[label])/total_num
        entropy-=label_prob*log(label_prob,2)  #  log(x,n)--->以n为底X的对数
    return entropy


def create_dataset():
    """
    this function create a example dataset to test on other functions
    
    
    Args: None
    
    
    Return:
        dataset: nparray,the row is the fea_vec containning the features data,and the according data
        labels:the classifications list
    """
    dataset=[[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def split_dataset(dataset,index,value):
    """
    this function splits the dataset into two parts based on the index of the fea_vec and its value


    Args:
        dataset: nparray with fea_vec as rows
        index: int, the index on which to split the dataset, it means the according feature
        value: int, if the index element has this value, the reduced fea_vec will be returned


    Return:
        ret_dataset:list, cotaining the fea_vec who's index element has value and without index element
    """
    #  the change to the dataset array inside the function will also affect the array it self, 
    #  so you must create new lists to manipulate and return
    ret_dataset=[] 
    for fea_vec in dataset:
        if fea_vec[index]==value:
            redfea_vec=fea_vec[:index]
            redfea_vec.extend(fea_vec[index+1:])  #  extend() append muliple elements at a time
            ret_dataset.append(redfea_vec)
    return ret_dataset


# 
def bestfeature_split(dataset):
    """
    this function iterably split the dataset and calculate the information gain, 
    and return the best feature index on which the dataset is best splited with maximum information gain

    Args:
        np.array

    Return:
        the index of the best feature
    """
    fea_nums=len(dataset[0])-1
    base_entro=dataentro_cal(dataset)
    best_gain=0.0
    for i in range (fea_nums):  # iterate all the features
        con_entro=0.0
        fea_list=[sample[i] for sample in dataset ]  # collect all the feature values
        fea_set=set(fea_list)  # make the values unique to each other
        for value in fea_set:  # iterate all the values of a feature to seperate the dataset
            sub_dataset=split_dataset(dataset,i,value)
            entropy=dataentro_cal(sub_dataset)
            prob_subdataset=len(sub_dataset)/float(len(dataset))
            con_entro+=prob_subdataset*entropy
            info_gain=base_entro-con_entro
        if (info_gain>best_gain):
            best_gain=info_gain
            best_feature=i
    return best_feature


def majority_cnt(classlist):
    """
    this function count the majority of the class list and return the class with most number

    args:
        classlist: list

    return:
        the class name of the majority one

    """
    class_count={}
    for vote in classlist:
        class_count[vote]=class_count.get(vote,0)+1
    sortedclass_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclass_count[0][0]


def create_tree(dataset,labels):
    """
    this function create the decision tree based on dataset, and feature labels

    args:
        dataset: np.array, the dataset, the last element is classification
        labels: list, represents the feature names

    returns:
        mytree: a dict, with the structure of a decision tree
    """
    classlist=[sample[-1] for sample in dataset]
    # if the the all the samples in the dataset have the same class return
    if classlist.count(classlist[0])==len(classlist):  # list.count(list[i])
        return classlist[0]
    # if all the featurs has been splited
    elif len(dataset[0])==1:
        return majority_cnt(classlist)
    bestfea_index=bestfeature_split(dataset)
    bestfea_label=labels[bestfea_index]
    mytree={bestfea_label:{}}
    feavalue_list=[sample[bestfea_index] for sample in dataset]
    feavalue_set=set(feavalue_list)
    # delete the feature label in the labels list
    sub_labels=labels[:]
    del(sub_labels[bestfea_index])
    for value in feavalue_set:
        sub_dataset=split_dataset(dataset,bestfea_index,value)
        mytree[bestfea_label][value]=create_tree(sub_dataset,sub_labels)
    return mytree




    
