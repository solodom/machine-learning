import matplotlib.pyplot as plt


decision_node=dict(boxstyle='sawtooth',fc='0.8')
leaf_node=dict(boxstyle='round4',fc='0.8')
arrow_args=dict(arrowstyle='<-')

def plot_node(nodetxt,centerpt,parentpt,nodetype):
    # nodetxt: string, 注释文本
    # xy: (x,y) 注释点的坐标
    # xycoords: 注释点的坐标系， axes fraction 表示坐标系左下角1*1 的区域
    # xytext: 注释文本的坐标
    # textcoords: 注释文本的坐标系
    # va：方框垂直方向位置
    # ha：方框水平方向位置
    # bbox：bbox={}表示对方框的设置
    # arrowprops： arrowprops={}表示对箭头的设置
    create_plot_1.ax1.annotate(nodetxt,xy=parentpt,
    xycoords='axes fraction',
    xytext=centerpt,textcoords='axes fraction',
    va='center',ha='center',bbox=nodetype,arrowprops=arrow_args)


def create_plot_1():
    fig=plt.figure(1,facecolor='white')
    # Clear the current figure
    fig.clf()
    # make the ax1 a attribute
    create_plot_1.ax1=plt.subplot(111,frameon=False)
    plot_node('a decision node',(0.5,0.1),(0.1,0.5),decision_node)
    plot_node('a leaf node',(0.8,0.1),(0.3,0.8),leaf_node)
    plt.show()


def get_leafnum(mytree):
    """
    this function get the numbers of the leaf nodes to determine x axis

    args:
        mytree: dict of tree construction

    return:
        leafnum:int,the number of leaves
    """
    leaf_num=0
    root=list(mytree)[0]  # the root node is not a leaf
    branch_dict=mytree[root]
    for key in branch_dict.keys():
        if type(branch_dict[key]).__name__=='dict':
            sub_tree=branch_dict[key]
            leaf_num+=get_leafnum(sub_tree)
        else:
            leaf_num+=1
    return leaf_num


def get_treedepth(mytree):
    """
    this function find out the max depth of a tree

    args:
        mytree: a dict construction tree

    returns:
        maxdepth: int, the maximum depth of the tree
    """
    maxdepth=0
    root=list(mytree)[0]
    branch_dict=mytree[root]
    for key in branch_dict.keys():
        if type(branch_dict[key]).__name__=='dict':
            sub_tree=branch_dict[key]
            this_depth=1+get_treedepth(sub_tree)
        else:
            this_depth=1
        if this_depth>maxdepth:
            maxdepth=this_depth
    return maxdepth