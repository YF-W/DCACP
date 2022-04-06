import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import seaborn as sns
import pandas as pd
from matplotlib import colors

def scatterPlot(data):
    """
    Data visualization ( scatter plot )
    :param data: data set
    :return: No return
    """
    x = data[:, 0]
    y = data[:, 1]
    n = np.arange(data.shape[0])

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(n):
        ax.annotate(txt + 1, (x[i], y[i]))
    plt.savefig('img\meta_data.svg', dpi=600)
    plt.show()


def scatterPlotPhe(data, pheList):
    """
    Data visualization ( pheromone map )
    :param data: Data set
    :return: No return
    """

    x = data[:, 0]
    y = data[:, 1]
    fig, ax = plt.subplots()

    for i in range(0, len(pheList)):
        now_pheList = normalize(pheList, pheList[i])
        plt.scatter(x[i], y[i], marker=".", s=now_pheList * 1000, c='orange')
        ax.annotate(i + 1, (x[i], y[i]))
    plt.savefig('img\phefig.svg', dpi=600)
    plt.show()

def scatterPlotLine(data, num,history_run_list):
    """
    Data visualization ( path map )
    :param data: data set
    :return: 无返回
    """

    x = data[:, 0]
    y = data[:, 1]
    fig, ax = plt.subplots()
    x_list=[]
    y_list=[]
    for j in range(0,len(history_run_list)):
        now_point=history_run_list[j]-1
        now_x=x[now_point]
        now_y=y[now_point]
        x_list.append(now_x)
        y_list.append(now_y)
    ax.plot(x_list, y_list, color='black', label='.', marker='.',markersize=10,linewidth=1,alpha=0.3,markerfacecolor='red')
    figname='img\line_'+str(num)+'.svg'
    plt.savefig(figname, dpi=600)
    plt.show()


def distanceMatrix(matrix):
    """
    distanceMatrix
    :param matrix: Matrix composed of original coordinate data
    :return: dis_matrix: distance matrix
    """
    matrix = np.array(matrix, dtype=np.float64)  # Transform the incoming matrix into a numpy type matrix ( ndarray )
    dis_matrix = distance.cdist(matrix, matrix,
                                'euclidean')  # Call the distance function, find the distance of matrix AB, A = matrix, B = matrix, Euclidean distance
    return dis_matrix  # Returns a distance matrix


def neighborPointList(p1, disMatrix):
    """
    Gets the column of p1 in the nearest neighbor matrix and arranges the nearest neighbor of p1 in turn
    :param p1:To find the nearest neighbor point
    :param disMatrix: distance matrix
    :return: The neighbor of p1 arranged in order
    """
    keys = []  # Create list
    for i in range(1, disMatrix.shape[1] + 1):  # Create key value
        keys.append(i)
    a = dict(zip(keys, disMatrix[p1 - 1]))  # compression dictionary
    a = sorted(a.items(), key=lambda x: x[1])  # Updated to list
    neighborArr = []  # The column of p1 nearest neighbor matrix

    for i in range(1, len(a)):
        neighborArr.append(a[i][0])  # Add the returned value to the nearest neighbor column
    return neighborArr


def neighborListALL(disMatrix):
    """
    Get a complete nearest neighbor matrix
    :param disMatrix: Distance Matrix
    :return:neighborMatrix
    """
    res = []
    for i in range(0, disMatrix.shape[1]):  # 把近邻列加入到近邻矩阵内
        sorted_id = sorted(range(len(disMatrix[i])), key=lambda k: disMatrix[i][k], reverse=False)
        temp = sorted_id
        sorted_id = sorted(range(len(temp)), key=lambda k: temp[k], reverse=False)
        res.append(sorted_id)
    return np.array(res)


def pheromonesList(data):
    pointNum = data.shape[0]  # Number of points sought
    pheList = []  # Create a list of pheromones
    for i in range(0, pointNum):
        pheList.append(1)  # Add values to the pheromone list
    return pheList


def randomData(mu, sigma, row, col):
    """
    Production of Gaussian distribution data
    :param mu: mean
    :param sigma: standard deviation
    :param row: number of rows
    :param col: number of columns
    :return: Gaussian distribution data set
    """
    data1=np.random.normal(mu, sigma, [row, col])
    data2=np.random.normal(mu/2, sigma/2, [row, col])
    data3=np.append(data1,data2,axis=0)
    return data3


def normalize(list, value):
    range = max(list) - min(list)
    if range == 0:
        return 1
    else:
        value2 = (value - min(list)) / range
        return value2


def getReciprocal(matrix):
    """
    Get the reciprocal of matrices
    :param matrix:incoming matrix
    :return: reciprocal of matrix
    """

    return np.divide(1, matrix, out=np.zeros_like(matrix, np.float64), where=matrix != 0)


def getNND(neighborMatrix):
    """
    Get NND of Matrix
    :param matrix: The incoming matrix
    :return: Returns the NND list
    """
    NND = []  # NND List
    # i j represents the serial number of points, so start with 1
    for i in range(0, neighborMatrix.shape[0]):
        NND.append(np.mean(np.divide(neighborMatrix[i], neighborMatrix.T[i], out=np.zeros_like(neighborMatrix[i], np.float64), where=neighborMatrix.T[i] != 0)))
    return NND

def getTypeList(data):
    """
    Gets TypeList to represent the type of data
    :param data: data matrix
    :return: TypeList
    """
    TypeList = []
    for i in range(0, data.shape[0]):
        TypeList.append(-1)
    return TypeList


def resPlot(data, typeList):
    """
    Make clustering diagrams ( for 2D data only )
    :param data: Data
    :param typeList: typeList
    :return:
    """
    df1 = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1]})  # Get X, Y corresponding data
    df2 = pd.DataFrame(typeList)  # Get type
    df1.insert(df1.shape[1], 'type', df2)
    sns.lmplot(x='x', y='y', hue='type',
               data=df1, fit_reg=False)

    return plt

def resPlot3D(data, typeList,fig_x,fig_y):
    """
    Make clustering maps ( for 3D data only )
    :param data: data
    :param typeList: typeList
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    mycolors=list(colors.CSS4_COLORS.keys())
    for i in range(0,len(typeList)):
        ax.scatter(data[i, 0], data[i, 1], data[i, 2], c=mycolors[(typeList[i]+5)])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(fig_x, fig_y)
    return plt

