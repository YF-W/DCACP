import pandas as pd
import AntRun
import dataPreprocessing
import numpy as np

"""
DCACP

Proposed By Shijie Zeng, Yufei Wang
Chengdu University
2022. 4

"""
"""For research and clinical study only, commercial use is strictly prohibited"""



def antModel(data, round, niu, k, alpha, beta,ant_num,n_cluster,type):
    """
    :param data: data set
    :param round: The num of implemented rounds, every round has an ant crawling
    :param niu: The maximum number of repeated crawling at the same point
    :param k: Traversal neighbor range, k = 10, count only 10 nearest neighbors
    :param alpha: Factors controlling ant death. Is also the maximum value of normal distribution curve
    :param beta: Represents the base value to be added to the normal distribution curve.
                Because the first and last values of the normal distribution are the smallest, the ant will die meaninglessly
    :param ant_num: Number of ants
    :param n_cluster: Expected clusters ( valid only when type = = 2 )
    :param type: Type of Ant Clustering
    :return:
    """

    dis = dataPreprocessing.distanceMatrix(data)  # distance matrix
    dis = (dis - np.min(dis)) / (np.max(dis) - np.min(dis))  # Normalization of distance matrix
    dis_Recip = dataPreprocessing.getReciprocal(dis)  # Finding the reciprocal of distance matrix
    dis_Recip = (dis_Recip - np.min(dis_Recip)) / (np.max(dis_Recip) - np.min(dis_Recip))  # Normalization of reciprocal of distance matrix
    neighborMatrix = dataPreprocessing.neighborListALL(dis) # Finding nearest neighbor matrix
    pheList = dataPreprocessing.pheromonesList(data)  # Find the list of pheromones
    n_s=ant_num
    NND = dataPreprocessing.getNND(neighborMatrix)  # Obtaining NND Matrix
    dataAnal = [0] * data.shape[0]  # Number of records selected
    typeList = dataPreprocessing.getTypeList(data)  # List of final point type
    maxType = 0
    ant_order_max = round * n_s
    history_run_list = []
    birth_list=[]

    for i in range(0, round):  # Traversing each round
        for j in range(0, n_s):  # Traversing every ant
            eta=1
            p = pheList.index(min(pheList)) # Ant Birth Point
            birth_list.append(p)
            maxType = AntRun.antRun(pheList, p, dis_Recip, NND, eta,niu, dataAnal, k, maxType, typeList, i, n_s,
                                    j, ant_order_max, alpha, beta, neighborMatrix,history_run_list)  # Let the ants run

    # type==1 direct return

    if type==2:
        allPoint=len(typeList)
        group_point_num=int(allPoint/n_cluster)
        typeList_count=pd.value_counts(typeList)
        hub_group_type=list(typeList_count.index[0:(n_cluster)])
        for i in range(0,allPoint):
            if typeList[i] in hub_group_type:
                continue
            else:
                neighborMatrix_index = sorted(range(len(neighborMatrix[i])), key=lambda k: neighborMatrix[i][k])
                nowpoint_type_list=[]
                for j in range(0,group_point_num):
                    neighbor_num=neighborMatrix_index[j]
                    nowpoint_type_list.append(typeList[neighbor_num])
                typeList[i]=pd.value_counts(typeList).index[0]

    elif type==3:
        # --------example------------
        # old_pheList [1 2 5 3 2 1]
        # old_typeList [1 1 2 2 3 4]
        # old_index 0,1,2,3,4,5
        # new_pheList [1 1 2 2 3 5]
        # new_typeList [1, 4, 1, 3, 2, 2]
        # new_index [0 5 1 4 3 2]
        # --X---------------X--
        hub_index_old=[]
        typeSet=set(typeList)# {1,2,3,4}
        typeSet_listType=list(typeSet)# [1,2,3,4]
        len_typeSet=len(typeSet)
        len_typeList=len(typeList)
        new_typeList=[]
        new_index = np.argsort(pheList) # new_index [0 5 1 4 3 2]
        for i in range(0,len_typeList):
            now_index=new_index[i]# old_typeList [1 1 2 2 3 4]
            new_typeList.append(typeList[now_index])# new_typeList [1, 4, 1, 3, 2, 2]
        for i in range(0,len_typeSet):
            nowType=typeSet_listType[i]                         ##  typeSet:{1,2,3,4} typeSet[3]=4
            type_first_time_index=new_typeList.index(nowType) ## new_typeList.index(4)=1
            old_position=new_index[type_first_time_index]## new_index=[0 5 1 4 3 2]  new_index[1]=5
            hub_index_old.append(old_position) ### hub_index_old=[5] old_pheList[5]=1 old_typeList[5]=4
        for i in range(0,len_typeList):
            nearest_neighbor=hub_index_old[0]
            min_neighbor_num=len_typeSet
            for choice in range(0,len(hub_index_old)):
                if neighborMatrix[i][hub_index_old[choice]]<min_neighbor_num:
                    min_neighbor_num=neighborMatrix[i][nearest_neighbor]
                    nearest_neighbor=hub_index_old[choice]
            typeList[i]=typeList[nearest_neighbor]

    return pheList, dataAnal, typeList

