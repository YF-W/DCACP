import math
import random
import pandas as pd
import scipy.stats as st


def antRun(pheList, p, dis_Recip, NND, eta, niu, dataAnal, k, maxType, typeList, now_round, round_ant_num,
           now_ant_num, ant_order_max, alpha, beta, neighborMatrix, history_run_list):
    """
    Crawling and clustering after death
    :param pheList: List of pheromones
    :param p: starting point
    :param dis_Recip: Inverted distance matrix
    :param NND: NND Matrix
    :param eta: Ant life ( a sign of survival )
    :param niu: Factors determining type 2 mortality. Triggered less, used to shorten clustering time and correct clustering results
    :param dataAnal: Record how many times each point was crawled
    :param k: k-nearest neighbor
    :param maxType: Maximum number of clusters
    :param typeList: Clustering matrix. Represents the type of each point
    :param now_round: Number of rounds now
    :param round_ant_num: Number of ants per round
    :param now_ant_num: Now ant num of this round
    :param ant_order_max: Total number of ants
    :param alpha: Upper Limit for Controlling Ant Type 1 Death
    :param beta: Lower Limits for Controlling Ant Type 1 Death
    :param neighborMatrix: nearest neighbor matrix
    :param history_run_list: Record the Historical Path of Ant Crawling ( Drawing )
    :return: Maximum number of clusters updated
    """
    delta_List = []  # delta used to record climbing points
    nabla_List = []  # nabla used to record crawl points
    run_List = []  # Record the number of points ants crawl through
    run_List.append(p)  # Add the starting point to run_List
    history_run_list.append(p)

    ant_order = now_round * round_ant_num + now_ant_num  # The current ant in the overall ant number. Round and num start counting from 0, so finally + 1
    N_mu = ant_order_max / 2  # Take the middle ant to reach the maximum curve
    N_sigma = ant_order_max / 6  # (u-3*sig,u+3*sig)  6*sig=ant_order_max
    multiple_for_max_neibor = alpha * (
        math.sqrt(2 * math.pi)) * N_sigma  # Used to raise a normal distribution to a required value ( multiple )
    mu = multiple_for_max_neibor * st.norm.pdf(ant_order, N_mu, N_sigma) + beta
    # This is the ant ' s mu, mu affects the ant ' s crawling range, if the nabla + delta value is greater than mu, then terminate the ant ' s crawling.
    # Here replace the current ant number, normal distribution of mu,
    # sigma get the corresponding normal distribution position, and then add beta to improve the basic value.
    while eta >= 0:
        nextpoint, probability = AntChoice(pheList, p, dis_Recip, NND, k, dataAnal, neighborMatrix)  # Finding the next point and its probability
        run_List.append(nextpoint)  # Add the point to climb to run_List
        history_run_list.append(p)

        # Stop finding condition 1
        new_delta = neighborMatrix[p][nextpoint]  # Get the delta of this step
        new_nabla = neighborMatrix[nextpoint][p]  # Get the nabla for this step
        delta_List.append(new_delta)  # Add a new delta to delta_List
        nabla_List.append(new_nabla)  # Add a new nabla to nabla_List
        temp_sum2 = new_nabla + new_delta  # A new definition, added later

        if temp_sum2 > mu:  # If the value of nabla + delta is greater than mu, stop the ant crawling
            eta = -1  # Let ants die
            run_List.pop()  # Let ants die
            history_run_list.pop()
            break  # Delete the point to add crawling

        # Stop searching condition 2
        max_num = max(run_List, key=run_List.count)  # Statistics of the most frequent points
        max_time = run_List.count(max_num)  # Count the number of occurrences of this point
        if max_time > niu:  # If you want a point to crawl more than niu, stop the ant crawling
            eta = -1  # Let ants die
            run_List.pop()  # Delete the point to add crawling
            history_run_list.pop()
            break  # Exit cycle
        # Normal situation

        pheList[p] = pheUpdate(pheList[p], probability)
        pheList[nextpoint] = pheUpdate(pheList[nextpoint], probability)  # Update endpoint pheromone
        p = nextpoint  # Update the next point
        dataAnal[nextpoint] += 1  # Update the number of crawls at the next point
        eta = etaUpdate(eta)  # Update eta

    # ---------------Ants begin clustering after death-------------
    if ((eta == -1) & (len(run_List) != 0)):  # If the ants die and the crawling list is not empty.
        dead_run_list = list(set(run_List))  # Get the run_List that 's gone, and see where the ant climbs all his life
        dead_run_list_num = len(dead_run_list)  # Gets the size of dead_run_list for subsequent loop traversal
        color_list = []  # Cluster list
        for m in range(0, dead_run_list_num):  # Start traversing every point you crawl through
            now_point = dead_run_list[m]  # Get which point the current climb is.
            color_list.append(typeList[now_point])  # Add the class corresponding to this point to the cluster list
        color_list2 = pd.value_counts(color_list)  # Sort the cluster list to see which category appears most frequently
        mark = color_list2.index[0]  # Get the most frequent cluster
        if (mark == -1):  # If the largest cluster is-1, -1 denotes the undefined cluster. Then make the crawl point a new cluster
            maxType = maxType + 1  # If the largest cluster is-1, -1 denotes the undefined cluster. Then make the crawl point a new cluster
            for n in range(0, dead_run_list_num):  # Go through every point you crawl through
                now_point = dead_run_list[n]  # Get this point number
                typeList[now_point] = maxType  # Set this point to a new cluster
        else:  # If the most frequent clusters are not - 1 clusters, then all points are summed up as the most frequent clusters
            for n in range(0, dead_run_list_num):  # To traverse every point
                now_point = dead_run_list[n]  # Get the current point serial number
                typeList[now_point] = mark  # Set the most frequent cluster to the cluster at the current point
    return maxType


def AntChoice(pheList, p, dis_Recip, NND, k, dataAnal, neighborMatrix):
    """
    :param pheList: List of pheromones
    :param p: starting point
    :param dis_Recip: reciprocal of distance matrix
    :param NND: NND matrix
    :param dis: distance matrix
    :param k: k nearest neighbor
    :param run_List: list of points that have been crawled
    :return: the next point selected
    """
    PC_List = []  # Path reliability matrix
    PC_sum = 0  # Sum of path reliability
    Probability = []  # Probability list
    neighborMatrix_index = sorted(range(len(neighborMatrix[p])), key=lambda k: neighborMatrix[p][k])
    for i in range(1, k + 1):
        phe = pheList[i]  # Finding the pheromone of the end point
        factor1 = getPC(phe, p, neighborMatrix_index[i], dis_Recip, NND, neighborMatrix)  # The path reliability from point p to point i is
        # Another influencing factor (factor 2) was added here to act on PC_temp
        # The purpose is to solve the problem of ants running back and forth at two points.
        # Factor 2 is similar to sigmoid function, but take x as negative
        appear_num = dataAnal[i]  # Count the number of crawls at this point
        if (appear_num > 5):
            appear_num = 5
        factor2 = 1 / (1 + math.e ** appear_num)  # Calculation factor2
        PC_temp = factor1 * factor2  # Definition of PC_temp
        PC_List.append(PC_temp)  # Add path reliability to the list
        PC_sum = PC_sum + PC_temp  # Cumulative summation
    for i in range(0, len(PC_List)):  # The probability of each point is calculated
        if PC_sum != 0:
            Probability.append(PC_List[i] / PC_sum)
        else:
            Probability.append(PC_List[i])
    chosen_point_index = roulette(Probability)  # The end point is obtained
    chosen_point = neighborMatrix_index[chosen_point_index]
    probability = Probability[chosen_point_index]  # The probability from the starting point to the end point
    return chosen_point, probability


def getPC(phe, p, pap, dis_Recip, NND, neighborMatrix):
    """
    :param phe: pheromone of the endpoint
    :param p: starting point
    :param pap: endpoint
    :param dis_Recip: reciprocal of distance matrix
    :param NND: NND matrix
    :param neighborMatrix: neighbor matrix
    :return:
    """
    nabla = neighborMatrix[pap][p]  # Get the nabla
    delta = neighborMatrix[p][pap]  # Get the delta
    dis_p_pap = dis_Recip[p][pap]  # Find the reciprocal of p p ' distance
    pc = phe * (nabla + 1) / (delta + 1) * dis_p_pap * NND[pap]  # Seeking Path Reliability
    return pc  # Return path reliability


def roulette(fitness):
    """
    Implementation method of stochastic acceptance for roulette strategy
    :param fitness:Imported probabilistic data that can be arranged in a small to large order ( list or tuple )
    :return: selected point
    """
    N = len(fitness)  # Length of probability data
    maxFit = max(fitness)
    if maxFit == 0:
        return -1
    while True:
        # randomly select an individual with uniform probability
        ind = int(N * random.random())
        # with probability wi/wmax to accept the selection
        if random.random() <= fitness[ind] / maxFit:
            return ind


def pheUpdate(oldphe, probability):
    newphe = oldphe + probability  # Update pheromone
    return newphe  # Return the updated pheromone


def etaUpdate(oldeta):
    neweta = oldeta
    return neweta
