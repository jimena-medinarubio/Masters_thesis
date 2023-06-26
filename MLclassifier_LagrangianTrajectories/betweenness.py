#%%
import networkx as nx
import numpy as np
import xarray as xr
import math
#%%

connectivity_matrix= np.load('connectivity_matrix.npy')

#%%

def betweenness(connec_matrix):

    """
    BETWEENNESS CENTRALITY:
    measure of the number of times each regions appears in the shortest path
    from each node to any other node of the graph 
    """

    #create graph
    A=np.array(connec_matrix)
    G = nx.DiGraph(A)

    #calculate shortest path
    M=nx.shortest_path(G)

    #empty array to store values of betweeness
    reg=np.arange(1, np.shape(connec_matrix)[0]+1)
    B=np.zeros_like(reg)

    for i in range(len(M)): #sources
        for j in range(len(M[0])): #sinks
            nodes=M[i][j] #shortest paths to any node from node (i,j)
            for n in nodes:
                B[n-1]=B[n-1]+1 #add to the count

    #normalisation
    norm=math.factorial(len(reg))/math.factorial(len(reg)-2)

    between=B/norm

    return between

#%%

#calcualte degree of betweenness for each region
betweeness_centrality=betweenness(connectivity_matrix)
