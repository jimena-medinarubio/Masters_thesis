#%%
import numpy as np
import xarray as xr
import infomap
import networkx as nx
#%%

connectivity_matrix= np.load('connectivity_matrix.npy')
#%%

def infomap(connectivity_matrix):

    #create graph
    A=np.array(connectivity_matrix)
    G = nx.DiGraph(A)

    #create an Infomap object
    im = infomap.Infomap(markov_time=y)

    #add edges and their weights to the Infomap object
    for u, v, data in G.edges(data=True):
        im.addLink(u, v, data['weight'])

    #run the Infomap algorithm
    im.run()

    #access the community structure results
    tree = im.tree

    print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")

    nodes_id=[]
    nodes_com=[]

    #access the communities
    for node in tree:
        if node.isLeaf():
            print(f"Node ID: {node.node_id} - Module ID: {node.module_id}")
            nodes_id.append(node.node_id+1)
            nodes_com.append(node.module_id)
    
    return nodes_id, nodes_com
#%%

nodes_id, nodes_com= infomap(connectivity_matrix)