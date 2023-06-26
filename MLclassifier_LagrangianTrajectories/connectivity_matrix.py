#%%
import numpy as np
import xarray as xr
#%%

itineraries = np.load('itineraries.npy')

#%%

def connectivity_matrix(itineraries):

    #calculate number of distinct regions
    num_regions=len(np.unique(itineraries))

    #empty array
    connectivity_matrix = np.zeros((num_regions, num_regions))

    #iterate over the itinerary of each particle
    for particle in itineraries:
        labels = []

        #compress data by removing consecutive same characters (e.g., 335556 → 356)
        for i, label in enumerate(particle):
            if i == 0 or label != particle[i-1]:
                labels.append(int(label))
            elif label == particle[i-1]:
                labels.append(label)

        #add value of (i,j)th element of the connectivity matrix for each pair of consecutive labels (ij)
        for i in range(len(labels)-1):
            source_region = labels[i]
            dest_region = labels[i+1]
            connectivity_matrix[int(source_region), int(dest_region)] += 1
        
        #discard transitions to land
        connectivity_matrix=connectivity_matrix[1:, 1:]

        #normalise for row-stochastic
        row_sums = connectivity_matrix.sum(axis=1)
        connectivity_matrix_norm = connectivity_matrix / row_sums[:, np.newaxis]
        
        connectivity_matrix_norm[np.isnan(connectivity_matrix_norm)]=0

        return connectivity_matrix_norm

#%%
c=connectivity_matrix(itineraries)
#%%
path='/write/path/here'
np.save('connectivity_matrix.npy', c)