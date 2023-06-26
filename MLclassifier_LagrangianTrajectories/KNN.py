#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
#%%
#upload domain partition
regions=xr.open_dataset('regions.nc')
#upload Parcels output
parcels=xr.open_dataset('trajectories.nc')

print(regions.keys())
print(parcels.keys())

# %%

def X_classifier(regions):
    """
    Stack into column the longitude and latitude coordinates
    covered by the domain to input in the ML classifier

    Input:
    Xarray dataset mapping a two dimensional space with lon and lat coordinates
    and a variable assigning a label to each grid point

    Output:
    Single column with (lon, lat) coordinates for each grid point with
    the optimal format for ML classifier input
    """
    #extract the latitude and longitude
    lat=regions['lat'].values
    lon=regions['lon'].values
    # Create a meshgrid of lat and lon
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    # Reshape the lat and lon arrays to be 1D
    lat_flat = lat_mesh.flatten()
    lon_flat = lon_mesh.flatten()
    #stack in (lon, lat) format
    X = np.column_stack((lon_flat, lat_flat))
    return X

def classifier(k, X, Y):
    """
    KNN classifier learns the domain decomposition

    Input:
    -k: number of nearest neighbours
    -X: single column of (lon, lat) coordinates of each grid point
    -Y: flat array of the labels of each grid point in X
    """
    #select the KNN classifier with k nearest neighbours
    clf=KNeighborsClassifier(n_neighbors=k)
    # Split the values into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = make_pipeline(StandardScaler(), clf)
    #fit to training data = domain decomposition
    clf.fit(X_train, Y_train)
    #estimate performance accuracy
    score = clf.score(X_test, Y_test)
    print('Estimation of accuracy '+str(score))
    return clf

def X_trajectories(regions, parcels):
    """
    Stack into column the longitude and latitude coordinates
    of each trajectories 

    Input:
    -regions: Xarray dataset mapping a two dimensional space with lon and lat coordinates
    and a variable assigning a label to each grid point
    -parcels: Parcels output netCDF file with lon, lat coordinates
    of the particles at each timestep

    Output:
    -flat array with the sample data 

    """
    X_trajectories=[]
    total_trajs = len(parcels.traj)
    #define boundary coordinate
    coords = [regions.lon.max(), regions.lat.max()]

    for traj in range(total_trajs):
        # define the input particle positions
        lon_in = parcels.lon[traj]
        lat_in = parcels.lat[traj]

        # determine which grid cell each particle is in at each time step
        lon_idx = np.digitize(lon_in, regions.lon)
        lat_idx = np.digitize(lat_in, regions.lat)

        X_traj=np.column_stack((regions.lon[lon_idx-1], regions.lat[lat_idx-1]))
        
        #change coordinates in boundary to land coordinates to exclude from analysis
        array_coords = np.full_like(X_traj, coords)
        X_traj=np.where(X_traj==array_coords, [-68.9, 12.13], X_traj)
        
        X_trajectories.append(X_traj)
        print(f"Processed {traj+1}/{total_trajs} trajectories")
    
    print('Sample data is ready')
    return X_trajectories

#%%
#coordinates for training data
X=X_classifier(regions)
#labels of each coordinate for training data
Y=regions.label.values.flatten()

#execute KNN classifier to learn label of grid points
clf=classifier(3, X, Y)

#%%

#extract coordinate from sample data 
X_trajs=X_trajectories(regions, parcels)
#%%

Y_trajs=[]
total_trajs=len(X_trajs)

#assign label at each coordinate from samples using KNN classifier
for i in range(total_trajs):
    y = clf.predict(X_trajs[i])
    Y_trajs.append(y)
    print(f"Processed {i+1}/{total_trajs} trajectories")
    print('The trajectories\' itineraries are ready')

#%%
path='/write/path/here/
np.save(path+'itineraries.npy', Y_trajs)
