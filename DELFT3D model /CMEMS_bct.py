#%%
#author: JIMENA MEDINA RUBIO (jimena.medinarubio@gmail.com)
#Last updated: 05/06/2022

WRITNG TIME SERIES BOUNDARY CONDITIONS IN DELFT3D FORMAT FROM CMEMS DATA

#%%

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata



#read velocity field from global hydrodynamic model (netCDF format)
path='/insert/path/here/'
file='Venezuela-bct.nc' 
ds=xr.open_dataset(path+file)

print(ds.keys)

#2DH time series velocities
u=ds.utotal.mean('depth')
v=ds.vtotal.mean('depth')

#%%

"""INTERPOLATE VELOCITIES TO DELFT3D GRID 
*assuming velocity field has finer resolution than global hydrodynamic model """

#dimensions of DELFT3D MODEL
M=612 #x
N=317 #y

def interpolation(land_mask, x_bins, y_bins):
    # Define fine grid coordinates
    coarse_x, coarse_y = np.meshgrid(np.arange(land_mask.shape[1]), np.arange(land_mask.shape[0]))
    # Define coarse grid coordinates
    fine_x, fine_y = np.meshgrid(np.linspace(0, land_mask.shape[1] - 1, x_bins), np.linspace(0, land_mask.shape[0] - 1, y_bins))
    # Flatten coarse grid coordinates
    flat_coarse_x, flat_coarse_y = coarse_x.flatten(), coarse_y.flatten()
    # Interpolate mask onto fine grid
    flat_fine_x, flat_fine_y = fine_x.flatten(), fine_y.flatten()
    flat_coarse_mask = land_mask.values.flatten()

    flat_fine_mask = griddata((flat_coarse_x, flat_coarse_y), flat_coarse_mask, (flat_fine_x, flat_fine_y), method='linear', fill_value=np.nan)
    fine_mask = flat_fine_mask.reshape((y_bins, x_bins))

    return fine_mask

#create empty arrays to store the interpolated data
U=[[] for i in range(len(ds.time))] 
V=[[] for i in range(len(ds.time))]

#iterate over time to interpolate each 2D scalar field
for i in range(len(ds.time)):
    print('Timesteps computed: '+str(i+1)+'/'+str(len(ds.time)))
    U[i]=interpolation(u[i], M, N)
    V[i]=interpolation(v[i], M, N)


# Create new xarray DataArrays for U and V
lon=np.linspace(ds.longitude.min(), ds.longitude.max(), M)
lat=np.linspace(ds.latitude.min(), ds.latitude.max(), N)

U_da = xr.DataArray(U, coords=[ds.time, lat, lon], dims=['time', 'lat', 'lon'])
V_da = xr.DataArray(V, coords=[ds.time, lat, lon], dims=['time', 'lat', 'lon'])

# Create the dataset with U and V variables
dsf = xr.Dataset({'U': U_da, 'V': V_da})

#%%
"""
SELECTION OF THE VELOCITY COMPONENTS AT THE BOUNDARIES
"""

#zonal velocity
uN=dsf.U.sel(lat=dsf.lat[-1]).T #north
uW=dsf.U.sel(lon=dsf.lon[0]).T #west
#uS=u.sel(latitude=ds.latitude[0]) #south *not necessary if land boundary*
uE=dsf.U.sel(lon=dsf.lon[-1]).T #east

#meridional velocity
vN=dsf.V.sel(lat=dsf.lat[-1]).T #north
vW=dsf.V.sel(lon=dsf.lon[0]).T #west
#vS=v.sel(latitude=ds.latitude[0]) #south
vE=dsf.V.sel(lon=dsf.lon[-1]).T #east

#%%
"""READING BND FILE TO GET GRID COORDINATES OF THE POINTS WHERE
BOUNDARY IS SPECIFIED"""


file = 'insert/path/here/outer.bnd'
# Define the data types for each column
dtype = [('boundary', 'U10'), ('col1', 'U1'), ('col2', 'U1'), ('end1', int),
         ('start2', int), ('end2', int), ('value', int), ('type', 'U10')]

# Read the file using genfromtxt
data = np.genfromtxt(file, dtype=dtype)

# Extract name of boundary 
boundary_names = data['boundary'] 

# Count the number of elements containing 'North'
N_bins = np.count_nonzero(np.char.startswith(boundary_names, 'North'))
# Count the number of elements containing 'East'
E_bins = np.count_nonzero(np.char.startswith(boundary_names, 'East'))
# Count the number of elements containing 'West'
W_bins = np.count_nonzero(np.char.startswith(boundary_names, 'West'))
# Count the number of elements containing 'South'
#S_bins = np.count_nonzero(np.char.startswith(boundary_names, 'South'))

#extract lon & lat for edges of boundaries
lon1, lat1, lon2, lat2 = data['end1'], data['start2'], data['end2'], data['value'] 

#%%

#specify output file directory and name
dir='insert/path/here/'
outfile_name = dir+'Venezuela-def.bct'
outfile = open(outfile_name, 'w')

#%%

time=np.arange(0, len(ds.time))
#iterate over each boundary section
for s, key in enumerate(boundary_names):
    print("Progress: "+str(s)+'/'+str(len(boundary_names)))

    #preamble for boundary section
    outfile.write('table-name          '+"'Boundary Section : "+str(s+1)+"'")
    outfile.write('\n')
    outfile.write('contents            '+"'uniform'")
    outfile.write('\n')
    outfile.write('location            '+"'"+key+"'")
    outfile.write('\n')
    outfile.write('time-function       '+"'non-equidistant'")
    outfile.write('\n')
    outfile.write('reference-time       '+"20220220")
    outfile.write('\n')
    outfile.write('time-unit           '+"'minutes'")   
    outfile.write('\n')
    outfile.write('interpolation       '+"'linear'")
    outfile.write('\n')
    outfile.write('parameter           '+"'time' unit '[min]'")
    outfile.write('\n')
    outfile.write("parameter           'Riemann         (R)  End A' unit '[m/s]'")
    outfile.write('\n')
    outfile.write("parameter           'Riemann         (R)  End B' unit '[m/s]'")
    outfile.write('\n')
    outfile.write("records-in-table     "+str(len(ds.time)))
    outfile.write('\n')

    #spatial grid coordinates of start and end of boundary section
    X1=lon1[s]
    X2=lon2[s]
    Y1=lat1[s]
    Y2=lat2[s]

    
    #selecting v & u values for that boundary section

    #if boundary section covers only one grid in y direction
    if Y1==Y2 or Y2==Y1+1:
        #select grid using array indexing (starts in 0)
        index=Y1-1
        #get u, v values for western and eastern boundary
        uWt=uW[index]
        vWt=vW[index]
        uEt=uE[index]
        vEt=vE[index]
    else: 
        #calculate mean value over y-dimension
        uWt=uW[Y1:Y2].mean('lat')
        vWt=vW[Y1:Y2].mean('lat')
        uEt=uE[Y1:Y2].mean('lat')
        vEt=vE[Y1:Y2].mean('lat')
    

    #if boundary section covers only one grid in x direction
    if X1==X2 or X2==12+1:
        #select grid using array indexing (starts in 0)
        index=X1-1
        #get u, v values for northern and southern boundary
        uNt=uN[index]
        vNt=vN[index]
      #  uSt=uS[index]
      #  vSt=vS[index]
    else:
        uNt=uN[X1:X2].mean('lon')
        vNt=vN[X1:X2].mean('lon')
      #  uSt=uS[X1:X2].mean('lon')
      #  vSt=vS[X1:X2].mean('lon')


    #for each row, write (time, u, v)
    for i in time:
        if i==0: #first row
            outfile.write("{:>{width}.2f}   {:>{width}.4f}   {:>{width}.4f}".format(i*60, 0.0,0.0, width=10))
        else:
            if np.char.startswith(key, 'North')==True: #north boundary
                outfile.write("{:>{width}.2f}   {:>{width}.4f}   {:>{width}.4f}".format(i*60, uNt[i],vNt[i], width=10))
            elif np.char.startswith(key, 'West')==True: #west boundary
                outfile.write("{:>{width}.2f}   {:>{width}.4f}   {:>{width}.4f}".format(i*60, uWt[i],vWt[i], width=10))
            elif np.char.startswith(key, 'East')==True: #east boundary 
                outfile.write("{:>{width}.2f}   {:>{width}.4f}   {:>{width}.4f}".format(i*60, uEt[i],vEt[i], width=10))
        outfile.write('\n')

