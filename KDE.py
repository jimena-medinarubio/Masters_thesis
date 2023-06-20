#%%

import numpy as np
import xarray as xr
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# %%

#Pacels output of simulated trajectories
file_t = 'trajectories.nc'
ds=xr.open_dataset(file_t)
print(ds.keys)

#Hydodynamic model dataset
#used to perform KDE on its grid
file_v= 'velocity_field.nc'
vel=xr.open_dataset(file_v)

#%%
kde_values = []

total_trajs=ds.obs.size

# loop over each observation in the dataset
for i in range(ds.obs.size):
    print('Processed: '+i+'/'+total_trajs+'trajectories')

    lon=ds.lon[:, i].where( ((~np.isnan(ds.lon[:, i]))), drop=True)
    lat=ds.lat[:, i].where( ((~np.isnan(ds.lat[:, i]))), drop=True)

    if not np.isnan(lon).all():
        #extract the lon/lat values for the current observation
        lon_lat  = np.vstack([lon, lat]).T
        #drop samples with missing values from lon_lat
        lon_lat = lon_lat[~np.isnan(lon_lat).any(axis=1)]

        # dreate a 2D grid of lat/lon values to evaluate the KDE on based on
        # grid from hydrodynamic model
        x, y = np.meshgrid(vel['XCOR'][:, 0].values, vel['YCOR'][0,:].values)

        xy = np.column_stack([x.ravel(), y.ravel()])
    
        #initialize a KDE object and fit it to the current observation's lon/lat values
        kde = KernelDensity( kernel="gaussian", algorithm="ball_tree")
        #suggest a range of possible values of bandwidth & find best fit
        bandwidth = np.arange(0.01, 0.1, 0.05)
        grid = GridSearchCV(kde, {'bandwidth': bandwidth})
        grid.fit(lon_lat)
        kde = grid.best_estimator_

        #specifying the bandwidth
        #bandwidth=0.01
        #kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian", algorithm="ball_tree")
        #kde.fit(lon_lat)

        #evaluate the KDE on the 2D grid created 
        log_density = kde.score_samples(xy)
    
        density = np.exp(log_density)
        #normalise results
        density /= density.sum()
        z = density.reshape(x.shape)
        #append the KDE values to the list
        kde_values.append(z)
    else:
        t=np.zeros_like(vel['XCOR'].values.T)
        kde_values.append(t)

#combine the KDE values into a DataArray
kde_values = np.stack(kde_values)
kde_da = xr.DataArray(kde_values,
                    dims=("obs", "lat", "lon"),
                    coords={"obs": ds.obs.values,
                            "lat": y[:, 0],
                            "lon": x[0, :]},
                    name="kde")

#compute the cumulative sum of the particle distribution over time
kde_cum=np.sum(kde_da, axis=0)

kde_cum = kde_cum.where(kde_cum >= 1e-6, np.nan)/np.nansum(kde_cum)
