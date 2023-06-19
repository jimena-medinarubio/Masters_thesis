#%%
import numpy as np
import xarray as xr

#%%
regions=xr.open_dataset('regions.nc')
itineraries=xr.open_dataset('itineraries.nc')

#%%

def surface_area(regions):

    #calculate surface area of each grid cell
    dlat = np.radians(np.diff(regions.lat)[0])
    dlon = np.radians(np.diff(regions.lon)[0])
    def area(dlat, dlon, R=6.371e6):
        dx=dlon*R*np.cos(np.radians(regions.lat.mean()))
        dy=dlat*R
        return dx*dy #meters^2 in each cell
    area_grid_cell=area(dlat, dlon)

    surface_area=[] #meters^2

    labels=np.unique(regions.label)

    for i in labels[1:]:
        mask=regions.label.where(regions.label==i, drop=True)
        a=np.sum(mask*area_grid_cell)
        surface_area.append(a)
    
    return surface_area


def exposure_time(regions, itineraries, surface_area):

    """
    Calculate the mean exposure time in each region from the itineraries

    Input:
    -regions: Xarray dataset mapping a two dimensional space with lon and lat coordinates
    and a variable assigning a label to each grid point.
    -itineraries: matrix of the itineraries of each trajectory, specifying the 
    label of the regions where they are located at each timestep

    Output:
    Numpy array with M elements, corresponding to the mean normalised
    exposure time in each region
    """

    #create empty array to store results
    k=len(np.unique(regions.label))
    exposure_time=[[] for _ in range(k)]
    
    #iterate over each trajectory
    for traj in range(len(itineraries)):
        
        #obtain histogram for repetitions of labels inside itineraries
        hist=np.histogram(itineraries[traj], bins=np.arange(k+1)) 

        #append result of each region to exposure time matrix
        for label, counts in enumerate(hist):
            exposure_time[label].append(counts)

        print(f"Processed {traj+1}/{len(itineraries)} trajectories")
    
    total_area=np.sum(surface_area)#km
    #calculate mean_exposure_time of all regions excluding land
    mean_exposure_time=np.nanmean(exposure_time[1:], axis=1)/surface_area*total_area
    percentage_mean_exposure_time=mean_exposure_time/np.sum(mean_exposure_time)*100
    
    return percentage_mean_exposure_time

#%%
area=surface_area(regions)
mean_exposure_time=exposure_time(regions, itineraries, area)

