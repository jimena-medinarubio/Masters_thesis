#Parcels simulation: initial position of particles in structured grid covering entire domain
#credits to @vesnaber for firt part of code manipulating the output from DELFT3D to use in Parcels 
#(not yet available in Parcels new version)

# %%
#import modules

import parcels
import numpy as np
from parcels import FieldSet, AdvectionRK4_3D, ParticleFile, ParticleSet, ErrorCode, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile, NestedField, Field
import numpy as np
import xarray as xr
import math
import matplotlib.pyplot as plt
from datetime import timedelta as delta
from datetime import datetime
from operator import attrgetter
from scipy.ndimage import binary_dilation
from scipy.interpolate import griddata


# %%
#BEGIN CODE FROM: @vesnaber

path='/insert/path/here'
ds = xr.open_dataset(path)[["U1", "V1", "S1", 'DP0']]

#coercing into mitgcm format (where everything is from the reference point of the bottom left gridpoint for a cell)
ds = ds.assign(
    U1 = ds.U1.swap_dims({"N":"NC"}),
    V1 = ds.V1.swap_dims({"M":"MC"}),
    S1 = ds.S1.swap_dims({"N":"NC", "M": "MC"}),
    XZ = ds.XZ.swap_dims({"N":"NC", "M":"MC"}),
    YZ = ds.YZ.swap_dims({"N":"NC", "M":"MC"})
)[["U1", "V1", "S1", 'DP0']].drop(["XZ", "YZ"]).isel(KMAXOUT_RESTR=0) # Dropping extraneous coordinates

xcor = ds.XCOR.values

#adding in coordinates for the axes
ds = ds.assign(
    x = ds.XCOR.isel(NC=0),
    y = ds.YCOR.isel(MC=0),
).set_coords(["x", "y"])

ds = ds.transpose("time", "NC", "MC", ...) # Transposing xarray dataset as per https://github.com/OceanParcels/parcels/issues/1180
ds = ds.rename({"U1": "U", "V1": "V", "S1": "S", "XCOR": "x_mesh", "YCOR": "y_mesh", "MC": "x_index", "NC": "y_index", 'DP0': 'depth'}) # Renaming variables

u=ds.U.values
v=ds.V.values
x_mesh=ds.x_mesh.values
y_mesh=ds.y_mesh.values
x=x_mesh[0, :]
y=y_mesh[ :, 0]
depth=ds.depth.values

ds=xr.Dataset(#create the dataset
    {
        'U': (['time', 'y', 'x'], u),
        'V': (['time', 'y', 'x'], v), 'x_mesh': (['y', 'x'], x_mesh),
        'y_mesh': (['y', 'x'], y_mesh), 'depth': (['y', 'x'], depth)
    },
    coords={ 'x': x, 'y': y, 'time': ds.time})

# define fieldset
fieldset = FieldSet.from_xarray_delft3d(
     ds,
     variables,
     dimensions, 
     mesh="spherical", allow_time_extrapolation=True,
     interp_method="cgrid_velocity",
     gridindexingtype="mitgcm", time=np.arange(0, len(ds.time))
     )
#end contribution

# %%
land_mask=ds.x.where(ds.depth<0).T.values

#extend the land mask by one grid cell using binary dilation
extended_mask = binary_dilation(~np.isnan(land_mask), iterations=1)

#apply the extended mask to the original land mask
land_mask_extended = np.where(extended_mask, 100, np.nan)
ocean_mask=np.where(np.isnan(land_mask_extended), -ds.depth.T , land_mask_extended)

bathymetry_field = Field("bathymetry", data=ocean_mask, lon=ds.x.values, lat=ds.y.values)
fieldset.add_field(bathymetry_field)

#%%

X=np.linspace(ds.x.min(), ds.x.max(), int(len(ds.x)/6))
Y=np.linspace(ds.y.min(), ds.y.max(), int(len(ds.y)/6))
#one less grid cell for particle release
lons, lats=np.meshgrid(X, Y)

def interpolation(land_mask, x_bins, y_bins):
    #define fine grid coordinates
    fine_x, fine_y = np.meshgrid(np.arange(0, land_mask.shape[1]), np.arange(0, land_mask.shape[0]))
    #define coarse grid coordinates
    coarse_x, coarse_y = np.meshgrid(np.linspace(0, land_mask.shape[1], x_bins), np.linspace(0, land_mask.shape[0], y_bins))
    #flatten fine grid coordinates and mask
    flat_x, flat_y, flat_mask = fine_x.flatten(), fine_y.flatten(), land_mask.ravel()
    #interpolate mask onto coarse grid
    coarse_mask = griddata((flat_x, flat_y), flat_mask, (coarse_x, coarse_y), method='nearest')
    return coarse_mask

#interpolate extended land mask to grid of initial position of particles
land_mask_extended_interpolated=interpolation(land_mask_extended, int(len(ds.x)/6), int(len(ds.y)/6))
land_mask_extended_interpolated[0]=100
land_mask_extended_interpolated[:, 0]=100

#set lon_grid to NaN where filtered.mask is NaN
lon_grid = np.where(~np.isnan(land_mask_extended_interpolated.round(decimals=10)), np.nan, lons)
lat_grid = np.where(~np.isnan(land_mask_extended_interpolated.round(decimals=10)), np.nan, lats)


#%%
initial_time = datetime(2022, 3, 1)  #set the initial time for spin-up time

#set the time for the first release of particles
time_of_first_release = initial_time + delta(days=2)  #adjust the time offse


#%%
class SampleParticleInitZero(JITParticle):            #define a new particle class
    bathymetry = Variable('bathymetry', initial=0)  #variable 'bathymetry' initially zero

pset = ParticleSet(fieldset=fieldset, pclass=SampleParticleInitZero, 
                   lon=lon_grid, lat=lat_grid, 
                   time=time_of_first_release, repeatdt=delta(days=6))

def delete_out_of_bounds(particle, fieldset, time):
    particle.delete()  # delete particle if out of bounds

def SampleZ(particle, fieldset, time):
    particle.bathymetry = fieldset.bathymetry[time, particle.depth, particle.lat, particle.lon]

def delete_land(particle, fieldset, time):
    if particle.bathymetry>0:
        particle.delete()

sample_kernel = pset.Kernel(SampleZ)
land_kernel=pset.Kernel(delete_land)

#pset.show(field=fieldset.bathymetry, cmap='deep', 
          show_time=0, title=None)

#%%
path='writhe/path/here'
output_filename = "trajectory_original.nc"
output_file = pset.ParticleFile(name=path+output_filename,  outputdt=timedelta(hours=1))
kernels = AdvectionRK4 + sample_kernel+ land_kernel

pset.execute(kernels,                 # kernel (which defines how particles move)
             runtime=timedelta(days=29),    # total length of the run
             dt=timedelta(hours=1),      # timestep of the kernel
            recovery={ErrorCode.ErrorOutOfBounds: delete_out_of_bounds}, 
             output_file=output_file)

output_file.export()

#%%
#manipulation of data to relocate longitude and latitude values according to their releasing time

ds=xr.open_dataset(output_filename)

lons=np.zeros_like(ds.lon)
lats=np.zeros_like(ds.lat)

#target datetime
target_datetime = initial_time

#calculate the time difference in hours
diff = (ds.time[:, 0].values.astype('datetime64[s]') - np.datetime64(target_datetime)).astype('timedelta64[s]').astype(int) / 3600

for i, (I, J, times) in enumerate(zip(ds.lon, ds.lat, ds.time)):
    I_values = I.values
    J_values = J.values
    times_values = times.values

    #remove trailing zeros from the fractional seconds part and handle NaT values
    datetime_str = np.where(times_values != np.datetime64('NaT'), np.char.rstrip(np.char.rstrip(times_values.astype(str), '0'), '.'), 'NaT')
    datetime_obj = np.array([datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S') if dt != 'NaT' else np.datetime64('NaT') for dt in datetime_str])
    
    hours_diff = (datetime_obj.astype('datetime64[s]') - np.datetime64(target_datetime, 's')).astype('timedelta64[s]').astype(int) / 3600
    
    mask = np.isclose(hours_diff, -2.56204779e+15, atol=10**-6)
    hours_diff = np.where(mask, np.nan, hours_diff)
    
    valid_indices = ~np.isnan(times_values) & (datetime_str != 'NaT')

    g = []
    for t in hours_diff:
        if not np.isnan(t):
            g.append((t-diff[i]).astype(int))

    lons[i, hours_diff[valid_indices].astype(int)] = I_values[g]
    lats[i, hours_diff[valid_indices].astype(int)] = J_values[g]

print('done')   

lats_nan=[[] for i in range(len(ds.traj))]
lons_nan=[[] for i in range(len(ds.traj))]
for i in range(len(ds.traj)):
    lats_nan[i]=np.where(lats[i]==0, np.nan, lats[i])
    lons_nan[i]=np.where(lons[i]==0, np.nan, lons[i])


parcels=xr.Dataset(
{ 'lon': (['traj', 'obs'], lons_nan), 
 'lat': (['traj', 'obs'], lats_nan), 
 'z': (['traj', 'obs'], ds.z.values),  'time': (['traj', 'obs'], ds.time.values),
  'trajectory': (['traj', 'obs'], ds.trajectory.values), 'bathymetry': (['traj', 'obs'], ds.bathymetry.values)},
  coords={ 'traj': ds.traj.values, 'obs': ds.obs.values} )

path='/write/path/here'
parcels.to_netcdf(path+'trajectories.nc')
