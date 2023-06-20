#%%
import cdsapi
import xarray as xr
from urllib.request import urlopen

#%%
# start the client
cds = cdsapi.Client()
# dataset you want to read
dataset = 'reanalysis-era5-single-levels'
# flag to download data
download_flag = False
# api parameters 

fl=cds.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', 'surface_pressure',
        
        ],
        'year': '2022',
        'month': ['03', '04'],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            12.522294933660206, -69.2696812891174, 11.881294933660204,
            -68.5456812891174,
        ],
        'format': 'netcdf',
    },
    'WindCuraçao.nc')

#%%

# download the file 
path='write/path/here'
fl.download(path+"WindCuraçao.nc")
# load into memory
with urlopen(fl.location) as f:
    ds = xr.open_dataset(f.read())
print(ds.keys())
