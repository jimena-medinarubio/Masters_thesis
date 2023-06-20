import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

path='/enter/path/here'
ds=xr.open_dataset(path+"trajectories.nc")

#%%
#count number of non-NAN longitude coordinates at each timestep inside domain
count=[[] for i in range(len(ds.obs))]
for i in range(len(ds.obs)):
    npart= np.count_nonzero(~np.isnan(ds.lon.T[i]))
    count[i]=npart

delta=6*24 #hours between each release
num_releases=5
releases=[[] for i in range(num_releases)]

#separate count of trajectories for each release
for i in range(num_releases-1):
   start=i*delta
   end=(i+1)*delta-1
   releases[i]=count[start:end]
   print(end)

#counts for incomplete last release
releases[-1]=count[4*delta:]

#%%
#define the exponential function
def exp(x,A,a, c):

    """INPUT PARAMETERS:
    A= amplitude
    a= rate of decay
    c= background value """

    y=A*np.exp(-x/a)+c
    return y

#initial guesses
A= 10000
a=54
c=10000

rate=[]
rate_err=[]

for i in range(num_releases):
    t=np.arange(0, len(releases[i]))
    po,po_cov= curve_fit(exp, t , releases[i], p0=[A, a,c], absolute_sigma=True)

    print("The signal parameters: release"+str(i))
    print(" A = %.3f +/- %.3f" %(po[0],np.sqrt(po_cov[0,0])))
    print(" a = %.3f +/- %.3f"%(po[1],np.sqrt(po_cov[1,1])))
    print(" c = %.3f +/- %.3f"%(po[2],np.sqrt(po_cov[2,2])))
    
    rate.append(po[1])
    rate_err.append(np.sqrt(po_cov[1,1]))

#%%
residence_time=[-np.log(0.3679)*rate[i] for i in range(len(rate))]
avg_residence_time=np.mean(residence_time)
avg_residence_err=np.sum(rate_err)
print('Average residence time: '+str(avg_residence_time)+'+/-'+str(avg_residence_err))

half_time=[-np.log(0.5)*rate[i] for i in range(len(rate))]
print('Time taken for number of particles to half: '+str(np.mean(half_time)))
