import numpy as np
import xarray as xr
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as rObj

pandas2ri.activate()
rMBC = importr("MBC")

def QDM_hist(obs_dat,mod_dat,mod_nf,ratio=False,subs=rObj.NULL,ntau=rObj.NULL):
    qdm = rMBC.QDM(obs_dat,mod_dat,mod_nf,ratio=ratio,subsample=subs,n_tau=ntau)[0]
    return(qdm)
def QDM_proj(obs_dat,mod_dat,mod_nf,ratio=False,subs=rObj.NULL,ntau=rObj.NULL):
    qdm = rMBC.QDM(obs_dat,mod_dat,mod_nf,ratio=ratio,subsample=subs,n_tau=ntau)[1]
    return(qdm)

def apply_QDM(obs_dat,mod_dat,mod_nf,**kwargs):
    import time
    start = time.time()
    
    mod_nf = mod_nf.rename({'time':'time2'})
    time1 = mod_dat['time'].values
    time2 = mod_nf['time2'].values
    hist = xr.apply_ufunc(QDM_hist, obs_dat.drop('time'), mod_dat.drop('time'), mod_nf.drop('time2'),
                      kwargs=kwargs,input_core_dims=[['time'],['time'],['time2']],
                      output_core_dims=[['time']], vectorize=True,dask='parallelized')
    proj = xr.apply_ufunc(QDM_proj, obs_dat.drop('time'), mod_dat.drop('time'), mod_nf.drop('time2'),
                      kwargs=kwargs,input_core_dims=[['time'],['time'],['time2']],
                      output_core_dims=[['time2']], vectorize=True,dask='parallelized')
    oc = hist.assign_coords({'time':time1}).transpose('time','latitude','longitude')
    pc = proj.assign_coords({'time2':time2}).transpose('time2','latitude','longitude').rename({'time2':'time'})
    
    end = time.time()
    elapsed = end-start
    if elapsed<60:
        print("▮▮▮ Elapsed time in real time :" , time.strftime("%M:%S",time.gmtime(elapsed)),"seconds ▮▮▮")
    else:
        print("▮▮▮ Elapsed time in real time :" , time.strftime("%M:%S",time.gmtime(elapsed)),"minutes ▮▮▮")
    return oc,pc
