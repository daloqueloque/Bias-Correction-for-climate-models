import numpy as np
import xarray as xr
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
rMBC = importr("MBC")

def QDM_hist(obs_dat,mod_dat,mod_nf):
    qdm = rMBC.QDM(obs_dat,mod_dat,mod_nf,ratio=False)[0]
    return(qdm)
def QDM_proj(obs_dat,mod_dat,mod_nf):
    qdm = rMBC.QDM(obs_dat,mod_dat,mod_nf,ratio=False)[1]
    return(qdm)
def apply_QDM(obs_dat,mod_dat,mod_nf):
    var_list = list(obs_dat.keys())
    time1 = mod_dat['time'].values
    time2 = mod_nf['time'].values
    hist = xr.apply_ufunc(QDM_hist, obs_dat[var_list[0]].drop('time'), mod_dat[var_list[0]].drop('time'), mod_nf[var_list[0]].drop('time'),
                      input_core_dims=[['time'],['time'],['time']],
                      output_core_dims=[['time']], vectorize=True)
    proj = xr.apply_ufunc(QDM_proj, obs_dat[var_list[0]].drop('time'), mod_dat[var_list[0]].drop('time'), mod_nf[var_list[0]].drop('time'),
                      input_core_dims=[['time'],['time'],['time']],
                      output_core_dims=[['time']], vectorize=True)
    oc = hist.assign_coords({'time':time1}).to_dataset()
    pc = proj.assign_coords({'time':time2}).to_dataset()
    return oc,pc
