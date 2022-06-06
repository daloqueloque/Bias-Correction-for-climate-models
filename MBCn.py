import numpy as np
import xarray as xr
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr   
pandas2ri.activate()
rMBC = importr("MBC")

def MBCn(obs_h,mod_h,mod_p,num_iter):
    import time
    start = time.time()
    '''Calculates for the Multivate Bias Correction using N-pdft of climate model outputs from MBC package by 
    Cannon (2018) <doi:10.1007/s00382-017-3580-6>'''
    
    #### Store input descriptions ####
    var_list,var_count = list(obs_dat.keys()),len(obs_dat.keys())
    shape = np.array(obs_dat[var_list[0]].shape)
    size = np.prod(shape)
    
    #### Inputs must have the same lengths, so check for lengths ####
    print('▮▮▮ Checking if Datasets have equal lengths.....')
    check1 = np.array_equal(np.array(obs[var_list[0]].shape),np.array(obs[var_list[0]].shape))
    check2 = np.array_equal(np.array(obs[var_list[0]].shape),np.array(hist[var_list[0]].shape))
    if check1 == True and check2 == True:
        print('▮▮▮ Datasets have the same dimension lengths, continue executing script.....')
    else:
        print('▮▮▮ Datasets does not have the same dimension lengths, abort executing script and Sayonara!')
        import sys 
        sys.exit()
        
    ### Begin processing inputs and executing MBCn aglorith ####  
    
    ### Create function for MBCn ####    
    def applyMBCnR(*argv,num_iter=20):    
        datList = [arg for arg in argv]
        datArr = np.array_split(datList,3)

        obs_h,mod_h,mod_p, var_cnt = datArr[0].T, datArr[1].T, datArr[2].T, len(datArr[0])
        res = rMBC.MBCn(obs_h,mod_h,mod_p,num_iter,silent=True,qmap_precalc=False,ratio_seq=np.repeat(False,var_cnt))
        hist,proj = [res[0][:,i] for i in np.arange(var_cnt)],[res[1][:,i] for i in np.arange(var_cnt)]
        dat_collect = *hist,*proj

        return dat_collect
    
    ### Apply MBCn function to dataset ###
    def pyMBCn(obs_h,mod_h,mod_p,**kwargs):
        time_hist,time_proj = obs_h['time'].values, mod_p['time'].values
        obs_h,mod_h,mod_p = obs_h.drop('time'),mod_h.drop('time'),mod_p.drop('time')

        datSet_list,var_list = [obs_h,mod_h,mod_p],sorted(list(obs_h.keys()))
        datArr_list = [dat[var] for dat in datSet_list for var in var_list]

        time_sgn_i= [['time']]*len(datArr_list) 
        time_sgn_o = [['time']]*(len(datArr_list) - len(var_list))

        mod_corr = xr.apply_ufunc(applyMBCnR,*datArr_list,kwargs=kwargs,input_core_dims=time_sgn_i,output_core_dims=time_sgn_o,vectorize=True,dask='parallelized')                                                                  

        dset_h, dset_p = [],[]
        for (k,j,var) in zip(np.arange(len(var_list)),np.arange(len(var_list),len(mod_corr)),var_list):
            dset_h.append(xr.Dataset({var:mod_corr[k]}))
            dset_p.append(xr.Dataset({var:mod_corr[j]}))

        mod_corr_h, mod_corr_p = xr.merge(dset_h).round(2).assign_coords({'time':time_hist}).transpose('time','latitude','longitude'),xr.merge(dset_p).round(2).assign_coords({'time':time_proj}).transpose('time','latitude','longitude')
        return mod_corr_h,mod_corr_p
    
    mbc = pyMBCn(obs_h,mod_h,mod_p,num_iter=num_iter)
    end = time.time()
    print("▮▮▮ Elapsed time in real time :" , time.strftime("%M:%S",time.gmtime(end-start)),"minutes ▮▮▮")
    
    return(mbc)
