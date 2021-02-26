import numpy as np
import xarray as xr
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr   
pandas2ri.activate()
rMBC = importr("MBC")

def MBCn(obs_dat,mod_dat,mod_nf,num_iter):
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
    dat1,dat2,dat3=[],[],[]
    for i in var_list:
        joint_obs = obs_dat[i].values.reshape(-1)
        joint_hist = mod_dat[i].values.reshape(-1)
        joint_proj = mod_nf[i].values.reshape(-1)
        dat1.append(joint_obs)
        dat2.append(joint_hist)
        dat3.append(joint_proj)
    obs1,hist1,proj1 = np.dstack(dat1).reshape(size,var_count),np.dstack(dat2).reshape(size,var_count),np.dstack(dat3).reshape(size,var_count)
    rf = rMBC.MBCn(obs1,hist1,proj1,num_iter,qmap_precalc=False,ratio_seq=np.repeat(False,var_count))
    model_rf,model_nf = np.hsplit(rf[0],var_count),np.hsplit(rf[1],var_count)
    
    ref,pred = {},{}
    for i in range(var_count):
        ref["oc" + str(i)] = model_rf[i].reshape(shape)
    for i in range(var_count):
        pred["pc" + str(i)] = model_nf[i].reshape(shape)
    
    lists1,lists2 = [*ref.keys()],[*pred.keys()]
    oc,pc = mod_dat,mod_nf
    for j,k in enumerate(var_list):
        oc[k] = (oc[k]*0+1)*ref[lists1[j]]
        pc[k] = (pc[k]*0+1)*pred[lists2[j]]
        
    end = time.time()
    print("▮▮▮ Elapsed time in real time :" , time.strftime("%M",time.gmtime(end-start)),"minutes ▮▮▮")
    
    return(oc,pc)
