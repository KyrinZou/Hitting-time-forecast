# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:49:23 2020

@author: Haolin Zou


"""
#%%
from os import chdir
chdir(r'C:\Users\Kyrin\OneDrive\桌面\Nino3.4 Forcast')
#%%
import numpy as np
import pandas as pd
import netCDF4
import matplotlib.pyplot as plt
#from scipy.stats import spearmanr
#%% Read Data and preprocessing
dt=netCDF4.Dataset('cleanPlumes.nc','r')
real=pd.read_csv('real.csv')
tbl=dt.variables['nino'] #projections start from FMA 2002
real_history=real['observedNino3.4']  #real_path contains 218 real historical nino3.4 temperatures, starting from JFM 2002.
real_history=np.array(real_history)
real_history=real_history[1:]  #Delete the first JFM to align with real path

#%%Plot a model map, 1=inactive, 0=active
import seaborn as sns
tbl_clean=tbl[:,:,np.setdiff1d(np.arange(42),[11,14])]
#Deleted CPC CONSOL which is an ensemble median itself and CSI-IRI-MM which is redundant
M=len(tbl_clean[0,0,:])
T=len(tbl_clean)
model_map=np.zeros((M,T))
model_names=['AUS/ACCESS',
'AUS/POAMA',
'BCC_CSM11m',
'BCC_RZDM',
'CDC LIM',
'CMC CANSIP',
'COLA ANOM',
'COLA CCSM3',
'COLA CCSM4',
'CPC CA',
'CPC CCA',
'CPC MRKOV',
'CS-IRI-MM',
'CSU CLIPR',
'ECHAM/MOM',
'ECMWF',
'ESSIC ICM',
'FSU REGR',
'GFDL CM2.1',
'GFDL CM2.5',
'GFDL FLOR',
'IAP-NN',
'IOCAS ICM',
'JMA',
'JPN-FRCGC',
'KMA SNU',
'LDEO',
'MetFRANCE',
'NASA GMAO',
'NCEP CFS',
'NCEP CFSv2',
'NTU CODA',
'PSD-CU LIM',
'SAUDI-KAU',
'SCRIPPS',
'SINTEX-F',
'UBC NNET',
'UCLA-TCD',
'UKMO',
'UNB/CWC',
]
for t in range(T):
    model_map[:,t]=np.int32(tbl_clean[t,0,:].mask)

plt.figure(figsize=(18,12))

hmp=sns.heatmap(model_map)
plt.yticks(np.arange(len(model_names)),model_names,rotation=45)
plt.title('Model map')
plt.xlabel('month')
plt.ylabel('model')

   

#%%
def find_nino(path,b):
    '''
    A function computing hitting time under given boundary b. 
    Returns len(path)+1 if no boundary-crossing occurs.

    Parameters
    ----------
    path :  1D array-like
        the trajectory.
        
    b : float
        The boundary.

    Returns
    -------
    integer
        The index of the first hitting from below.

    '''
    t=0
    while t < len(path):
        if path[t]>=b:
            return t+1
        else: t+=1
    return t+1

#%%
#A graph of (a subset of) the data
t=1
b=0.5
L=9
plt.figure(figsize=(9,6))
path_real = real_history[t-1:t+9]
plt.plot(path_real,color='black',label='observed')
init=real_history[t-1:t]
projections=tbl_clean[t,:,:]
projections=np.transpose(projections)
projections = projections[~projections.mask.any(axis=1)] 
xtick=['FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ','DJF']
plt.xticks([0,1,2,3,4,5,6,7,8,9,10],xtick)
taus=[]
for i in range(len(projections)):
    proj=projections[i]
    proj=np.ma.append(init,proj)
    taus.append(find_nino(proj,b))
    if i==0:
        plt.plot(proj,color='aqua',label='March 2002 Plume')
    else: 
        plt.plot(proj,color='aqua')
path_avg=np.ma.mean(projections,axis=0)
path_avg=np.ma.append(init,path_avg)
tau_real = find_nino(path_real,b)
tau_avg = find_nino(path_avg,b)
avgtau_ce = np.mean(taus)
if avgtau_ce > L: avgtau_ce = L+1
plt.plot(path_avg,color='blue',label='ensemble average')

plt.hlines([0.5],0,10,colors='black',linestyles='--',linewidths=0.5,label='boundary')
plt.vlines(x=[tau_real-1,tau_avg-1,avgtau_ce-1], ymin=[-0.5,-0.5,-0.5], ymax=[0.5,0.5,0.5],
           linestyles='--',linewidth=0.5,colors='black')
#Add markers to taus, tau_real, HAI and EAH
plt.scatter(np.array(taus)-1,[0.5 for i in range(9)],s=30,marker='+',c='black',label='hitting time of each projection')
plt.plot(tau_real-1,0.5,marker='D',color='black')
plt.text(tau_real-0.7,0.55,r'$\tau^{(real)}$', fontsize=15)
plt.plot(tau_avg-1,0.5,marker='D',color='black')
plt.text(tau_avg-0.7,0.55,r'$EAH$', fontsize=12)
plt.plot(avgtau_ce-1,0.5,marker='D',color='black')
plt.text(avgtau_ce-0.7,0.55,r'$HAI$', fontsize=12)
plt.xlabel('Months')
plt.ylabel('Oceanic Nino Index (ONI), degree')
plt.ylim(-0.5,1.5)
plt.legend()
plt.show()
#%%
figure1=pd.DataFrame(projections) 
figure1.to_csv('figure1.csv')      


#%% P and MSE components
#boundaries
bs=[0.5,1.0,1.5] 
L=9
#Empty arrays to store the results of crossing probabilities and MSE components.
#The three components are {TP,FN,FP}. {TN} is not recorded since it is just the complement of the first three.
count=np.zeros((3,6,3))  
SSE_components=np.zeros((3,6,3))
for i in range(3):
    b=bs[i]
    
    for t in range(209):
        path_real=real_history[t:t+L] #real path, sliced to length = 9 to align with forecast trajectories (projections).
        paths=np.transpose(tbl_clean[t,:,:]) #42*9, projections in masked-array form 
        paths = paths[~paths.mask.any(axis=1)]
        taus=np.array([find_nino(path,b) for path in paths])  
        
        #calculate the six estimators based on taus
        avgtau_ce=np.mean(taus) #could be between L and L+1
        medtau_ce=np.median(taus)
        
        taus=np.ma.masked_values(taus,L+1)
        avgtau_tr=np.ma.mean(taus)  
        medtau_tr=np.ma.median(taus)
        if np.ma.is_masked(avgtau_tr): ##all unobserved hitting time are converted to 10.
            avgtau_tr=L+1
        if np.ma.is_masked(medtau_tr):
            medtau_tr=L+1
        
        path_avg=np.mean(paths,axis=0)
        path_med=np.median(paths,axis=0)
        
        tau_real=find_nino(path_real,b)
        tau_avg=find_nino(path_avg,b)
        tau_med=find_nino(path_med,b)
        
        estimators=[tau_real,avgtau_ce,tau_avg,medtau_ce,tau_med, avgtau_tr, medtau_tr]
        #Calculate the counts and MSE component for each estimator
        for j in range(6):
            est=estimators[j+1]
            if est<L+1:
                if tau_real<L+1:
                    SSE_components[i,j,0]+=(tau_real-est)**2
                    count[i,j,0]+=1
                else:
                    SSE_components[i,j,2]+=(9-est)**2
                    count[i,j,2]+=1
            elif tau_real<L+1:
                    SSE_components[i,j,1]+=(tau_real-9)**2
                    count[i,j,1]+=1

P_components=count/209
MSE_components=SSE_components/209
#%%
# x_loc=np.array([1,2,3,4,5,6])
# width=0.4
# plt.figure(figsize=(6,4))

# plt.bar(x_loc, P_components[1,:,0],width=width,linewidth=1,edgecolor='black',color='g',label='TP')
# plt.bar(x_loc, P_components[1,:,1],bottom=P_components[1,:,0],width=width,linewidth=1,edgecolor='black',color='b',label='FN')
# plt.bar(x_loc, P_components[1,:,2],bottom=P_components[1,:,0]+P_components[1,:,1],width=width,linewidth=1,edgecolor='black',color='y',label='FP')
# plt.bar(x_loc, 1-P_components[1,:,0]-P_components[1,:,1]-P_components[1,:,2],bottom=P_components[1,:,0]+P_components[1,:,1]+P_components[1,:,2],width=width,linewidth=1,edgecolor='black',color='0.8',label='TN')
# plt.hlines(P_components[1,0,0]+P_components[1,0,1],0,6.5, linewidth=1.2,linestyle='dashed',colors='black')
# plt.xlim(0.5,6.5)
# plt.xticks(x_loc,['EAH','EMH', 'HAC','HAT','HMC','HMT'])
# #plt.title('Probability of TP, FP, FN of all estimators under b=0.5, 1.0 and 1.5')
# plt.yticks([0,0.2,0.4,0.6,0.8,1.0],['0','20%','40%','60%','80%','100%'])
# plt.ylabel('Cumulative Percentage')
# plt.legend(loc='upper left')
# plt.show()
#%%Probability decomposition bar plots
x_loc=np.array([1,2,3,4,5,6])
width=0.2
plt.figure(figsize=(6,4))
plt.bar(x_loc-width, P_components[0,:,0],width=width,color='g',linewidth=1,edgecolor='black',label='TP')
plt.bar(x_loc-width, P_components[0,:,1],bottom=P_components[0,:,0],width=width,color='b',linewidth=1,edgecolor='black',label='FN')
plt.bar(x_loc-width, P_components[0,:,2],bottom=P_components[0,:,0]+P_components[0,:,1],width=width,color='y',linewidth=1,edgecolor='black',label='FP')
plt.bar(x_loc-width, 1-P_components[0,:,0]-P_components[0,:,1]-P_components[0,:,2],bottom=P_components[0,:,0]+P_components[0,:,1]+P_components[0,:,2],width=width,color='0.8',linewidth=1,edgecolor='black',label='TN')

plt.bar(x_loc, P_components[1,:,0],width=width,linewidth=1,edgecolor='black',color='g')
plt.bar(x_loc, P_components[1,:,1],bottom=P_components[1,:,0],width=width,linewidth=1,edgecolor='black',color='b')
plt.bar(x_loc, P_components[1,:,2],bottom=P_components[1,:,0]+P_components[1,:,1],width=width,linewidth=1,edgecolor='black',color='y')
plt.bar(x_loc, 1-P_components[1,:,0]-P_components[1,:,1]-P_components[1,:,2],bottom=P_components[1,:,0]+P_components[1,:,1]+P_components[1,:,2],width=width,linewidth=1,edgecolor='black',color='0.8')

plt.bar(x_loc+width, P_components[2,:,0],width=width,linewidth=1,edgecolor='black',color='g')
plt.bar(x_loc+width, P_components[2,:,1],bottom=P_components[2,:,0],width=width,linewidth=1,edgecolor='black',color='b')
plt.bar(x_loc+width, P_components[2,:,2],bottom=P_components[2,:,0]+P_components[2,:,1],width=width,linewidth=1,edgecolor='black',color='y')
plt.bar(x_loc+width, 1-P_components[2,:,0]-P_components[2,:,1]-P_components[2,:,2],bottom=P_components[2,:,0]+P_components[2,:,1]+P_components[2,:,2],width=width,linewidth=1,edgecolor='black',color='0.8')

plt.xticks(x_loc,['HAI', 'EAH', 'HMI', 'EMH', 'HAD', 'HMD'])
#plt.title('Probability of TP, FP, FN of all estimators under b=0.5, 1.0 and 1.5')
plt.yticks([0,0.2,0.4,0.6,0.8,1.0],['0','20%','40%','60%','80%','100%'])
plt.hlines([P_components[0,0,0]+P_components[0,0,1],P_components[1,0,0]+P_components[1,0,1],P_components[2,0,0]+P_components[2,0,1]],0.5,6.5, linewidth=1.2,linestyle='dashed',colors='black')
plt.xlim(0.5,6.5)

plt.ylabel('Cumulative Percentage')
plt.legend(loc='lower right')
plt.show()
#%%
# figure2_1=pd.DataFrame(P_components[0,:,:])
# figure2_1.to_csv('figure2_1.csv')
# figure2_2=pd.DataFrame(P_components[1,:,:])
# figure2_2.to_csv('figure2_2.csv')
# figure2_3=pd.DataFrame(P_components[2,:,:])
# figure2_3.to_csv('figure2_3.csv')
#%%MSE decomposition bar plot
width=0.2
plt.figure(figsize=(6,4))
plt.bar(x_loc-width, MSE_components[0,:,0],width=width,color='g',linewidth=1,edgecolor='black',label='TP')
plt.bar(x_loc-width, MSE_components[0,:,1],bottom=MSE_components[0,:,0],width=width,color='b',linewidth=1,edgecolor='black',label='FN')
plt.bar(x_loc-width, MSE_components[0,:,2],bottom=MSE_components[0,:,0]+MSE_components[0,:,1],width=width,color='y',linewidth=1,edgecolor='black',label='FP')

plt.bar(x_loc, MSE_components[1,:,0],width=width,linewidth=1,edgecolor='black',color='g')
plt.bar(x_loc, MSE_components[1,:,1],bottom=MSE_components[1,:,0],width=width,linewidth=1,edgecolor='black',color='b')
plt.bar(x_loc, MSE_components[1,:,2],bottom=MSE_components[1,:,0]+MSE_components[1,:,1],width=width,linewidth=1,edgecolor='black',color='y')

plt.bar(x_loc+width, MSE_components[2,:,0],width=width,linewidth=1,edgecolor='black',color='g')
plt.bar(x_loc+width, MSE_components[2,:,1],bottom=MSE_components[2,:,0],width=width,linewidth=1,edgecolor='black',color='b')
plt.bar(x_loc+width, MSE_components[2,:,2],bottom=MSE_components[2,:,0]+MSE_components[2,:,1],width=width,linewidth=1,edgecolor='black',color='y')

plt.xticks(x_loc,['HAI', 'EAH', 'HMI', 'EMH', 'HAD', 'HMD'])
#plt.title('MSE Components of TP, FP, FN of all estimators under b=0.5, 1.0 and 1.5')
plt.ylabel('Cumulative MSE, unit=month^2')
plt.legend(loc='upper left')
plt.show()
#%%
# figure2_4=pd.DataFrame(MSE_components[0,:,:])
# figure2_4.to_csv('figure2_4.csv')
# figure2_5=pd.DataFrame(MSE_components[1,:,:])
# figure2_5.to_csv('figure2_5.csv')
# figure2_6=pd.DataFrame(MSE_components[2,:,:])
# figure2_6.to_csv('figure2_6.csv')
#%%
# width=0.2
# GMSE=np.sum(SSE_components, axis=2)/np.sum(count,axis=2)

# plt.figure(figsize=(6,4))
# plt.bar(x_loc-width,GMSE[0,:],width=width,color='grey',linewidth=1,edgecolor='black')

# plt.bar(x_loc,GMSE[1,:],width=width,color='grey',linewidth=1,edgecolor='black')

# plt.bar(x_loc+width,GMSE[2,:],width=width,color='grey',linewidth=1,edgecolor='black')

# plt.xticks(x_loc,['EAH','EMH', 'HAI','HAD','HMI','HMD'])
# #plt.title('MSE_111 of all estimators under b=0.5, 1.0 and 1.5')
# plt.ylabel('MSE_TNc')
# plt.show()

#%%Correlation and MSE_TP|FN|FP against all b
bs=[i/100 for i in range(200)]  #100 different boundary levels, from 0.00 to 1.00
corrs=[]     #Similar, correlations 
MSEs=[]
for b in bs:
    corr=[]
    SSE=np.zeros(6)
    count=np.zeros(6)
    tau_tbl=[]  #209*5, each column is the value of estimator at each month (total 209 months) in history under current b level
    for t in range(209):
        path_real=real_history[t:t+L] #real path, sliced to length = 9 to align with forecast trajectories (projections).
        paths=np.transpose(tbl_clean[t,:,:]) #42*9, projections in masked-array form 
        paths = paths[~paths.mask.any(axis=1)]
        taus=np.array([find_nino(path,b) for path in paths])  
            #m_paths=np.vstack((m_paths,M_t(proj)))
        
        avgtau_ce=np.mean(taus)
        medtau_ce=np.median(taus)
        
        taus=np.ma.masked_values(taus,L+1)
        avgtau_tr=np.ma.mean(taus)  
        medtau_tr=np.ma.median(taus)
        if np.ma.is_masked(avgtau_tr): ##all unobserved hitting time are converted to 9.
            avgtau_tr=L+1
        if np.ma.is_masked(medtau_tr):
            medtau_tr=L+1
        
        path_avg=np.ma.mean(paths,axis=0)
        path_med=np.ma.median(paths,axis=0)
        
        tau_real=find_nino(path_real,b)
        tau_avg=find_nino(path_avg,b)
        tau_med=find_nino(path_med,b)
        
        estimators=[tau_real,avgtau_ce,tau_avg,medtau_ce,tau_med, avgtau_tr,  medtau_tr]
        #here we have 1 real + 6 estimators (a series of 5 taus) and append to tau_tbl
        tau_tbl.append(estimators)
        for i in range(6):
            if tau_real<L+1 or estimators[i+1]<L+1:
                SSE[i]+=(estimators[i+1]-tau_real)**2
                count[i]+=1
    
                
            
    tau_tbl=np.transpose(np.array(tau_tbl))
    #print(tau_tbl)
    #here we have tau_tbl containing 7 arrays of taus,each length=209.
    #next we copute MSE and Corr then store them
    real_array=tau_tbl[0]
    for i in range (6):
        est_array=tau_tbl[i+1]
        corr.append(np.corrcoef(real_array,est_array)[0][1])
    
    corrs.append(corr)
    MSEs.append(SSE/count)

    print(b*50, '%','completed')
    
#The order of estimators is: HAI, EAH, HMI, EMH, HAD, HMD
corrs=np.transpose(np.array(corrs))
#corrs=corrs[[2,0,1,3,4,5]]
MSEs=np.transpose(np.array(MSEs))
#MSEs=MSEs[[2,0,1,3,4,5]]

#%%Define a function that does moving average to smooth the above results for better readability.
def np_move_avg(a,n,mode="valid"):
    return(np.convolve(a, np.ones((n,))/n, mode=mode))
#%%Correlation plot

bs=[i/100 for i in range(4,196)]
col_seq=['b','g','r','c','y','m']
names=['HAI', 'EAH', 'HMI', 'EMH', 'HAD', 'HMD']
plt.figure(figsize=(6,4))
for i in range(6):
    plt.plot(bs,np_move_avg(corrs[i],9,mode="valid"),color=col_seq[i],label=names[i])
plt.xlabel('boundary, degree')
plt.ylabel('Correlation')
plt.legend(loc='upper left')
plt.show()
#%%Rooted MSE plot
plt.figure(figsize=(6,4))
for i in range(6):
    plt.plot(bs,np_move_avg(np.sqrt(MSEs[i]),9,mode="valid"),color=col_seq[i],label=names[i])
plt.xlabel('boundary, degree')
plt.ylabel('RMSE_TNc, unit=month')
plt.legend()
plt.show()
#%%
# figure3_1=pd.DataFrame(corrs)
# figure3_1.to_csv('figure3_1.csv')
# figure3_2=pd.DataFrame(np.sqrt(MSEs))
# figure3_2.to_csv('figure3_2.csv')

#%%Exchangeability test
from ExchangeabilityTest import exchangeability_test
ex_test_highest=np.array([])
for t in range(209):
    projections=tbl_clean[t,:,:]
    projections=np.transpose(projections)
    projections = projections[~projections.mask.any(axis=1)] # ~ means 'not'
    #projections = np.vstack([ real_history[t:t+9],projections])
    highest=exchangeability_test(projections,traj=False, random=True)
    ex_test_highest = np.append(ex_test_highest,highest)

#%%
plt.figure(figsize=(6,4))
plt.plot(ex_test_highest,linewidth=1.2)
plt.hlines(np.log(20),xmin=0,xmax=209,linewidth=1.2,linestyle='dashed',colors='black')
plt.xlabel('Month')
plt.ylabel('Power martingale maximum (log scale)')

#The result seems interesting. For most months the exchangeability is not rejected. 
#13/209 months (calculated below),have  M>20, and by Doob's inequality we know P(M>20)<=EM/20=1/20=0.05
print(np.sum(ex_test_highest>np.log(20)))


#%%Wilcoxon rank test
from scipy.stats import wilcoxon
wilcoxon_nino_mse = np.ones((6,6))
for i in range(6):
    for j in range(6):
        if j==i: continue
        res = wilcoxon(MSEs[i],MSEs[j],alternative='less').pvalue
        wilcoxon_nino_mse[i][j] = '{:.2e}'.format(res)

wilcoxon_nino_corr = np.ones((6,6))
for i in range(6):
    for j in range(6):
        if j==i: continue
        res = wilcoxon(corrs[i],corrs[j],alternative='greater').pvalue
        wilcoxon_nino_corr[i][j] = '{:.2e}'.format(res)
#%%
x_loc=np.array([1,2,3,4,5,6])-0.5
y_loc=np.array([1,2,3,4,5,6])-0.5
plt.figure(figsize=(6,4))
sns.heatmap(wilcoxon_nino_corr)
plt.xlabel('model')
plt.ylabel('model')
plt.xticks(x_loc,['HAI', 'EAH', 'HMI', 'EMH', 'HAD', 'HMD'])
plt.yticks(y_loc,['HAI', 'EAH', 'HMI', 'EMH', 'HAD', 'HMD'])
#%%
x_loc=np.array([1,2,3,4,5,6])-0.5
y_loc=np.array([1,2,3,4,5,6])-0.5
plt.figure(figsize=(6,4))
sns.heatmap(wilcoxon_nino_mse)
plt.xlabel('model')
plt.ylabel('model')
plt.xticks(x_loc,['HAI', 'EAH', 'HMI', 'EMH', 'HAD', 'HMD'])
plt.yticks(y_loc,['HAI', 'EAH', 'HMI', 'EMH', 'HAD', 'HMD'])

#%%
np.savetxt("nino_wil_corr.csv", wilcoxon_nino_corr, delimiter=' & ', fmt='%2.2e',newline=' \\\\\n')
np.savetxt("nino_wil_rmse.csv", wilcoxon_nino_mse, delimiter=' & ', fmt='%2.2e',newline=' \\\\\n')
































