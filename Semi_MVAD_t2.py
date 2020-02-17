# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:17:49 2019

@author: Administrator
"""

import numpy as np
from numpy import genfromtxt
data = genfromtxt('pima_scale.csv', delimiter = ',')

from scipy.linalg import block_diag
from numpy.linalg import inv
from sklearn import metrics
from funpack import EVALAD

def loglikelyhood(X,mu,cov):
    X_cent = X - mu
    exponent = -0.5*np.diag(np.matmul(np.matmul(X_cent,inv(cov)),X_cent.T))
    return exponent

itrs = 20
ratios = [0.05]
dims = [1,2,3,4,5,6]
AUC_mean = []
AUC_std = []
for ratio_idx in range(len(ratios)):
    for dim_idx in range(len(dims)):
        aucs_probM = np.zeros(itrs)
        fars_probM = []
        drs_probM = []
        for itr in range(itrs):
            n = data.shape[0]
            sample = np.copy(data[0:n,0:-1])
            label = np.copy(data[0:n,-1])
            outlier_ratio = ratios[ratio_idx]
            n_feats = sample.shape[1]    
            n_feats_SView = n_feats//2  
            
            # type-2 outliers (sample has different behaviour under multiple views)
            X_Pos = sample[label==0,:]
            X_Neg = sample[label==1,:]
            n_outliers = int(n*outlier_ratio)
            idx1 = np.random.permutation(X_Pos.shape[0])[:n_outliers]
            idx2 = np.random.permutation(X_Neg.shape[0])[:n_outliers]
            temp = X_Pos[idx1,0:n_feats_SView]
            X_Pos[idx1,0:n_feats_SView] = X_Neg[idx2,0:n_feats_SView]
            X_Neg[idx2,0:n_feats_SView] = temp
            label_ProbM = -np.ones(n)
            label_ProbM[idx1] = 1
            label_ProbM[X_Pos.shape[0]+idx2] = 1
            X = np.concatenate((X_Pos,X_Neg),axis=0)
            
            mask = np.ones(n,dtype=bool)
            mask[idx1] = False
            mask[X_Pos.shape[0]+idx2] = False
            x_nor = X[mask]
            
            x_nor_train = x_nor[0:int(0.5*x_nor.shape[0]),:]
            x_nor_test = x_nor[int(0.5*x_nor.shape[0]):,:]
            
            # abnormal data
            mask_ab = np.zeros(n,dtype=bool)
            mask_ab[idx1] = True
            mask_ab[X_Pos.shape[0]+idx2] = True
            x_abnor = X[mask_ab]
            
            # test data
            x_test = np.concatenate((x_nor_test,x_abnor),axis=0)
            label_test = -np.ones(x_test.shape[0])
            label_test[x_nor_test.shape[0]:] = 1
            
            x_train_view1 = x_nor_train[:,0:n_feats_SView]
            x_train_view2 = x_nor_train[:,n_feats_SView:]
            m = dims[dim_idx]
            n_train = x_train_view1.shape[0]
            
            # calculate mean and covariance of view 1
            view1_shape = x_train_view1.shape
            d1 = view1_shape[1]
            mu_view1 = np.mean(x_train_view1,axis=0)
            x_train_view1_cent = x_train_view1 - mu_view1
            cov_view1 = 1/(view1_shape[0])*np.matmul(x_train_view1_cent.T,x_train_view1_cent)
            W1 = np.ones([d1,m])
            Phi1 = np.identity(d1)
            
            # calculate mean and covariance of view 2
            view2_shape = x_train_view2.shape
            d2 = view2_shape[1]
            mu_view2 = np.mean(x_train_view2,axis=0)
            x_train_view2_cent = x_train_view2 - mu_view2
            cov_view2 = 1/(view2_shape[0])*np.matmul(x_train_view2_cent.T,x_train_view2_cent)
            W2 = np.ones([d2,m])
            Phi2 = np.identity(d2)
        
            I_m = np.identity(m)
            epsilon = 1e-5
            count = 0
            x_train_cent = np.asmatrix(np.concatenate((x_train_view1_cent,x_train_view2_cent),axis=1))
            W = np.concatenate((W1,W2),axis=0)
            Phi = block_diag(Phi1,Phi2)
            
            W_old = np.zeros(W.shape)
            Phi_old = np.zeros(Phi.shape)
            while np.amax(np.absolute(W-W_old)) > epsilon or np.amax(np.absolute(Phi-Phi_old)) > epsilon:
            
                W_old = np.copy(W)
                Phi_old = np.copy(Phi)
                inv_Phi = inv(Phi)
                cov_z = inv(I_m + np.matmul(np.matmul(W.T,inv_Phi),W))
                
                # element-wise operation
                sum_x1Ez = np.zeros([d1,m])
                sum_Ezx1 = np.zeros([m,d1])
                sum_EzzT = np.zeros([m,m])
                for i in range(n_train):
                    E_z = np.matmul(np.matmul(np.matmul(cov_z,W.T),inv_Phi),x_train_cent[i,:].T)
                    E_zzT = np.matmul(E_z,E_z.T) + cov_z
                    sum_x1Ez = sum_x1Ez + np.matmul(np.asmatrix(x_train_view1_cent[i,:]).T,E_z.T)
                    sum_EzzT = sum_EzzT + E_zzT
                    sum_Ezx1 = sum_Ezx1 + np.matmul(E_z,np.asmatrix(x_train_view1_cent[i,:]))
                W1 = np.matmul(sum_x1Ez,inv(sum_EzzT))
                Phi1 = np.diag(np.diag(cov_view1 - 1/n_train*np.matmul(W1,sum_Ezx1)))
                
                W = np.concatenate((W1,W2),axis=0)
                Phi = block_diag(Phi1,Phi2)
                inv_Phi = inv(Phi)
                cov_z = inv(I_m + np.matmul(np.matmul(W.T,inv_Phi),W))
                
                # element-wise operation
                sum_x2Ez = np.zeros([d2,m])
                sum_Ezx2 = np.zeros([m,d2])
                sum_EzzT = np.zeros([m,m])
                for i in range(n_train):
                    E_z = np.matmul(np.matmul(np.matmul(cov_z,W.T),inv_Phi),x_train_cent[i,:].T)
                    E_zzT = np.matmul(E_z,E_z.T) + cov_z
                    sum_x2Ez = sum_x2Ez + np.matmul(np.asmatrix(x_train_view2_cent[i,:]).T,E_z.T)
                    sum_EzzT = sum_EzzT + E_zzT
                    sum_Ezx2 = sum_Ezx2 + np.matmul(E_z,np.asmatrix(x_train_view2_cent[i,:]))
                W2 = np.matmul(sum_x2Ez,inv(sum_EzzT))
                Phi2 = np.diag(np.diag(cov_view2 - 1/n_train*np.matmul(W2,sum_Ezx2)))
                
                W = np.concatenate((W1,W2),axis=0)
                Phi = block_diag(Phi1,Phi2)
                count = count + 1
            print('count:',count)
                
            mu = np.concatenate((mu_view1,mu_view2))
            cov_row1 = np.concatenate((np.matmul(W1,W1.T)+Phi1,np.matmul(W1,W2.T)),axis=1)
            cov_row2 = np.concatenate((np.matmul(W2,W1.T),np.matmul(W2,W2.T)+Phi2),axis=1)
            cov = np.concatenate((cov_row1,cov_row2),axis=0)
            L = loglikelyhood(x_test,mu,cov) # bigger likelihood, smaller anomaly score
            
            far_probM, dr_probM = EVALAD(label_test,-L,500)
#            print('itr:',itr)
            fars_probM.append(far_probM)
            drs_probM.append(dr_probM)
            auc = metrics.auc(far_probM,dr_probM)
            aucs_probM[itr] = auc
            
       
        print("Multiple view type-2 outlier detection result:",'ratio=',ratios[ratio_idx],'dim=',dims[dim_idx])
        print('auc(ProbM) = %.7f, %.7f' % (np.mean(aucs_probM),np.std(aucs_probM)))
        AUC_mean.append(np.mean(aucs_probM))
        AUC_std.append(np.std(aucs_probM))
    
np.savetxt('AUCs(RatioDim)_t2.csv',np.concatenate((np.asmatrix(AUC_mean).T,np.asmatrix(AUC_std).T),axis=1), delimiter=",")