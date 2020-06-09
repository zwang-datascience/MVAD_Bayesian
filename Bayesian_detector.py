# -*- coding: utf-8 -*-

import numpy as np
from numpy import genfromtxt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# -----------------------------------------
# evaluate ROC of anomaly detection results 
# -----------------------------------------
def EVALAD(label_true,adscore,numbins):
    
    num_test = len(label_true)
    
    thres = np.linspace(0,num_test,numbins)
    thres = thres.astype(int)
    thres = np.delete(thres,-1)
    thres = np.delete(thres,-1)
    
    idx_adsort = np.argsort(adscore)
    
#    
    detectionrate = np.zeros(len(thres))
    falsealarmrate = np.zeros(len(thres)) 
    
    for i in range(len(thres)):
        
        label_pred = np.zeros(num_test)
        
        label_pred[idx_adsort[0:thres[i]]] = -1
        
        label_pred[label_pred!=-1] = 1
        
        fpr = np.sum(np.logical_and(label_pred==1,label_true==-1)) 
        fnr = np.sum(np.logical_and(label_pred==-1,label_true==1)) 
        tpr = np.sum(np.logical_and(label_pred==1,label_true==1)) 
        tnr = np.sum(np.logical_and(label_pred==-1,label_true==-1)) 
        
        falsealarmrate[i] = fpr / (fpr+tnr) 
        detectionrate[i] = tpr / (tpr+fnr) 
        
    return falsealarmrate,detectionrate

# -----------------------------------------
# Outlier score measurement 
# -----------------------------------------
def cal_outScore(mu,Lam,nu,x):
    base = 1 + 1/nu*np.diag((x-mu).T.dot(Lam).dot(x-mu))
    return -np.power(base,-1)

epochs = 10
AUCs = []
dims = [,] # for different datasets, setting their dimensionalities of 1st view and 2nd view 
fars = []
drs = []
for epoch in range(epochs):
    
    # import the dataset with generated multi-view outliers
     data = genfromtxt('') 
#    data = genfromtxt('webkb_Type123_' + str(epoch + 1) + '.csv',delimiter = ',')
    X = np.copy(data[:,0:-1])
    n_X = X.shape[0]
    label = np.copy(data[:,-1])
    label[label==0] = -1
    
    x_nor_train = X[label==-1]
    x_test = np.copy(X)
    label_test = np.copy(label)
    
    x_train_view1 = x_nor_train[:,0:dims[0]]
    x_train_view2 = x_nor_train[:,dims[0]:]
    n_train = x_train_view1.shape[0]
    
    # settings and hyperparameters
    V = 2
    d = np.array([dims[0],dims[1]])
    N = x_train_view1.shape[0]
    m = min(d)-1
    itrs = 6
    x = [x_train_view1.T,x_train_view2.T]
    
    a_nu = 2
    b_nu = .1
    nu_V = np.array([d[v]+1 for v in range(V)])
    K = [1e-3*np.identity(d[v]) for v in range(V)]
    beta = np.array([1e-3 for _ in range(V)])
    a_alpha = 1e-3
    b_alpha = 1e-3
    
    # initialize
    a_tidle_nu = a_nu + N/2  # 1
    b_tidle_nu = np.abs(np.random.randn())  # 1
    cov_z = [np.abs(np.random.randn())*np.identity(m) for n in range(N)] # N*m*m
    mean_z = np.random.randn(m,N)  #m*N
    a_tidle_u = np.abs(np.random.randn(N)) # (N,)
    b_tidle_u = np.abs(np.random.randn(N)) # (N,)
    K_tidle = [np.diag(np.abs(np.random.randn(d[v]))) for v in range(V)] # V*d*d 
    nu_tidle = np.array([nu_V[v] + N for v in range(V)]) #(V,)
    cov_mu = [np.identity(d[v]) for v in range(V)] # V*d*d
    mean_mu = [np.random.randn(d[v],1) for v in range(V)] # V*d*1
    a_tidle_alpha = np.array([a_alpha+d[v]/2 for v in range(V)]) #(V,)
    b_tidle_alpha = np.stack([np.abs(np.random.randn(m)) for _ in range(V)],axis = 0) # V*m
    cov_w = [[np.identity(m) for _ in range(d[v])] for v in range(V)] # V*d*m*m
    mean_w = [np.random.randn(d[v],m) for v in range(V)]  # V*d*m
    
    for itr in range(itrs):
        mean_Psi = [nu_tidle[v]*np.linalg.inv(K_tidle[v]) for v in range(V)]
        mean_nu = a_tidle_nu/b_tidle_nu
        
        # joint likelihood
        mu_opt = np.concatenate([mean_mu[v] for v in range(V)], axis = 0)
        Lam = np.linalg.inv(np.block([
                                [mean_w[0].dot(mean_w[0].T)+np.linalg.inv(mean_Psi[0]), mean_w[0].dot(mean_w[1].T)],
                                [mean_w[1].dot(mean_w[0].T), mean_w[1].dot(mean_w[1].T)+np.linalg.inv(mean_Psi[1])]
                                ]))
        nu_opt = mean_nu
        
        # update
        # u
        a_tidle_u = np.array([0.5*(mean_nu+m+np.sum([d[v] for v in range(V)])) for n in range(N)])
        b_tidle_u = []
        for n in range(N):
            mean_zz = mean_z[:,n].dot(mean_z[:,n])+np.trace(cov_z[n])
            c = 0
            for v in range(V):
                xn = x[v][:,n]
                m_zn = mean_z[:,n]
                m_mu = mean_mu[v].flatten()
                A = m_zn.reshape([-1,1]).dot(m_zn.reshape([-1,1]).T)+cov_z[n]
                wzzw = mean_w[v].dot(A).dot(mean_w[v].T) + np.diag([np.trace(cov_w[v][j].dot(A)) for j in range(d[v])])
                mean_zwPwz = np.trace(wzzw.dot(mean_Psi[v]))
                c += xn.dot(mean_Psi[v]).dot(xn)-2*xn.dot(mean_Psi[v].dot(mean_w[v])).dot(m_zn)-2*xn.dot(mean_Psi[v]).dot(m_mu) \
                        +2*m_zn.dot(mean_w[v].T.dot(mean_Psi[v])).dot(m_mu)+mean_zwPwz+np.trace(mean_Psi[v].dot(mean_mu[v].dot(mean_mu[v].T)+cov_mu[v]))
            b_tidle_u.append(0.5*(c+mean_zz+mean_nu))
        b_tidle_u = np.array(b_tidle_u)
        
        # nu
        mean_u = a_tidle_u/b_tidle_u
        var_u = a_tidle_u/(b_tidle_u**2)
        mean_logu = np.log(mean_u) - var_u/(2*mean_u**2)
        b_tidle_nu = b_nu - 0.5*(N+np.sum(mean_logu-mean_u))
        
        # z
        mean_WPW = [mean_w[v].T.dot(mean_Psi[v]).dot(mean_w[v])+
                             np.sum([mean_Psi[v][j,j]*cov_w[v][j] for j in range(d[v])],axis = 0) for v in range(V)]
        cov_z = [np.linalg.inv(mean_u[n]*(np.identity(m)+np.sum(mean_WPW,axis=0))) for n in range(N)]
        mean_WPx = [mean_w[v].T.dot(mean_Psi[v].dot(x[v]-mean_mu[v]))*mean_u for v in range(V)]
        mean_z = np.array([cov_z[n].dot(np.sum(mean_WPx,axis=0)[:,n]) for n in range(N)]).T
        
        # Psi
        K_tidle = []
        for v in range(V):
            xx = (x[v]*mean_u).dot(x[v].T)
            mumu = np.sum([(mean_mu[v].dot(mean_mu[v].T)+cov_mu[v])*mean_u[n] for n in range(N)],axis=0)
            xzw = np.zeros([d[v],d[v]])
            xmu = np.zeros([d[v],d[v]])
            wzx = np.zeros([d[v],d[v]])
            wzzw = np.zeros([d[v],d[v]])
            wzmu = np.zeros([d[v],d[v]])
            mux = np.zeros([d[v],d[v]])
            muzw = np.zeros([d[v],d[v]])
            for n in range(N):
                xn = x[v][:,n].reshape([-1,1])
                zn = mean_z[:,n].reshape([-1,1])
                xzw += xn.dot(zn.T).dot(mean_w[v].T)*mean_u[n]
                xmu += xn.dot(mean_mu[v].T)*mean_u[n]
                wzx += mean_w[v].dot(zn).dot(xn.T)*mean_u[n]
                A = (zn.dot(zn.T)+cov_z[n])*mean_u[n]
                wzzw += mean_w[v].dot(A).dot(mean_w[v].T) + np.diag([np.trace(cov_w[v][j].dot(A)) for j in range(d[v])])
                wzmu += mean_w[v].dot(zn).dot(mean_mu[v].T)*mean_u[n]
                mux += mean_mu[v].dot(xn.T)*mean_u[n]
                muzw += mean_mu[v].dot(zn.T).dot(mean_w[v].T)*mean_u[n]
            K_tidle.append(K[v]+ xx- xzw- xmu- wzx+ wzzw+ wzmu- mux+ muzw + mumu)
        
        # mu
        cov_mu = [np.linalg.inv(beta[v]*np.identity(d[v])+ np.sum(mean_u)*mean_Psi[v]) for v in range(V)]
        mean_mu = [cov_mu[v].dot(mean_Psi[v]).dot(np.sum((x[v]-mean_w[v].dot(mean_z))*mean_u,axis=1)).reshape([-1,1]) for v in range(V)]    
        
        # alpha
        a_tidle_alpha = [a_alpha + d[v]/2 for v in range(V)]
        b_tidle_alpha = [np.array([b_alpha + .5*(mean_w[v][:,j].dot(mean_w[v][:,j])+ np.sum(cov_w[v],axis=0)[j,j]
                                ) for j in range(m)]) for v in range(V)]
        
        # w
        mean_alpha = [a_tidle_alpha[v] / b_tidle_alpha[v] for v in range(V)]
        cov_w = [[np.linalg.inv(np.diag(mean_alpha[v])+ ((mean_z*mean_u).dot(mean_z.T)+ 
                                np.sum([cov_z[n]*mean_u[n] for n in range(N)],axis=0))*mean_Psi[v][j,j]) for j in range(d[v])] for v in range(V)]
        mean_w = [np.array([cov_w[v][j].dot(np.sum([mean_z[:,n].reshape([-1,1]).dot(mean_Psi[v][:,j].reshape([1,-1])*mean_u[n]).dot(x[v][:,n].reshape([-1,1])-
                          mean_mu[v]) for n in range(N)],axis=0)- ((mean_z*mean_u).dot(mean_z.T)+np.sum([cov_z[n]*mean_u[n] for n in range(N)],axis=0)).dot(np.sum(
                            [mean_w[v][l].reshape([-1,1])*mean_Psi[v][l,j] for l in range(d[v]) if l != j],axis=0))).squeeze() 
                            for j in range(d[v])]) for v in range(V)] 
        
        outlierScore = cal_outScore(mu_opt, Lam, nu_opt, x_test.T)
        auc = roc_auc_score(label_test,outlierScore)
        print('auc itr:', auc)
    
    far, dr = EVALAD(label_test, outlierScore, 500)
    auc = metrics.auc(far,dr)
    fars.append(far)
    drs.append(dr)
    AUCs.append(auc)
    print('epoch:', epoch, 'auc:',auc)