
import numpy as np

from scipy.linalg import eigh, inv

from sklearn.metrics.pairwise import pairwise_kernels,paired_cosine_distances


# -------------------------------------------------
# linear co-regularized amomaly detection technique
# -------------------------------------------------
def COREGAD(L1,L2,Yl,U1,U2,Z1,Z2,T1,T2,lam1,lam2,lam3):

    A1 = np.dot(np.transpose(L1),L1) + lam1*np.identity(L1.shape[1]) + lam2*np.dot(np.transpose(U1),U1) - lam3*np.dot(np.transpose(Z1),Z1)
    A2 = np.dot(np.transpose(L2),L2) + lam1*np.identity(L2.shape[1]) + lam2*np.dot(np.transpose(U2),U2) - lam3*np.dot(np.transpose(Z2),Z2)
    B1 = np.transpose(L1).dot(Yl)
    B2 = np.transpose(L2).dot(Yl)
    C1 = lam2*np.dot(np.transpose(U2),U1) - lam3*np.dot(np.transpose(Z2),Z1) ####
    C2 = lam2*np.dot(np.transpose(U1),U2) - lam3*np.dot(np.transpose(Z1),Z2) ####
#    print('A1:',A1.shape)
#    print('A2:',A2.shape)
#    print('B1:',B1.shape)
#    print('B2:',B2.shape)
#    print('U1:',U1.shape)
#    print('U2:',U2.shape)
#    print('Z1:',Z1.shape)
#    print('Z2:',Z2.shape)
#    print('C1:',C1.shape)
#    print('C2:',C2.shape)
    
    beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
    beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
#    print('beta1:',beta1)
    
    label_pred1 = T1.dot(beta1) # test view 1
    label_pred2 = T2.dot(beta2) # test view 2
    
    adscore = abs(label_pred1 - label_pred2)            
    
    return adscore 

def IRLS_ME_Hu_Update_W(L,beta,Y):
    resids = np.abs(Y-L.dot(beta)).flatten()
    sigma = np.median(resids)/0.6745
    k = 1.345*sigma
    n = len(resids)
    w = np.zeros(shape=(n,n))
#    print('resids:',resids)
    
    for i in range(n):
        if resids[i] > k:
            w[i,i] = k/resids[i]
        else:
            w[i,i] = 1.
    return w

def IRLS_ME_Bi_Update_W(L,beta,Y):
    resids = np.abs(Y-L.dot(beta)).flatten()
    sigma = np.median(resids)/0.6745
    k = 4.685*sigma
    n = len(resids)
    w = np.zeros(shape=(n,n))
    
    for i in range(n):
        if resids[i] > k:
            w[i, i] = 0
        else:
            w[i, i] = (1.-(resids[i]/k)**2)**2
    return w
    
def IRLS_Hu(U1,beta1,U2,beta2):
    resids = np.abs(U1.dot(beta1)-U2.dot(beta2)).flatten()
    sigma = np.median(resids)/0.6745
    k = 1.345*sigma
    n = len(resids)
    w = np.zeros(shape=(n,n))
#    print('resids:',resids)
    
    for i in range(n):
        if resids[i] > k:
            w[i,i] = k/resids[i]
        else:
            w[i,i] = 1.
    return w

def IRLS_Bi(U1,beta1,U2,beta2):
    resids = np.abs(U1.dot(beta1)-U2.dot(beta2)).flatten()
    sigma = np.median(resids)/0.6745
    k = 4.685*sigma
    n = len(resids)
    w = np.zeros(shape=(n,n))
#    print('resids:',resids)
    
    for i in range(n):
        if resids[i] > k:
            w[i,i] = k/resids[i]
        else:
            w[i,i] = 1.
    return w

# -------------------------------------------------
# linear co-regularized amomaly detection technique(weighted)
# -------------------------------------------------
def COREGADW_Hu(L1,L2,Yl,U1,U2,Z1,Z2,T1,T2,lam1,lam2,lam3,steps=20):
    
    for step in range(steps):
        if step == 0:
            A1 = np.dot(np.transpose(L1),L1) + lam1*np.identity(L1.shape[1]) + lam2*np.dot(np.transpose(U1),U1) - lam3*np.dot(np.transpose(Z1),Z1)
            A2 = np.dot(np.transpose(L2),L2) + lam1*np.identity(L2.shape[1]) + lam2*np.dot(np.transpose(U2),U2) - lam3*np.dot(np.transpose(Z2),Z2)
            B1 = np.transpose(L1).dot(Yl)
            B2 = np.transpose(L2).dot(Yl)
            C1 = lam2*np.dot(np.transpose(U2),U1) - lam3*np.dot(np.transpose(Z2),Z1)
            C2 = lam2*np.dot(np.transpose(U1),U2) - lam3*np.dot(np.transpose(Z1),Z2)
            
            beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
            beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
        else:
            W = IRLS_Hu(U1,beta1,U2,beta2)
            
            A1 = np.dot(np.transpose(L1),L1) + lam1*np.identity(L1.shape[1]) + lam2*np.dot(np.transpose(U1).dot(W),U1) - lam3*np.dot(np.transpose(Z1),Z1)
            A2 = np.dot(np.transpose(L2),L2) + lam1*np.identity(L2.shape[1]) + lam2*np.dot(np.transpose(U2).dot(W),U2) - lam3*np.dot(np.transpose(Z2),Z2)
            B1 = np.transpose(L1).dot(Yl)
            B2 = np.transpose(L2).dot(Yl)
            C1 = lam2*np.dot(np.transpose(U2).dot(W),U1) - lam3*np.dot(np.transpose(Z2),Z1)
            C2 = lam2*np.dot(np.transpose(U1).dot(W),U2) - lam3*np.dot(np.transpose(Z1),Z2)
            
            beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
            beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
    
    label_pred1 = T1.dot(beta1) 
    label_pred2 = T2.dot(beta2) 
    
    adscore = abs(label_pred1 - label_pred2)
    return adscore

# -------------------------------------------------
# linear co-regularized amomaly detection technique(weighted)
# -------------------------------------------------
def COREGADW_Bi(L1,L2,Yl,U1,U2,Z1,Z2,T1,T2,lam1,lam2,lam3,steps=20):
    
    for step in range(steps):
        if step == 0:
            A1 = np.dot(np.transpose(L1),L1) + lam1*np.identity(L1.shape[1]) + lam2*np.dot(np.transpose(U1),U1) - lam3*np.dot(np.transpose(Z1),Z1)
            A2 = np.dot(np.transpose(L2),L2) + lam1*np.identity(L2.shape[1]) + lam2*np.dot(np.transpose(U2),U2) - lam3*np.dot(np.transpose(Z2),Z2)
            B1 = np.transpose(L1).dot(Yl)
            B2 = np.transpose(L2).dot(Yl)
            C1 = lam2*np.dot(np.transpose(U2),U1) - lam3*np.dot(np.transpose(Z2),Z1)
            C2 = lam2*np.dot(np.transpose(U1),U2) - lam3*np.dot(np.transpose(Z1),Z2)
            
            beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
            beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
        else:
            W = IRLS_Bi(U1,beta1,U2,beta2)
            
            A1 = np.dot(np.transpose(L1),L1) + lam1*np.identity(L1.shape[1]) + lam2*np.dot(np.transpose(U1).dot(W),U1) - lam3*np.dot(np.transpose(Z1),Z1)
            A2 = np.dot(np.transpose(L2),L2) + lam1*np.identity(L2.shape[1]) + lam2*np.dot(np.transpose(U2).dot(W),U2) - lam3*np.dot(np.transpose(Z2),Z2)
            B1 = np.transpose(L1).dot(Yl)
            B2 = np.transpose(L2).dot(Yl)
            C1 = lam2*np.dot(np.transpose(U2).dot(W),U1) - lam3*np.dot(np.transpose(Z2),Z1)
            C2 = lam2*np.dot(np.transpose(U1).dot(W),U2) - lam3*np.dot(np.transpose(Z1),Z2)
            
            beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
            beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
    
    label_pred1 = T1.dot(beta1) 
    label_pred2 = T2.dot(beta2) 
    
    adscore = abs(label_pred1 - label_pred2)
    return adscore

# -------------------------------------------------
# linear co-regularized amomaly detection technique(weighted)
# -------------------------------------------------
def COREGADW_Bi(L1,L2,Yl,U1,U2,Z1,Z2,T1,T2,lam1,lam2,lam3,steps=20):
    
    for step in range(steps):
        if step == 0:
            A1 = np.dot(np.transpose(L1),L1) + lam1*np.identity(L1.shape[1]) + lam2*np.dot(np.transpose(U1),U1) - lam3*np.dot(np.transpose(Z1),Z1)
            A2 = np.dot(np.transpose(L2),L2) + lam1*np.identity(L2.shape[1]) + lam2*np.dot(np.transpose(U2),U2) - lam3*np.dot(np.transpose(Z2),Z2)
            B1 = np.transpose(L1).dot(Yl)
            B2 = np.transpose(L2).dot(Yl)
            C1 = lam2*np.dot(np.transpose(U2),U1) - lam3*np.dot(np.transpose(Z2),Z1) ####
            C2 = lam2*np.dot(np.transpose(U1),U2) - lam3*np.dot(np.transpose(Z1),Z2) ####
            
            beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
            beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
        else:   
            w1 = IRLS_ME_Bi_Update_W(L1,beta1,Yl)
            w2 = IRLS_ME_Bi_Update_W(L2,beta2,Yl)
            
            A1 = np.dot(np.transpose(L1).dot(w1),L1) + lam1*np.identity(L1.shape[1]) + lam2*np.dot(np.transpose(U1),U1) - lam3*np.dot(np.transpose(Z1),Z1)
            A2 = np.dot(np.transpose(L2).dot(w2),L2) + lam1*np.identity(L2.shape[1]) + lam2*np.dot(np.transpose(U2),U2) - lam3*np.dot(np.transpose(Z2),Z2)
            B1 = np.transpose(L1).dot(w1).dot(Yl)
            B2 = np.transpose(L2).dot(w2).dot(Yl)
            
            beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
            beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
            
    
    label_pred1 = T1.dot(beta1) # test view 1
    label_pred2 = T2.dot(beta2) # test view 2
    
    adscore = abs(label_pred1 - label_pred2)            
    
    return adscore  


# -------------------------------------------------
# linear co-regularized amomaly detection technique(weighted)
# -------------------------------------------------
def COREGAD_Weighted_Hu(L1,L2,Yl,U1,U2,Z1,Z2,T1,T2,lam1,lam2,lam3,steps=20):
    
    for step in range(steps):
        if step == 0:
            A1 = np.dot(np.transpose(L1),L1) + lam1*np.identity(L1.shape[1]) + lam2*np.dot(np.transpose(U1),U1) - lam3*np.dot(np.transpose(Z1),Z1)
            A2 = np.dot(np.transpose(L2),L2) + lam1*np.identity(L2.shape[1]) + lam2*np.dot(np.transpose(U2),U2) - lam3*np.dot(np.transpose(Z2),Z2)
            B1 = np.transpose(L1).dot(Yl)
            B2 = np.transpose(L2).dot(Yl)
            C1 = lam2*np.dot(np.transpose(U2),U1) - lam3*np.dot(np.transpose(Z2),Z1) ####
            C2 = lam2*np.dot(np.transpose(U1),U2) - lam3*np.dot(np.transpose(Z1),Z2) ####
            
            beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
#            print(beta1)
            beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
        else:
#            if step == 1:
#                print(beta1)
            w1 = IRLS_ME_Hu_Update_W(L1,beta1,Yl)
            w2 = IRLS_ME_Hu_Update_W(L2,beta2,Yl)
            
#            w1 = IRLS_ME_Bi_Update_W(L1,beta1,Yl)
#            w2 = IRLS_ME_Bi_Update_W(L2,beta2,Yl)
            
            A1 = np.dot(np.transpose(L1).dot(w1),L1) + lam1*np.identity(L1.shape[1]) + lam2*np.dot(np.transpose(U1),U1) - lam3*np.dot(np.transpose(Z1),Z1)
            A2 = np.dot(np.transpose(L2).dot(w2),L2) + lam1*np.identity(L2.shape[1]) + lam2*np.dot(np.transpose(U2),U2) - lam3*np.dot(np.transpose(Z2),Z2)
            B1 = np.transpose(L1).dot(w1).dot(Yl)
            B2 = np.transpose(L2).dot(w2).dot(Yl)
            
            beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
            beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
            
    
    label_pred1 = T1.dot(beta1) # test view 1
    label_pred2 = T2.dot(beta2) # test view 2
    
    adscore = abs(label_pred1 - label_pred2)            
    
    return adscore

# -------------------------------------------------
# linear co-regularized amomaly detection technique(weighted)
# -------------------------------------------------
def COREGAD_Weighted_Bi(L1,L2,Yl,U1,U2,Z1,Z2,T1,T2,lam1,lam2,lam3,steps=20):
    
    for step in range(steps):
        if step == 0:
            A1 = np.dot(np.transpose(L1),L1) + lam1*np.identity(L1.shape[1]) + lam2*np.dot(np.transpose(U1),U1) - lam3*np.dot(np.transpose(Z1),Z1)
            A2 = np.dot(np.transpose(L2),L2) + lam1*np.identity(L2.shape[1]) + lam2*np.dot(np.transpose(U2),U2) - lam3*np.dot(np.transpose(Z2),Z2)
            B1 = np.transpose(L1).dot(Yl)
            B2 = np.transpose(L2).dot(Yl)
            C1 = lam2*np.dot(np.transpose(U2),U1) - lam3*np.dot(np.transpose(Z2),Z1) ####
            C2 = lam2*np.dot(np.transpose(U1),U2) - lam3*np.dot(np.transpose(Z1),Z2) ####
            
            beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
            beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
        else:   
            w1 = IRLS_ME_Bi_Update_W(L1,beta1,Yl)
            w2 = IRLS_ME_Bi_Update_W(L2,beta2,Yl)
            
            A1 = np.dot(np.transpose(L1).dot(w1),L1) + lam1*np.identity(L1.shape[1]) + lam2*np.dot(np.transpose(U1),U1) - lam3*np.dot(np.transpose(Z1),Z1)
            A2 = np.dot(np.transpose(L2).dot(w2),L2) + lam1*np.identity(L2.shape[1]) + lam2*np.dot(np.transpose(U2),U2) - lam3*np.dot(np.transpose(Z2),Z2)
            B1 = np.transpose(L1).dot(w1).dot(Yl)
            B2 = np.transpose(L2).dot(w2).dot(Yl)
            
            beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
            beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
            
    
    label_pred1 = T1.dot(beta1) # test view 1
    label_pred2 = T2.dot(beta2) # test view 2
    
    adscore = abs(label_pred1 - label_pred2)            
    
    return adscore  

# -------------------------------------------------
# kernel co-regularized amomaly detection technique
# -------------------------------------------------
def COREGAD_Kernel(L1,L2,Yl,U1,U2,Z1,Z2,T1,T2,lam1,lam2,lam3,metric_kernel,gam1,gam2):
    
    s1 = np.r_[L1,U1,Z1] # view 1 (n+m1+m2)
    s2 = np.r_[L2,U2,Z2] # view 2 (n+m1+m2)
    
#    K1x = pairwise_kernels(L1,s1,metric=metric_kernel,gamma = gam1)
#    K2x = pairwise_kernels(L2,s2,metric=metric_kernel,gamma = gam2)
#    K1u = pairwise_kernels(U1,s1,metric=metric_kernel,gamma = gam1)
#    K2u = pairwise_kernels(U2,s2,metric=metric_kernel,gamma = gam2)
#    K1z = pairwise_kernels(Z1,s1,metric=metric_kernel,gamma = gam1)
#    K2z = pairwise_kernels(Z2,s2,metric=metric_kernel,gamma = gam2)
#    G1 = pairwise_kernels(s1,metric=metric_kernel,gamma = gam1)
#    G2 = pairwise_kernels(s2,metric=metric_kernel,gamma = gam2)
#    K1t = pairwise_kernels(T1,s1,metric=metric_kernel,gamma = gam1)
#    K2t = pairwise_kernels(T2,s2,metric=metric_kernel,gamma = gam2)
    K1x = pairwise_kernels(L1,s1,metric=metric_kernel,degree=2)
    K2x = pairwise_kernels(L2,s2,metric=metric_kernel,degree=2)
    K1u = pairwise_kernels(U1,s1,metric=metric_kernel,degree=2)
    K2u = pairwise_kernels(U2,s2,metric=metric_kernel,degree=2)
    K1z = pairwise_kernels(Z1,s1,metric=metric_kernel,degree=2)
    K2z = pairwise_kernels(Z2,s2,metric=metric_kernel,degree=2)
    G1 = pairwise_kernels(s1,metric=metric_kernel,degree=2)
    G2 = pairwise_kernels(s2,metric=metric_kernel,degree=2)
    K1t = pairwise_kernels(T1,s1,metric=metric_kernel,degree=2)
    K2t = pairwise_kernels(T2,s2,metric=metric_kernel,degree=2)
    
    A1 = np.dot(np.transpose(K1x),K1x) + lam1*G1 + 1e1*np.identity(G1.shape[0]) + lam2*np.dot(np.transpose(K1u),K1u) - lam3*np.dot(np.transpose(K1u),K1u)
    A2 = np.dot(np.transpose(K2x),K2x) + lam1*G2 + 1e1*np.identity(G2.shape[0]) + lam2*np.dot(np.transpose(K2u),K2u) - lam3*np.dot(np.transpose(K2u),K2u)
    B1 = np.transpose(K1x).dot(Yl)
    B2 = np.transpose(K2x).dot(Yl)
    C1 = lam2*np.dot(np.transpose(K2u),K1u) - lam3*np.dot(np.transpose(K2z),K1z) ####
    C2 = lam2*np.dot(np.transpose(K1u),K2u) - lam3*np.dot(np.transpose(K1z),K2z) ####
    
    beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
    beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
#    print(beta1.shape)
#    
#    beta1_sum = 0
#    for i in range(s1.shape[0]):
#        beta1_sum += beta1[i]*G1[i,:]
#    beta2_sum = 0
#    for i in range(s2.shape[0]):
#        beta2_sum += beta2[i]*G2[i,:]
#    print(beta1_sum.shape)
#    label_pred1 = K1t.dot(beta1_sum)
#    label_pred2 = K2t.dot(beta2_sum)
#    
    label_pred1 = K1t.dot(beta1)
    label_pred2 = K2t.dot(beta2)
    
    adscore = abs(label_pred1 - label_pred2)            
            
    return adscore 
            

# -------------------------------------------------
# HOAD [Gao, ICDM 2011]
# -------------------------------------------------
def HOAD(SimV1,SimV2,k,keval,m):
    
    n = SimV1.shape[0]

    Z = np.block([
                [SimV1, m*np.identity(n)],
                [m*np.identity(n), SimV2]
                ])
    
    D = np.diag(np.sum(Z,axis=1))
    
    L = D - Z 
    
    w,vr = eigh(L,eigvals=(0,k-1))
    
    adscore = np.zeros((n,len(keval)))
    
    for i in range(len(keval)):
    
        Hv1 = vr[0:n,0:keval[i]]
        Hv2 = vr[n:,0:keval[i]]
    
        adscore[:,i] = 1 - paired_cosine_distances(Hv1,Hv2)
    
    return adscore 


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