#-----------------------------------------------------------------------
#Copyright 2016 Centrum Wiskunde & Informatica, Amsterdam
#National Research Institute for Mathematics and Computer Science in the Netherlands
#Author: Dr. Xiaodong ZHUGE
#Contact: x.zhuge@cwi.nl/zhugexd@hotmail.com
#
#
#This file is the Python implementation of TVR-DART algorithm (Total Variation 
#Regularized Discrete Algebraic Reconstruction Technique), a robust and automated 
#reconsturction algorithm for performing discrete tomography
#
#References: 
# [1] X. Zhuge, W.J. Palenstijn, K.J. Batenburg, "TVR-DART: 
# A More Robust Algorithm for Discrete Tomography From Limited Projection Data 
# With Automated Gray Value Estimation," IEEE Transactions on Imaging Processing, 
# 2016, vol. 25, issue 1, pp. 455-468
# [2] X. Zhuge, H. Jinnai, R.E. Dunin-Borkowski, V. Migunov, S. Bals, P. Cool,
# A.J. Bons, K.J. Batenburg, "Automated discrete electron tomography - Towards
# routine high-fidelity reconstruction of nanomaterials," Ultramicroscopy 2016
#
#This Python implementaton of TVR-DART is a free software: you can use 
#it and/or redistribute it under the terms of the GNU General Public License as 
#published by the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This Python implementaton is distributed in the hope that it will 
#be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#the GNU General Public License can be found at
#<http://www.gnu.org/licenses/>.
#
#------------------------------------------------------------------------------
#
import astra
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import SIRT

# objective function for parameter estimation
def objective_param(x,W,p,lamb,epsilon):
    def objectivefun(param):
        [gv,K] = param2gv(param)
        S = softseg(x,K,gv)
        if len(x.shape)==2:
            WS = W * S
        else:
            WS = FP3D(S,W)

        fit = (p-WS)**2
        fit = np.sum(fit)
        tv = hubernorm(S,epsilon)
        f = fit + lamb*tv
        return f;
    
    return objectivefun;

# objective function for reconstruction
def objective_rec(param,W,p,lamb,epsilon,siz):
    def objectivefun(x):
        x = x.reshape(siz)        
        [gv,K] = param2gv(param)
        S = softseg(x,K,gv)
        if len(x.shape)==2:
            WS = W * S
        else:
            WS = FP3D(S,W)
    
        fit = (p-WS)**2
        fit = np.sum(fit)
        tv = hubernorm(S,epsilon)
        f = fit + lamb*tv
        return f;
    
    return objectivefun;

# derivative of the objective function over the parameters
def der_param(x,W,p,lamb,epsilon):
    def derobjectivefun(param):
        N = len(param)
        [gv,K] = param2gv(param)
        th = np.diff(gv)/2+gv[0:-1]
        Ns = len(K)
        
        siz = x.shape
        Npixel = np.prod(siz)
        
        S = softseg(x,K,gv)
        if len(x.shape)==2:
            WS = W * S
        else:
            WS = FP3D(S,W)
        
        WSmp = WS-p
        
        JhuberS = GHuberx(S,epsilon)
        JhuberS = JhuberS.reshape([Npixel,1])
        
        g = np.zeros(N)
        for si in range(Ns):
            taoi = th[si]
            Ki = K[si]
            Gdif = gv[si+1] - gv[si]
            kg = Ki/Gdif
            
            uo = logistic_np(x-taoi,kg)
            xmtaoi = np.abs(x-taoi)
            u = logistic_np(xmtaoi,np.abs(kg))
            u2 = u**2
            e = np.exp(-2*np.abs(kg)*xmtaoi)
            eu2 = e*u2
            
            if si == Ns-1:
                Sdp = uo - eu2*Ki*(2*(x-taoi)/Gdif+1)
            else:
                taoip = th[si+1]
                Kip = K[si+1]
                Gdifp = gv[si+2] - gv[si+1]
                kgp = Kip/Gdifp
                uop = logistic_np(x-taoip,kgp)
                xmtaoip = np.abs(x-taoip)
                up = logistic_np(xmtaoip,np.abs(kgp))
                up2 = up**2
                ep = np.exp(-2*np.abs(kgp)*xmtaoip)
                eup2 = ep*up2
                Sdp = uo - eu2*Ki*(2*(x-taoi)/Gdif+1) - uop - eup2*Ki*(1-2*(x-taoip)/Gdifp)
            if len(x.shape)==2:
                WSdp = W*Sdp
            else:
                WSdp = FP3D(Sdp,W)
            Sdp = Sdp.reshape((Npixel,1))
            
            g[int(si)] = 2*np.dot(WSmp.T,WSdp) + np.dot(JhuberS.T,Sdp)
            
            SdK = 2*(x-taoi)*eu2
            if len(x.shape)==2:
                WSdK = W * SdK
            else:
                WSdK = FP3D(SdK,W)
            SdK = SdK.reshape((Npixel,1))
            
            g[int(si+Ns)] = 2*np.dot(WSmp.T,WSdK) + np.dot(JhuberS.T,SdK)
        
        return g;
    return derobjectivefun;

# derivative of the objective function over the reconstruction
def der_rec(param,W,p,lamb,epsilon,siz):
    def derobjectivefun(x):
        x = x.reshape(siz)
        [gv,K] = param2gv(param)
    
        Npixel = np.prod(siz)
        
        S = softseg(x,K,gv)
        if len(x.shape)==2:
            WS = W * S
        else:
            WS = FP3D(S,W)
        
        Sx = dersoftseg(x,K,gv)
        
        WSmp = WS - p;
        if len(siz)==2:
            Jfit_S = 2*(W.H*WSmp)
        else:
            Jfit_S = 2*BP3D(WSmp,W,siz)
        Jfit_S = Jfit_S.reshape(siz)
        
        JhuberS = GHuberx(S,epsilon)
        g = (Jfit_S + JhuberS)*Sx;
        g = g.reshape([Npixel])
        return g;
    return derobjectivefun;

# Complete TVR-DART workflow for creating a 3D discrete tomographic reconstruction
def DT(W,p, vol_size, proj_size, angles, Ngv, lamb=10, PU = 'cuda', K = 4, Niter = 50, epsilon=1e-4):
    
    [Nx,Nz] = vol_size
    [Ndetx, Ndety] = proj_size
    Nan = len(angles)
    if Ndety==1:
        sinogram = p.reshape([Nan, Ndetx])
    else:
        sinogram = p.reshape([Nan, Ndetx, Ndety])
    
    # create projection geometry and operator
    print('TVR-DART ...')
    print('Create projection geometry and operator...')
    proj_geom = astra.create_proj_geom('parallel', 1.0, Ndetx, angles)
    vol_geom = astra.create_vol_geom(Nz,Nx)
    proj_id = astra.create_projector(PU,proj_geom,vol_geom)
    W = astra.OpTomo(proj_id)
    
    # initial reconstruction
    print('Initial reconstruction...')
    recsirt = SIRT.recon(sinogram, 200, proj_geom, vol_geom, PU)
    sf = np.max(recsirt)
    p = p/sf
    if Ndety==1:
        sinogram = p.reshape([Nan, Ndetx])
    else:
        sinogram = p.reshape([Nan, Ndetx, Ndety])
    recsirt = recsirt/sf
    
    # set initial TVR-DART parameters
    K = K*np.ones(Ngv-1)
    gv = np.linspace(0, 1, Ngv,True)
    param0 = gv2param(gv,K)
    
    # Esimating parameter using a small section of the dataset
    print('Estimation of optimal parameters...')
    if Ndety==1:
        Segrec,param_esti = joint(W,p, recsirt, param0 ,lamb)
    else:
        Elist = np.sum(recsirt,axis=(0,2))
        index = np.argmax(Elist)
    
        index1 = np.max(np.hstack((0,index-1)))
        index2 = np.min(np.hstack((index+1,Ndety-1)))
    
        x0_esti = recsirt[:,index1:index2+1,:]
        sinogram_esti = sinogram[:,:,index1:index2+1]
        p_esti = sinogram_esti.reshape(Nan*Ndetx*(index2-index1+1))
    
        Segrec,param_esti = joint(W,p_esti, x0_esti, param0 ,lamb)
    
    # Reconstruction with estimated parameter
    print('Reconstruction with estimated parameters...')
    Segrec,rec = recon(W,p, recsirt, param_esti, lamb, Niter, epsilon)
    [gv,K] = param2gv(param_esti)
    param_esti = gv2param(gv*sf,K)
    Segrec = Segrec*sf;
    return Segrec, param_esti;

# Joint estimation of reconstruction and parameters
def joint(W,p,x0, param0 ,lamb=1, Niter = 20):
    rec = x0
    param_esti = param0
    
    param_esti = estiparam(W,p, param_esti, rec, lamb)
    print(str(param2gv(param_esti)))
    for iter in range(Niter):
        gvpre,Kpre = param2gv(param_esti)
        Segrec,rec = recon(W,p, rec, param_esti, lamb)
        param_esti = estiparam(W,p, param_esti, rec, lamb)
        gv,K = param2gv(param_esti)
        if np.sum(np.abs(gv-gvpre))/np.sum(np.abs(gvpre))<1e-3:
            print('parameter estimation converged')
            break
        print(str(param2gv(param_esti)))
    return Segrec,param_esti;
    
#  Reconstruction
def recon(W,p,x0 ,param,lamb=1, Niter = 2, epsilon=1e-4):
    siz = x0.shape
    obj = objective_rec(param,W,p,lamb,epsilon,siz)
    objder = der_rec(param,W,p,lamb,epsilon,siz)
    res = minimize(obj, x0, method='L-BFGS-B', jac=objder, options={'maxiter': Niter,'disp': False})

    rec = res.x
    rec = rec.reshape(siz)
    [gv,K] = param2gv(param)
    Segrec = softseg(rec,K,gv)
    return Segrec,rec;

# Parameter estimation    
def estiparam(W,p, param0, x, lamb=1, Niter = 5, epsilon=1e-4):
    obj = objective_param(x, W,p,lamb,epsilon)
    objder = der_param(x,W,p,lamb,epsilon)
    res = minimize(obj, param0, method='L-BFGS-B', jac=objder, options={'maxiter': Niter,'disp': False})
    
    param = res.x
    return param;

def param2gv(param):
    N = len(param)
    gvplus = param[0:int(N/2)]
    gv = np.hstack((0,gvplus))  
    K = param[int(N/2):]
    return gv,K;
    
def gv2param(gv,K):
    param = np.hstack((gv[1:],K))
    return param;
    
# derivative of Huber norm over the segmented solution
def GHuberx(S,epsilon):
    if len(S.shape)==2:
        [Nz,Nx] = S.shape
        dxplus = np.zeros((Nz,Nx))
        dyplus = np.zeros((Nz,Nx))
        dxplus[:,0:Nx-1] = S[:,1:] - S[:,0:Nx-1]
        dyplus[0:Nz-1,:] = S[1:,:] - S[0:Nz-1,:]
        absg = np.sqrt(np.power(dxplus,2) + np.power(dyplus,2))
    
        M1 = absg<=epsilon
        M1r = ~M1
        M1 = M1.astype(np.int)
        M1r = M1r.astype(np.int)
        M2 = absg<=epsilon
        M2[1:,:] = M1[0:Nz-1,:]
        M2r = ~M2
        M2 = M2.astype(np.int)
        M2r = M2r.astype(np.int)
        M3 = absg<=epsilon
        M3[:,1:] = M1[:,0:Nx-1]
        M3r = ~M3
        M3 = M3.astype(np.int)
        M3r = M3r.astype(np.int)
    
        dxminus = np.zeros((Nz,Nx))
        dyminus = np.zeros((Nz,Nx))
        dxminus[:,1:] = S[:,1:] - S[:,0:Nx-1]
        dyminus[1:,:] = S[1:,:] - S[0:Nz-1,:]
    
        absg_uvMinus1 = np.zeros((Nz,Nx))
        absg_uvMinus1[1:,:] = absg[0:Nz-1,:]
        absg_uMinus1v = np.zeros((Nz,Nx))
        absg_uMinus1v[:,1:] = absg[:,0:Nx-1]
    
        sumplus = dxplus + dyplus
        Jhuber_term1a = -(1/epsilon)*sumplus*M1
        absg[absg==0] = 1e-9
        Jhuber_term1b = -(sumplus/absg)*M1r
    
        Jhuber_term2a = (1/epsilon)*dyminus*M2
        absg_uvMinus1[absg_uvMinus1==0] = 1e-9
        Jhuber_term2b = (dyminus/absg_uvMinus1)*M2r
    
        Jhuber_term3a = (1/epsilon)*dxminus*M3
        absg_uMinus1v[absg_uMinus1v==0] = 1e-9
        Jhuber_term3b = (dxminus/absg_uMinus1v)*M3r
    
        JhuberS = Jhuber_term1a + Jhuber_term1b + Jhuber_term2a + Jhuber_term2b + Jhuber_term3a + Jhuber_term3b
    else:
        [Nx,Ny,Nz] = S.shape
        dxplus = np.zeros((Nx,Ny,Nz))
        dyplus = np.zeros((Nx,Ny,Nz))
        dzplus = np.zeros((Nx,Ny,Nz))
        dxplus[0:Nx-1,:,:] = S[1:,:,:] - S[0:Nx-1,:,:]
        dyplus[:,0:Ny-1,:] = S[:,1:,:] - S[:,0:Ny-1,:]
        dzplus[:,:,0:Nz-1] = S[:,:,1:] - S[:,:,0:Nz-1]
        absg = np.sqrt(np.power(dxplus,2) + np.power(dyplus,2) + np.power(dzplus,2))
    
        M1 = absg<=epsilon
        M1r = ~M1
        M1 = M1.astype(np.int)
        M1r = M1r.astype(np.int)
    
        M2 = absg<=epsilon
        M2[1:,:,:] = M1[0:Nx-1,:,:]
        M2r = ~M2
        M2 = M2.astype(np.int)
        M2r = M2r.astype(np.int)
    
        M3 = absg<=epsilon
        M3[:,1:,:] = M1[:,0:Ny-1,:]
        M3r = ~M3
        M3 = M3.astype(np.int)
        M3r = M3r.astype(np.int)
    
        M4 = absg<=epsilon
        M4[:,:,1:] = M1[:,:,0:Nz-1]
        M4r = ~M4
        M4 = M4.astype(np.int)
        M4r = M4r.astype(np.int)
    
        dxminus = np.zeros((Nx,Ny,Nz))
        dyminus = np.zeros((Nx,Ny,Nz))
        dzminus = np.zeros((Nx,Ny,Nz))
        dxminus[1:,:,:] = S[1:,:,:] - S[0:Nx-1,:,:]
        dyminus[:,1:,:] = S[:,1:,:] - S[:,0:Ny-1,:]
        dzminus[:,:,1:] = S[:,:,1:] - S[:,:,0:Nz-1]
        
        absg_uvMinus1W = np.zeros((Nx,Ny,Nz))
        absg_uvMinus1W[:,1:,:] = absg[:,0:Ny-1,:]
        absg_uMinus1vW = np.zeros((Nx,Ny,Nz))
        absg_uMinus1vW[1:,:,:] = absg[0:Nx-1,:,:]
        absg_uvwMinus1 = np.zeros((Nx,Ny,Nz))
        absg_uvwMinus1[:,:,1:] = absg[:,:,0:Nz-1]
        
        sumplus = dxplus + dyplus + dzplus
        Jhuber_term1a = -(1/epsilon)*sumplus*M1
        absg[absg==0] = 1e-9
        Jhuber_term1b = -(sumplus/absg)*M1r
    
        Jhuber_term2a = (1/epsilon)*dxminus*M2
        absg_uMinus1vW[absg_uMinus1vW==0] = 1e-9
        Jhuber_term2b = (dxminus/absg_uMinus1vW)*M2r
        
        Jhuber_term3a = (1/epsilon)*dyminus*M3
        absg_uvMinus1W[absg_uvMinus1W==0] = 1e-9
        Jhuber_term3b = (dyminus/absg_uvMinus1W)*M3r
        
        Jhuber_term4a = (1/epsilon)*dzminus*M4
        absg_uvwMinus1[absg_uvwMinus1==0] = 1e-9
        Jhuber_term4b = (dzminus/absg_uvwMinus1)*M4r
        
        JhuberS = Jhuber_term1a + Jhuber_term1b + Jhuber_term2a + Jhuber_term2b + Jhuber_term3a + Jhuber_term3b + Jhuber_term4a + Jhuber_term4b
    
    return JhuberS;
    
# Huber Norm
def hubernorm(x,epsilon):
    if len(x.shape)==2:
        [Nz,Nx] = x.shape
        dxplus = np.zeros((Nz,Nx))
        dyplus = np.zeros((Nz,Nx))
        dxplus[:,0:Nx-1] = x[:,1:] - x[:,0:Nx-1]
        dyplus[0:Nz-1,:] = x[1:,:] - x[0:Nz-1,:]
        absg = np.sqrt(np.power(dxplus,2) + np.power(dyplus,2))
        n = np.zeros((Nz,Nx))
    else:
        [Nx,Ny,Nz] = x.shape
        dxplus = np.zeros((Nx,Ny,Nz))
        dyplus = np.zeros((Nx,Ny,Nz))
        dzplus = np.zeros((Nx,Ny,Nz))
        dxplus[0:Nx-1,:,:] = x[1:,:,:] - x[0:Nx-1,:,:]
        dyplus[:,0:Ny-1,:] = x[:,1:,:] - x[:,0:Ny-1,:]
        dzplus[:,:,0:Nz-1] = x[:,:,1:] - x[:,:,0:Nz-1]
        absg = np.sqrt(np.power(dxplus,2) + np.power(dyplus,2) + np.power(dzplus,2))
        n = np.zeros((Nx,Ny,Nz))
        
    L2mask = absg<=epsilon
    L1mask = ~L2mask
    n[L2mask] = np.power(absg[L2mask],2)/(2*epsilon)
    n[L1mask] = absg[L1mask] - epsilon/2
    hn = np.sum(n)
    return hn;

# Forward projection 3D parallel beam geometry
def FP3D(x,W):    
    [Nx,Ny,Nz] = x.shape
    [Np,Npixel] = W.shape
    FP = np.zeros((Np,Ny))
    for yi in range(Ny):
        sli = x[:,yi,:].T
        Wx = W * sli
        FP[:,yi] = Wx
    
    FP = FP.reshape(Np*Ny)
    return FP;
    
# Backprojection 3D parallel beam geometry
def BP3D(p,W,siz):
    [Nx,Ny,Nz] = siz
    [Np,Npixel] = W.shape
    p = p.reshape((Np,Ny))
    BP = np.zeros((Nx,Ny,Nz))
    for yi in range(Ny):
        psli = p[:,yi]
        bp = W.H * psli
        bp = bp.reshape([Nz,Nx])
        BP[:,yi,:] = bp.T
    BP = BP.reshape(Npixel*Ny)
    return BP;

# Soft Segmentation function
def softseg(x,K,gv):
    Ngv = len(gv)-1
    th = np.diff(gv)/2+gv[0:-1]
    Segrec = gv[0]*np.ones(x.shape)
    
    for ii in range(Ngv):
        gi = ii+1
        ti = ii
        Gdif = gv[gi] - gv[gi-1]
        kg = K[ii]/Gdif
        u = logistic_np(x-th[ti],kg)
        Segrec += Gdif*u
    return Segrec;
    
# Derivative of Soft Segmentation function over reconstruction
def dersoftseg(x,K,gv):
    Ngv = len(gv)-1
    th = np.diff(gv)/2+gv[0:-1]
    Sdx = np.zeros(x.shape)
    
    for ii in range(Ngv):
        gi = ii+1
        ti = ii
        Gdif = gv[gi] - gv[gi-1]
        kg = K[ii]/Gdif
        taoi = th[ti]
        
        xmtaoi = np.abs(x-taoi)
        u = logistic_np(xmtaoi,kg)
        e = np.exp(-2*np.abs(kg)*xmtaoi)
        u2 = u**2
        eu2 = e*u2
        
        Sdx += 2*kg*Gdif*eu2;
    return Sdx;
    
# logistic function
def logistic_np(x,k):
    return sigmoid_scipy(2*k*x);

# sigmoid on numpy array
def sigmoid_np(x):                                        
    return 1 / (1 + np.exp(-x));
    
def sigmoid_scipy(x):                                        
    return expit(x);
    