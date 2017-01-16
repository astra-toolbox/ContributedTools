#-----------------------------------------------------------------------
#Copyright 2016 Centrum Wiskunde & Informatica, Amsterdam
#National Research Institute for Mathematics and Computer Science in the Netherlands
#Author: Dr. Xiaodong ZHUGE
#Contact: x.zhuge@cwi.nl/zhugexd@hotmail.com
#
#
#This file is part of the Python implementation of TVR-DART algorithm (Total Variation 
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
# Python wrapper of ASTRA toolbox's SIRT (Simultaneous Iterative Reconstruction
# Technique) implementation
# Please refer to the ASTRA toolbox for reference
#  W. J. Palenstijn, K J. Batenburg, and J. Sijbers, "Performance improvements
#  for iterative electron tomography reconstruction using graphics processing
#  units (GPUs)", Journal of Structural Biology, vol. 176, issue 2, pp. 250-253,
#  2011, http://dx.doi.org/10.1016/j.jsb.2011.07.017
#------------------------------------------------------------------------------

import astra
import numpy as np

def recon(sinogram, Niter, proj_geom, vol_geom, pu = 'cuda'):
    Nx = vol_geom['GridColCount']
    Nz = vol_geom['GridRowCount']
    if pu == 'cuda':
        if len(sinogram.shape)==3:
            Ny = sinogram.shape[2]
            siz = np.array((Nx,Ny,Nz))
            rec = SIRT3D_gpu(sinogram, Niter, proj_geom, vol_geom,siz)
        else:
            rec = SIRT2D_gpu(sinogram, Niter, proj_geom, vol_geom)
    else:
        if len(sinogram.shape)==3:
            Ny = sinogram.shape[2]
            siz = np.array((Nx,Ny,Nz))
            rec = SIRT3D_cpu(sinogram, Niter, proj_geom, vol_geom,siz)
        else:
            rec = SIRT2D_cpu(sinogram, Niter, proj_geom, vol_geom)    
    return rec;

def SIRT2D_cpu(sinogram, Niter, proj_geom, vol_geom):
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)
    proj_id = astra.create_projector('strip',proj_geom,vol_geom)
    
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    # Set up the parameters for a reconstruction algorithm using the CPU
    cfg = astra.astra_dict('SIRT')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = proj_id
    cfg['option']={}
    cfg['option']['MinConstraint'] = 0

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    astra.algorithm.run(alg_id, Niter)

    # Get the result
    rec = astra.data2d.get(rec_id)

    # Clean up.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)
    
    return rec;
    
def SIRT3D_cpu(sinogram, Niter, proj_geom, vol_geom,siz):
    [Nan, Ndet, Ny] = sinogram.shape
    [Nx,Ny,Nz] = siz
    rec = np.zeros((Nx,Ny,Nz))
    for yi in range(Ny):
        recsli = SIRT2D_cpu(sinogram[:,:,yi], Niter, proj_geom, vol_geom)
        rec[:,yi,:] = recsli.T
    return rec;
    
def SIRT2D_gpu(sinogram, Niter, proj_geom, vol_geom):
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict('SIRT_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option']={} 
    cfg['option']['MinConstraint'] = 0
    
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    astra.algorithm.run(alg_id, Niter)

    # Get the result
    rec = astra.data2d.get(rec_id)

    # Clean up.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
     
    return rec;
    
def SIRT3D_gpu(sinogram, Niter, proj_geom, vol_geom,siz):
    [Nan, Ndet, Ny] = sinogram.shape
    [Nx,Ny,Nz] = siz
    rec = np.zeros((Nx,Ny,Nz))
    for yi in range(Ny):
        recsli = SIRT2D_gpu(sinogram[:,:,yi], Niter, proj_geom, vol_geom)
        rec[:,yi,:] = recsli.T
    return rec;