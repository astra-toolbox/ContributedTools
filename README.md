-----------------------------------------------------------------------
This file is part of the TVR-DART Toolbox 
Author: Dr. Xiaodong ZHUGE
Copyright: 2016, CWI, Amsterdam
http://www.cwi.nl/
License: Open Source under GPLv3
Contact: x.zhuge@cwi.nl / zhugexd@hotmail.com
-----------------------------------------------------------------------
This is a Python implementation of TVR-DART algorithm 
(Total Variation Regularized Discrete Algebraic Reconstruction Technique), 
a robust and automated reconsturction algorithm for performing discrete tomography.
Currently we support 2D and 3D parallel beam geometries, orianted for electron tomography

The basic forward and backward projection operations are GPU-accelerated by utilizing
the python interface of the ASTRA tomography toolbox (http://www.astra-toolbox.com/)

Documentation / samples:
-------------------------
See the Python code samples:
s01_recon2D.py
s02_recon3D.py

References:
------------
If you use the TVR-DART Toolbox for your research, we would appreciate it if you would refer to the following papers:
[1] X. Zhuge, W.J. Palenstijn, K.J. Batenburg, "TVR-DART: A More Robust Algorithm for Discrete Tomography From Limited Projection Data 
With Automated Gray Value Estimation," IEEE Transactions on Imaging Processing, 2016, vol. 25, issue 1, pp. 455-468.
[2] X. Zhuge, H. Jinnai, R.E. Dunin-Borkowski, V. Migunov, S. Bals, P. Cool, A.J. Bons, K.J. Batenburg, 
"Automated discrete electron tomography - Towards routine high-fidelity reconstruction of nanomaterials," Ultramicroscopy, under revision
