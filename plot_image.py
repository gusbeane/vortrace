'''Hacky script for displaying projection data'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import colorcet as cc

UnitL = 1e3 #pc
UnitM = 1e10 #Msun
UnitSigma = UnitM / (UnitL**2)

fname = "test_proj.dat"
#fname = "test_slice.dat"
npix = 256
data = np.loadtxt(fname)
data = np.reshape(data,(npix,npix))
data = data.T
data = data[::-1,:]
#For proj
data *= UnitSigma
#For slice
#data *= UnitM

fig = plt.figure(figsize=(6,4))

ax1 = fig.add_subplot(111)
im1 = plt.imshow(np.log10(data), extent=[-3,3,-3,3], origin = 'upper', cmap=cc.m_CET_L16, interpolation='bicubic')
#Proj limits
plt.clim(-0.6, 1.5)

ax1.set_xlabel('kpc')
ax1.set_xlabel('kpc')

plt.colorbar(im1,label='$\Sigma\,[\mathrm{M_\odot\,pc^{-2}}]$')
#plt.colorbar(im1,label=r'$\rho\,[\mathrm{M_\odot\,kpc^{-3}}]$')
fig.tight_layout()
#savename = './proj.pdf'
#plt.savefig(savename,dpi=400)
#plt.close()
plt.show()