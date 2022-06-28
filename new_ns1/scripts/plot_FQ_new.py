from __future__ import unicode_literals
import numpy as np
import sys, os
import matplotlib.pylab as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams["grid.color"] = 'grey'
matplotlib.rcParams["grid.linestyle"] = '--'
plt.rc("text", usetex=False)
import h5py as h5
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm
import matplotlib.colors as colors
import pynbody
import pynbody.plot.sph as sph
import glob
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
fig = plt.figure()

im1 = mpimg.imread('/home/bulk826/Desktop/Stanford/Research1/figures/new_ns1/FQ_12z.png')


fig.set_size_inches(8, 16)
plt.subplots_adjust(wspace = 0., hspace = 0.)

ax1 = fig.add_subplot(1, 1, 1)
ax1.imshow(im1, aspect='auto')
ax1.axis('off')
ax1.text(1720, 2000, r'$\mathrm{SDSS\ centrals}$', fontsize = 14, alpha = 0.8, horizontalalignment = 'left', color = 'dimgray')
#ax1.text(1650, 1900, r'$\mathrm{Geha\ et\ al.\ (2012)}$', fontsize = 14, alpha = 0.8, horizontalalignment = 'left', color = 'dimgray')
ax1.text(1300, 3720, r'$\mathrm{SAGA\ satellites}$', fontsize = 14, alpha = 0.8, color = 'orangered', horizontalalignment = 'left')
#ax1.text(1300, 3620, r'$\mathrm{Mao\ et\ al.\ (2020)}$', fontsize = 14, alpha = 0.8, color = 'orangered', horizontalalignment = 'left')
ax1.text(740, 3920, r'$\mathrm{Local\ Group\ dwarfs}$', fontsize = 14, alpha = 0.8, color = 'crimson', horizontalalignment = 'left')
#ax1.text(750, 3900, r'$\mathrm{Weisz\ et\ al.\ (2014)}$', fontsize = 14, alpha = 0.8, color = 'crimson', horizontalalignment = 'left')
ax1.text(800, 4000, r'$\mathrm{(mostly\ quenched)}$', fontsize = 13, alpha = 0.8, color = 'crimson', horizontalalignment = 'left')



fig.savefig('/home/bulk826/Desktop/Stanford/Research1/figures/new_ns1/FQ.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()
