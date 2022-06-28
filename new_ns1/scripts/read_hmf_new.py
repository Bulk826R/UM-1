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

im1 = plt.imread('/home/bulk826/Desktop/Stanford/Research1/figures/new_ns1/hmf.png')
fig.set_size_inches(8, 6)

ax1 = fig.add_subplot(1, 1, 1)
ax1.imshow(im1)

ax1.text(1740, 100, 'Bright', fontsize = 10, horizontalalignment = 'right', 
         verticalalignment = 'top', color = 'black', alpha = 0.72)
ax1.text(1740, 180, 'Dwarfs', fontsize = 10, horizontalalignment = 'right', 
         verticalalignment = 'top', color = 'black', alpha = 0.72)

ax1.text(1450, 100, 'Classical', fontsize = 10, horizontalalignment = 'right', 
         verticalalignment = 'top', color = 'black', alpha = 0.56)
ax1.text(1450, 180, 'Dwarfs', fontsize = 10, horizontalalignment = 'right', 
         verticalalignment = 'top', color = 'black', alpha = 0.56)

ax1.text(1120, 100, 'Ultra-faint', fontsize = 10, horizontalalignment = 'right', 
         verticalalignment = 'top', color = 'black', alpha = 0.42)
ax1.text(1120, 180, 'Dwarfs', fontsize = 10, horizontalalignment = 'right', 
         verticalalignment = 'top', color = 'black', alpha = 0.42)

ax1.axis('off')

fig.savefig('/home/bulk826/Desktop/Stanford/Research1/figures/new_ns1/hmf_text.png', dpi = 400, bbox_inches = 'tight')
plt.show