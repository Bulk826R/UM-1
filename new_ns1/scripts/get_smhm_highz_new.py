from __future__ import unicode_literals
import numpy as np
import os
import itertools
from scipy.stats import norm
from helpers.SimulationAnalysis import readHlist, SimulationAnalysis, iterTrees

h = 0.7
omega_m = 0.286
src = '/sdf/group/kipac/u/ycwang/MWmass_new' 
wr = '/sdf/group/kipac/g/cosmo/ki21/ycwang/join_halos/MWmass'

####################################

D = np.loadtxt(wr+'/Data_join_MWmass_z0.txt')
Mstar = D[:, 23]
Vpeak = D[:, 14]
ID_all = np.where(Mstar > 0.)[0]
data_zoom = D[ID_all]
ID = data_zoom[:, 0].astype(np.int64)
lg_mhalo = np.log10(data_zoom[:, 13])
lg_mstar = np.log10(data_zoom[:, 23])
print('Mpeak range z0 =', np.min(lg_mhalo), np.max(lg_mhalo))

nbins = 14
bins_x = np.linspace(5.95, 12.775, nbins)
print(bins_x)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(bins_x) - bins_h, np.max(bins_x) + bins_h, nbins+1)
bin_tot = np.digitize(lg_mhalo, bins)
bin_means_tot = np.asarray([lg_mhalo[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([lg_mstar[bin_tot == i].mean() for i in range(1, len(bins))])
bin_std = np.asarray([lg_mstar[bin_tot == i].std() for i in range(1, len(bins))])
bin_med = np.asarray([np.percentile(lg_mstar[bin_tot == i], 50) for i in range(1, len(bins))])
bin_low = np.asarray([np.percentile(lg_mstar[bin_tot == i], 16) for i in range(1, len(bins))])
bin_high = np.asarray([np.percentile(lg_mstar[bin_tot == i], 84) for i in range(1, len(bins))])

data_bin_x_zoom = bins_x
data_bin_y_zoom = bin_means
data_bin_y_std_zoom = bin_std
data_bin_y_med_zoom = bin_med
data_bin_y_low_zoom = bin_low
data_bin_y_high_zoom = bin_high

np.savetxt(wr+'/smhm/z0/full_x.txt', data_bin_x_zoom)
np.savetxt(wr+'/smhm/z0/full_y.txt', data_bin_y_zoom)
np.savetxt(wr+'/smhm/z0/full_y_std.txt', data_bin_y_std_zoom)
np.savetxt(wr+'/smhm/z0/full_y_med.txt', data_bin_y_med_zoom)
np.savetxt(wr+'/smhm/z0/full_y_low.txt', data_bin_y_low_zoom)
np.savetxt(wr+'/smhm/z0/full_y_high.txt', data_bin_y_high_zoom)

####################################

D = np.loadtxt(wr+'/Data_join_MWmass_z0p2.txt')
Mstar = D[:, 23]
Vpeak = D[:, 14]
ID_all = np.where(Mstar > 0.)[0]
data_zoom = D[ID_all]
ID = data_zoom[:, 0].astype(np.int64)
lg_mhalo = np.log10(data_zoom[:, 13])
lg_mstar = np.log10(data_zoom[:, 23])
print('Mpeak range z0p2 =', np.min(lg_mhalo), np.max(lg_mhalo))

nbins = 14
bins_x = np.linspace(5.95, 12.775, nbins)
print(bins_x)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(bins_x) - bins_h, np.max(bins_x) + bins_h, nbins+1)
bin_tot = np.digitize(lg_mhalo, bins)
bin_means_tot = np.asarray([lg_mhalo[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([lg_mstar[bin_tot == i].mean() for i in range(1, len(bins))])
bin_std = np.asarray([lg_mstar[bin_tot == i].std() for i in range(1, len(bins))])
bin_med = np.asarray([np.percentile(lg_mstar[bin_tot == i], 50) for i in range(1, len(bins))])
bin_low = np.asarray([np.percentile(lg_mstar[bin_tot == i], 16) for i in range(1, len(bins))])
bin_high = np.asarray([np.percentile(lg_mstar[bin_tot == i], 84) for i in range(1, len(bins))])

data_bin_x_zoom = bins_x
data_bin_y_zoom = bin_means
data_bin_y_std_zoom = bin_std
data_bin_y_med_zoom = bin_med
data_bin_y_low_zoom = bin_low
data_bin_y_high_zoom = bin_high

np.savetxt(wr+'/smhm/z0p2/full_x.txt', data_bin_x_zoom)
np.savetxt(wr+'/smhm/z0p2/full_y.txt', data_bin_y_zoom)
np.savetxt(wr+'/smhm/z0p2/full_y_std.txt', data_bin_y_std_zoom)
np.savetxt(wr+'/smhm/z0p2/full_y_med.txt', data_bin_y_med_zoom)
np.savetxt(wr+'/smhm/z0p2/full_y_low.txt', data_bin_y_low_zoom)
np.savetxt(wr+'/smhm/z0p2/full_y_high.txt', data_bin_y_high_zoom)

####################################

D = np.loadtxt(wr+'/Data_join_MWmass_z0p5.txt')
Mstar = D[:, 23]
Vpeak = D[:, 14]
ID_all = np.where(Mstar > 0.)[0]
data_zoom = D[ID_all]
ID = data_zoom[:, 0].astype(np.int64)
lg_mhalo = np.log10(data_zoom[:, 13])
lg_mstar = np.log10(data_zoom[:, 23])
print('Mpeak range z0p5 =', np.min(lg_mhalo), np.max(lg_mhalo))

nbins = 13
bins_x = np.linspace(5.95, 12.25, nbins)
print(bins_x)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(bins_x) - bins_h, np.max(bins_x) + bins_h, nbins+1)
bin_tot = np.digitize(lg_mhalo, bins)
bin_means_tot = np.asarray([lg_mhalo[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([lg_mstar[bin_tot == i].mean() for i in range(1, len(bins))])
bin_std = np.asarray([lg_mstar[bin_tot == i].std() for i in range(1, len(bins))])
bin_med = np.asarray([np.percentile(lg_mstar[bin_tot == i], 50) for i in range(1, len(bins))])
bin_low = np.asarray([np.percentile(lg_mstar[bin_tot == i], 16) for i in range(1, len(bins))])
bin_high = np.asarray([np.percentile(lg_mstar[bin_tot == i], 84) for i in range(1, len(bins))])

data_bin_x_zoom = bins_x
data_bin_y_zoom = bin_means
data_bin_y_std_zoom = bin_std
data_bin_y_med_zoom = bin_med
data_bin_y_low_zoom = bin_low
data_bin_y_high_zoom = bin_high

np.savetxt(wr+'/smhm/z0p5/full_x.txt', data_bin_x_zoom)
np.savetxt(wr+'/smhm/z0p5/full_y.txt', data_bin_y_zoom)
np.savetxt(wr+'/smhm/z0p5/full_y_std.txt', data_bin_y_std_zoom)
np.savetxt(wr+'/smhm/z0p5/full_y_med.txt', data_bin_y_med_zoom)
np.savetxt(wr+'/smhm/z0p5/full_y_low.txt', data_bin_y_low_zoom)
np.savetxt(wr+'/smhm/z0p5/full_y_high.txt', data_bin_y_high_zoom)

####################################

D = np.loadtxt(wr+'/Data_join_MWmass_z1.txt')
Mstar = D[:, 23]
Vpeak = D[:, 14]
ID_all = np.where(Mstar > 0.)[0]
data_zoom = D[ID_all]
ID = data_zoom[:, 0].astype(np.int64)
lg_mhalo = np.log10(data_zoom[:, 13])
lg_mstar = np.log10(data_zoom[:, 23])
print('Mpeak range z1 =', np.min(lg_mhalo), np.max(lg_mhalo))

nbins = 13
bins_x = np.linspace(5.95, 12.25, nbins)
print(bins_x)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(bins_x) - bins_h, np.max(bins_x) + bins_h, nbins+1)
bin_tot = np.digitize(lg_mhalo, bins)
bin_means_tot = np.asarray([lg_mhalo[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([lg_mstar[bin_tot == i].mean() for i in range(1, len(bins))])
bin_std = np.asarray([lg_mstar[bin_tot == i].std() for i in range(1, len(bins))])
bin_med = np.asarray([np.percentile(lg_mstar[bin_tot == i], 50) for i in range(1, len(bins))])
bin_low = np.asarray([np.percentile(lg_mstar[bin_tot == i], 16) for i in range(1, len(bins))])
bin_high = np.asarray([np.percentile(lg_mstar[bin_tot == i], 84) for i in range(1, len(bins))])

data_bin_x_zoom = bins_x
data_bin_y_zoom = bin_means
data_bin_y_std_zoom = bin_std
data_bin_y_med_zoom = bin_med
data_bin_y_low_zoom = bin_low
data_bin_y_high_zoom = bin_high

np.savetxt(wr+'/smhm/z1/full_x.txt', data_bin_x_zoom)
np.savetxt(wr+'/smhm/z1/full_y.txt', data_bin_y_zoom)
np.savetxt(wr+'/smhm/z1/full_y_std.txt', data_bin_y_std_zoom)
np.savetxt(wr+'/smhm/z1/full_y_med.txt', data_bin_y_med_zoom)
np.savetxt(wr+'/smhm/z1/full_y_low.txt', data_bin_y_low_zoom)
np.savetxt(wr+'/smhm/z1/full_y_high.txt', data_bin_y_high_zoom)

####################################

D = np.loadtxt(wr+'/Data_join_MWmass_z2.txt')
Mstar = D[:, 23]
Vpeak = D[:, 14]
ID_all = np.where(Mstar > 0.)[0]
data_zoom = D[ID_all]
ID = data_zoom[:, 0].astype(np.int64)
lg_mhalo = np.log10(data_zoom[:, 13])
lg_mstar = np.log10(data_zoom[:, 23])
print('Mpeak range z2 =', np.min(lg_mhalo), np.max(lg_mhalo))

nbins = 13
bins_x = np.linspace(5.95, 12.25, nbins)
print(bins_x)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(bins_x) - bins_h, np.max(bins_x) + bins_h, nbins+1)
bin_tot = np.digitize(lg_mhalo, bins)
bin_means_tot = np.asarray([lg_mhalo[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([lg_mstar[bin_tot == i].mean() for i in range(1, len(bins))])
bin_std = np.asarray([lg_mstar[bin_tot == i].std() for i in range(1, len(bins))])
bin_med = np.asarray([np.percentile(lg_mstar[bin_tot == i], 50) for i in range(1, len(bins))])
bin_low = np.asarray([np.percentile(lg_mstar[bin_tot == i], 16) for i in range(1, len(bins))])
bin_high = np.asarray([np.percentile(lg_mstar[bin_tot == i], 84) for i in range(1, len(bins))])

data_bin_x_zoom = bins_x
data_bin_y_zoom = bin_means
data_bin_y_std_zoom = bin_std
data_bin_y_med_zoom = bin_med
data_bin_y_low_zoom = bin_low
data_bin_y_high_zoom = bin_high

np.savetxt(wr+'/smhm/z2/full_x.txt', data_bin_x_zoom)
np.savetxt(wr+'/smhm/z2/full_y.txt', data_bin_y_zoom)
np.savetxt(wr+'/smhm/z2/full_y_std.txt', data_bin_y_std_zoom)
np.savetxt(wr+'/smhm/z2/full_y_med.txt', data_bin_y_med_zoom)
np.savetxt(wr+'/smhm/z2/full_y_low.txt', data_bin_y_low_zoom)
np.savetxt(wr+'/smhm/z2/full_y_high.txt', data_bin_y_high_zoom)

####################################

D = np.loadtxt(wr+'/Data_join_MWmass_z4.txt')
Mstar = D[:, 23]
Vpeak = D[:, 14]
ID_all = np.where(Mstar > 0.)[0]
data_zoom = D[ID_all]
ID = data_zoom[:, 0].astype(np.int64)
lg_mhalo = np.log10(data_zoom[:, 13])
lg_mstar = np.log10(data_zoom[:, 23])
print('Mpeak range z4 =', np.min(lg_mhalo), np.max(lg_mhalo))

nbins = 11
bins_x = np.linspace(5.95, 11.2, nbins)
print(bins_x)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(bins_x) - bins_h, np.max(bins_x) + bins_h, nbins+1)
bin_tot = np.digitize(lg_mhalo, bins)
bin_means_tot = np.asarray([lg_mhalo[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([lg_mstar[bin_tot == i].mean() for i in range(1, len(bins))])
bin_std = np.asarray([lg_mstar[bin_tot == i].std() for i in range(1, len(bins))])
bin_med = np.asarray([np.percentile(lg_mstar[bin_tot == i], 50) for i in range(1, len(bins))])
bin_low = np.asarray([np.percentile(lg_mstar[bin_tot == i], 16) for i in range(1, len(bins))])
bin_high = np.asarray([np.percentile(lg_mstar[bin_tot == i], 84) for i in range(1, len(bins))])

data_bin_x_zoom = bins_x
data_bin_y_zoom = bin_means
data_bin_y_std_zoom = bin_std
data_bin_y_med_zoom = bin_med
data_bin_y_low_zoom = bin_low
data_bin_y_high_zoom = bin_high

np.savetxt(wr+'/smhm/z4/full_x.txt', data_bin_x_zoom)
np.savetxt(wr+'/smhm/z4/full_y.txt', data_bin_y_zoom)
np.savetxt(wr+'/smhm/z4/full_y_std.txt', data_bin_y_std_zoom)
np.savetxt(wr+'/smhm/z4/full_y_med.txt', data_bin_y_med_zoom)
np.savetxt(wr+'/smhm/z4/full_y_low.txt', data_bin_y_low_zoom)
np.savetxt(wr+'/smhm/z4/full_y_high.txt', data_bin_y_high_zoom)

####################################

D = np.loadtxt(wr+'/Data_join_MWmass_z6.txt')
Mstar = D[:, 23]
Vpeak = D[:, 14]
ID_all = np.where(Mstar > 0.)[0]
data_zoom = D[ID_all]
ID = data_zoom[:, 0].astype(np.int64)
lg_mhalo = np.log10(data_zoom[:, 13])
lg_mstar = np.log10(data_zoom[:, 23])
print('Mpeak range z6 =', np.min(lg_mhalo), np.max(lg_mhalo))

nbins = 11
bins_x = np.linspace(5.95, 11., nbins)
print(bins_x)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(bins_x) - bins_h, np.max(bins_x) + bins_h, nbins+1)
bin_tot = np.digitize(lg_mhalo, bins)
bin_means_tot = np.asarray([lg_mhalo[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([lg_mstar[bin_tot == i].mean() for i in range(1, len(bins))])
bin_std = np.asarray([lg_mstar[bin_tot == i].std() for i in range(1, len(bins))])
bin_med = np.asarray([np.percentile(lg_mstar[bin_tot == i], 50) for i in range(1, len(bins))])
bin_low = np.asarray([np.percentile(lg_mstar[bin_tot == i], 16) for i in range(1, len(bins))])
bin_high = np.asarray([np.percentile(lg_mstar[bin_tot == i], 84) for i in range(1, len(bins))])

data_bin_x_zoom = bins_x
data_bin_y_zoom = bin_means
data_bin_y_std_zoom = bin_std
data_bin_y_med_zoom = bin_med
data_bin_y_low_zoom = bin_low
data_bin_y_high_zoom = bin_high

np.savetxt(wr+'/smhm/z6/full_x.txt', data_bin_x_zoom)
np.savetxt(wr+'/smhm/z6/full_y.txt', data_bin_y_zoom)
np.savetxt(wr+'/smhm/z6/full_y_std.txt', data_bin_y_std_zoom)
np.savetxt(wr+'/smhm/z6/full_y_med.txt', data_bin_y_med_zoom)
np.savetxt(wr+'/smhm/z6/full_y_low.txt', data_bin_y_low_zoom)
np.savetxt(wr+'/smhm/z6/full_y_high.txt', data_bin_y_high_zoom)

####################################

D = np.loadtxt(wr+'/Data_join_MWmass_z8.txt')
Mstar = D[:, 23]
Vpeak = D[:, 14]
ID_all = np.where(Mstar > 0.)[0]
data_zoom = D[ID_all]
ID = data_zoom[:, 0].astype(np.int64)
lg_mhalo = np.log10(data_zoom[:, 13])
lg_mstar = np.log10(data_zoom[:, 23])
print('Mpeak range z8 =', np.min(lg_mhalo), np.max(lg_mhalo))

nbins = 10
bins_x = np.linspace(5.95, 10.675, nbins)
print(bins_x)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(bins_x) - bins_h, np.max(bins_x) + bins_h, nbins+1)
bin_tot = np.digitize(lg_mhalo, bins)
bin_means_tot = np.asarray([lg_mhalo[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([lg_mstar[bin_tot == i].mean() for i in range(1, len(bins))])
bin_std = np.asarray([lg_mstar[bin_tot == i].std() for i in range(1, len(bins))])
bin_med = np.asarray([np.percentile(lg_mstar[bin_tot == i], 50) for i in range(1, len(bins))])
bin_low = np.asarray([np.percentile(lg_mstar[bin_tot == i], 16) for i in range(1, len(bins))])
bin_high = np.asarray([np.percentile(lg_mstar[bin_tot == i], 84) for i in range(1, len(bins))])

data_bin_x_zoom = bins_x
data_bin_y_zoom = bin_means
data_bin_y_std_zoom = bin_std
data_bin_y_med_zoom = bin_med
data_bin_y_low_zoom = bin_low
data_bin_y_high_zoom = bin_high

np.savetxt(wr+'/smhm/z8/full_x.txt', data_bin_x_zoom)
np.savetxt(wr+'/smhm/z8/full_y.txt', data_bin_y_zoom)
np.savetxt(wr+'/smhm/z8/full_y_std.txt', data_bin_y_std_zoom)
np.savetxt(wr+'/smhm/z8/full_y_med.txt', data_bin_y_med_zoom)
np.savetxt(wr+'/smhm/z8/full_y_low.txt', data_bin_y_low_zoom)
np.savetxt(wr+'/smhm/z8/full_y_high.txt', data_bin_y_high_zoom)

####################################

D = np.loadtxt(wr+'/Data_join_MWmass_z10.txt')
Mstar = D[:, 23]
Vpeak = D[:, 14]
ID_all = np.where(Mstar > 0.)[0]
data_zoom = D[ID_all]
ID = data_zoom[:, 0].astype(np.int64)
lg_mhalo = np.log10(data_zoom[:, 13])
lg_mstar = np.log10(data_zoom[:, 23])
print('Mpeak range z10 =', np.min(lg_mhalo), np.max(lg_mhalo))

nbins = 11
bins_x = np.linspace(5.95, np.max(lg_mhalo), nbins)
print(bins_x)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(bins_x) - bins_h, np.max(bins_x) + bins_h, nbins+1)
bin_tot = np.digitize(lg_mhalo, bins)
bin_means_tot = np.asarray([lg_mhalo[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([lg_mstar[bin_tot == i].mean() for i in range(1, len(bins))])
bin_std = np.asarray([lg_mstar[bin_tot == i].std() for i in range(1, len(bins))])
bin_med = np.asarray([np.percentile(lg_mstar[bin_tot == i], 50) for i in range(1, len(bins))])
bin_low = np.asarray([np.percentile(lg_mstar[bin_tot == i], 16) for i in range(1, len(bins))])
bin_high = np.asarray([np.percentile(lg_mstar[bin_tot == i], 84) for i in range(1, len(bins))])

data_bin_x_zoom = bins_x
data_bin_y_zoom = bin_means
data_bin_y_std_zoom = bin_std
data_bin_y_med_zoom = bin_med
data_bin_y_low_zoom = bin_low
data_bin_y_high_zoom = bin_high

np.savetxt(wr+'/smhm/z10/full_x.txt', data_bin_x_zoom)
np.savetxt(wr+'/smhm/z10/full_y.txt', data_bin_y_zoom)
np.savetxt(wr+'/smhm/z10/full_y_std.txt', data_bin_y_std_zoom)
np.savetxt(wr+'/smhm/z10/full_y_med.txt', data_bin_y_med_zoom)
np.savetxt(wr+'/smhm/z10/full_y_low.txt', data_bin_y_low_zoom)
np.savetxt(wr+'/smhm/z10/full_y_high.txt', data_bin_y_high_zoom)