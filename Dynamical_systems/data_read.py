from __future__ import division
import numpy as np
from interpolationFastOBO import interp3_CPU
# from scipy.interpolate import RegularGridInterpolator
# from scipy.interpolate import griddata
# from scipy.interpolate import interp1d
# import h5py as h5
# import time

def uvw(x, y, z, tn, XX, YY, ZZ,  _uReshaped, _vReshaped, _wReshaped):
	
	# if tn % 1 == 0:
		
	tn = int(tn)
	
	u = interp3_CPU(x, y, z, _uReshaped, XX, YY, ZZ)
	v = interp3_CPU(x, y, z, _vReshaped, XX, YY, ZZ)
	w = interp3_CPU(x, y, z, _wReshaped, XX, YY, ZZ)
	
	# ufun = RegularGridInterpolator([XX, YY, ZZ], np.reshape(_uvwData[:,0,tn], [_resolution, _resolution, _resolution]), method = 'linear', bounds_error = False, fill_value = 0)
	# vfun = RegularGridInterpolator([XX, YY, ZZ], np.reshape(_uvwData[:,1,tn], [_resolution, _resolution, _resolution]), method = 'linear', bounds_error = False, fill_value = 0)
	# wfun = RegularGridInterpolator([XX, YY, ZZ], np.reshape(_uvwData[:,2,tn], [_resolution, _resolution, _resolution]), method = 'linear', bounds_error = False, fill_value = 0)
	
	# u = ufun([x, y, z])
	# v = vfun([x, y, z])
	# w = wfun([x, y, z])
	
	# u = griddata(np.c_[XX, YY, ZZ], _uvwData[:,0,tn], np.c_[x, y, z], method = 'linear', fill_value = 0)
	# v = griddata(np.c_[XX, YY, ZZ], _uvwData[:,1,tn], np.c_[x, y, z], method = 'linear', fill_value = 0)
	# w = griddata(np.c_[XX, YY, ZZ], _uvwData[:,2,tn], np.c_[x, y, z], method = 'linear', fill_value = 0)
	# print(u[0])
	# raise SystemError
		
	# else:
		
		# tn1 = int(np.floor(tn))
		# tn2 = int(np.ceil(tn))
		
		# _resolution = 10

		# ufun1 = RegularGridInterpolator([XX, YY, ZZ], np.reshape(_uvwData[:,0,tn1], [_resolution, _resolution, _resolution]), method = 'linear', bounds_error = False, fill_value = 0)
		# vfun1 = RegularGridInterpolator([XX, YY, ZZ], np.reshape(_uvwData[:,1,tn1], [_resolution, _resolution, _resolution]), method = 'linear', bounds_error = False, fill_value = 0)
		# wfun1 = RegularGridInterpolator([XX, YY, ZZ], np.reshape(_uvwData[:,2,tn1], [_resolution, _resolution, _resolution]), method = 'linear', bounds_error = False, fill_value = 0)
		
		# ufun2 = RegularGridInterpolator([XX, YY, ZZ], np.reshape(_uvwData[:,0,tn2], [_resolution, _resolution, _resolution]), method = 'linear', bounds_error = False, fill_value = 0)
		# vfun2 = RegularGridInterpolator([XX, YY, ZZ], np.reshape(_uvwData[:,1,tn2], [_resolution, _resolution, _resolution]), method = 'linear', bounds_error = False, fill_value = 0)
		# wfun2 = RegularGridInterpolator([XX, YY, ZZ], np.reshape(_uvwData[:,2,tn2], [_resolution, _resolution, _resolution]), method = 'linear', bounds_error = False, fill_value = 0)
		
		# u1 = ufun1([x, y, z])
		# v1 = vfun1([x, y, z])
		# w1 = wfun1([x, y, z])
		
		# u2 = ufun2([x, y, z])
		# v2 = vfun2([x, y, z])
		# w2 = wfun2([x, y, z])
		
		# # print tn, tn1, tn2
		
		# # print u1, u2
		
		# utfun = interp1d([tn1, tn2], [u1[0], u2[0]], kind = 'linear', bounds_error = False, fill_value = 0)
		# vtfun = interp1d([tn1, tn2], [v1[0], v2[0]], kind = 'linear', bounds_error = False, fill_value = 0)
		# wtfun = interp1d([tn1, tn2], [w1[0], w2[0]], kind = 'linear', bounds_error = False, fill_value = 0)
		
		# u = [utfun(tn)]
		# v = [vtfun(tn)]
		# w = [wtfun(tn)]
		
		# print u
		
		# if u[0] > 0:
		
			# raise SystemError
		
	return u, v, w
	# return u[0], v[0], w[0]
