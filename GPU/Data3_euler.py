from __future__ import division
from numba import cuda
from numba import *
from math import *

@cuda.jit
def gpu_advected(_appendAdvectedVelocityx, _appendAdvectedVelocityy, _appendAdvectedVelocityz, x, y, z, XX, YY, ZZ, t, h, _u, _v, _w):
	
	def interp3_fast(xloc, yloc, zloc, val):
		
		# Trilinear interpolation
		# Find closest neighbors
		
		x0 = 0
		y0 = 0
		z0 = 0
		x1 = 0
		y1 = 0
		z1 = 0
		
		if xloc > x[-1] or yloc > y[-1] or zloc > z[-1]:
		
			c = 0
			
		else:
		
			for i in range(len(x)):
		
				if x[i] >= xloc:
					
					if i == 0:
						
						x1 += 1
					
					break
					
				else:
					
					if i > 0:
						
						x0 += 1
					
					x1 += 1
						
			for i in range(len(y)):
				
				if y[i] >= yloc:
					
					if i == 0:
						
						y1 += 1
					
					break
					
				else:
					
					if i > 0:
						
						y0 += 1
					
					y1 += 1
						
			for i in range(len(z)):
				
				if z[i] >= zloc:
					
					if i == 0:
						
						z1 += 1
					
					break
					
				else:
					
					if i > 0:
						
						z0 += 1
					
					z1 += 1
					
			# Compute linear interpolation
			
			xd = (xloc - x[x0]) / (x[x1] - x[x0])
			yd = (yloc - y[y0]) / (y[y1] - y[y0])
			zd = (zloc - z[z0]) / (z[z1] - z[z0])
			
			# Compute bilinear interpolation in x
			
			c00 = (val[x0, y0, z0] * (1 - xd)) + (val[x1, y0, z0] * xd)
			c01 = (val[x0, y0, z1] * (1 - xd)) + (val[x1, y0, z1] * xd)
			c10 = (val[x0, y1, z0] * (1 - xd)) + (val[x1, y1, z0] * xd)
			c11 = (val[x0, y1, z1] * (1 - xd)) + (val[x1, y1, z1] * xd)
			
			# Compute bilinear interpolation in y
			
			c0 = (c00 * (1 - yd)) + (c10 * yd)
			c1 = (c01 * (1 - yd)) + (c11 * yd)
			
			# Compute linear interpolation in z
			
			c = (c0 * (1 - zd)) + (c1 * zd)
		
		return c
		
	def uvw(tn, xk, yk, zk):
			
		u = interp3_fast(xk, yk, zk, _u)
		v = interp3_fast(xk, yk, zk, _v)
		w = interp3_fast(xk, yk, zk, _w)
		
		return u, v, w
		
	def euler(tk, xk, yk, zk):
		
		_xx, _yy, _zz = uvw(tk, xk, yk, zk)
			
		xp1 = xk + (h * _xx)
		yp1 = yk + (h * _yy)
		zp1 = zk + (h * _zz)
		
		return xp1, yp1, zp1
	
	startX = cuda.grid(1)
	gridX = cuda.gridDim.x * cuda.blockDim.x;
	
	for i in range(startX, len(XX), gridX):
		
		x_new, y_new, z_new = euler(t, XX[i], YY[i], ZZ[i])
			
		_appendAdvectedVelocityx[i] = x_new
		_appendAdvectedVelocityy[i] = y_new
		_appendAdvectedVelocityz[i] = z_new
