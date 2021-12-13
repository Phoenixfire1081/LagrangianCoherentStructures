from __future__ import division
from numba import cuda
from numba import *
from math import *

@cuda.jit
def gpu_advected(_appendAdvectedVelocityx, _appendAdvectedVelocityy, _appendAdvectedVelocityz, x, y, z, XX, YY, ZZ, t, h, _u, _v, _w, sign):
	
	def interp3_fast(xloc, yloc, zloc, val):
		
		# Trilinear interpolation
		# Find closest neighbors
		
		x0 = 0
		y0 = 0
		z0 = 0
		x1 = 0
		y1 = 0
		z1 = 0
		
		if xloc > x[-1]: 
		
			xloc = xloc - x[-1]
		
		if yloc > y[-1]: 
		
			yloc = yloc - y[-1]
		
		if zloc > z[-1]: 
		
			zloc = zloc - z[-1]
		
		if xloc < x[0]: 
		
			xloc = xloc + x[-1]
		
		if yloc < y[0]: 
		
			yloc = yloc + y[-1]
		
		if zloc < z[0]: 
		
			zloc = zloc + z[-1]
		
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
		
	def rk4(tk, xk, yk, zk):
		
		k1x, k1y, k1z = uvw(tk, xk, yk, zk)
		
		xk1 = xk + sign*(k1x/2)
		yk1 = yk + sign*(k1y/2)
		zk1 = zk + sign*(k1z/2)
		tk1 = tk + sign*(h/2)
		
		k2x, k2y, k2z = uvw(tk, xk, yk, zk)
		
		xk2 = xk + sign*(k2x/2)
		yk2 = yk + sign*(k2y/2)
		zk2 = zk + sign*(k2z/2)
		tk2 = tk + sign*(h/2)
		
		k3x, k3y, k3z = uvw(tk, xk, yk, zk)
		
		xk3 = xk + sign*k3x
		yk3 = yk + sign*k3y
		zk3 = zk + sign*k3z
		tk3 = tk + sign*h
		
		k4x, k4y, k4z = uvw(tk, xk, yk, zk)
		
		xp1 = xk + sign*((h/6)*(k1x+2*k2x+2*k3x+k4x))
		yp1 = yk + sign*((h/6)*(k1y+2*k2y+2*k3y+k4y))
		zp1 = zk + sign*((h/6)*(k1z+2*k2z+2*k3z+k4z))
		
		return xp1, yp1, zp1
	
	startX = cuda.grid(1)
	gridX = cuda.gridDim.x * cuda.blockDim.x;
	
	for i in range(startX, len(XX), gridX):
		
		x_new, y_new, z_new = rk4(t, XX[i], YY[i], ZZ[i])
			
		_appendAdvectedVelocityx[i] = x_new
		_appendAdvectedVelocityy[i] = y_new
		_appendAdvectedVelocityz[i] = z_new
