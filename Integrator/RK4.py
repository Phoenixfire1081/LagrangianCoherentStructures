from __future__ import division
from scipy.interpolate import griddata
from numba import jit

def rk4_int(xk, yk, tk, psi, h):
	
	delta = 0.0001
	
	k1x = h*(-(psi(xk, yk+delta, tk) - psi(xk, yk-delta, tk))/(2*delta))
	k1y = h*((psi(xk+delta, yk, tk) - psi(xk-delta, yk, tk))/(2*delta))
	
	xk1 = xk+(k1x/2)
	yk1 = yk+(k1y/2)
	tk1 = tk+(h/2)
	
	k2x = h*(-(psi(xk1, yk1+delta, tk1) - psi(xk1, yk1-delta, tk1))/(2*delta))
	k2y = h*((psi(xk1+delta, yk1, tk1) - psi(xk1-delta, yk1, tk1))/(2*delta))
	
	xk2 = xk+(k2x/2)
	yk2 = yk+(k2y/2)
	tk2 = tk+(h/2)
	
	k3x = h*(-(psi(xk2, yk2+delta, tk2) - psi(xk2, yk2-delta, tk2))/(2*delta))
	k3y = h*((psi(xk2+delta, yk2, tk2) - psi(xk2-delta, yk2, tk2))/(2*delta))
	
	xk3 = xk+k3x
	yk3 = yk+k3y
	tk3 = tk+h
	
	k4x = h*(-(psi(xk3, yk3+delta, tk3) - psi(xk3, yk3-delta, tk3))/(2*delta))
	k4y = h*((psi(xk3+delta, yk3, tk3) - psi(xk3-delta, yk3, tk3))/(2*delta))
	
	xp1 = xk + ((1/6)*(k1x+2*k2x+2*k3x+k4x))
	yp1 = yk + ((1/6)*(k1y+2*k2y+2*k3y+k4y))
	
	return xp1, yp1	

def rk4_gyre(xk, yk, tk, psi, h, e):
	
	delta = 0.0001
	
	k1x = h*(-(psi(xk, yk+delta, tk, e) - psi(xk, yk-delta, tk, e))/(2*delta))
	k1y = h*((psi(xk+delta, yk, tk, e) - psi(xk-delta, yk, tk, e))/(2*delta))
	
	xk1 = xk+(k1x/2)
	yk1 = yk+(k1y/2)
	tk1 = tk+(h/2)
	
	k2x = h*(-(psi(xk1, yk1+delta, tk1, e) - psi(xk1, yk1-delta, tk1, e))/(2*delta))
	k2y = h*((psi(xk1+delta, yk1, tk1, e) - psi(xk1-delta, yk1, tk1, e))/(2*delta))
	
	xk2 = xk+(k2x/2)
	yk2 = yk+(k2y/2)
	tk2 = tk+(h/2)
	
	k3x = h*(-(psi(xk2, yk2+delta, tk2, e) - psi(xk2, yk2-delta, tk2, e))/(2*delta))
	k3y = h*((psi(xk2+delta, yk2, tk2, e) - psi(xk2-delta, yk2, tk2, e))/(2*delta))
	
	xk3 = xk+k3x
	yk3 = yk+k3y
	tk3 = tk+h
	
	k4x = h*(-(psi(xk3, yk3+delta, tk3, e) - psi(xk3, yk3-delta, tk3, e))/(2*delta))
	k4y = h*((psi(xk3+delta, yk3, tk3, e) - psi(xk3-delta, yk3, tk3, e))/(2*delta))
	
	xp1 = xk + ((1/6)*(k1x+2*k2x+2*k3x+k4x))
	yp1 = yk + ((1/6)*(k1y+2*k2y+2*k3y+k4y))
	
	return xp1, yp1	

def rk4_3(xk, yk, zk, tk, abc, h):
	
	# if integration_type == 'Forward':
		
		k1x, k1y, k1z = abc(xk, yk, zk, tk)
		
		xk1 = xk+(k1x/2)
		yk1 = yk+(k1y/2)
		zk1 = zk+(k1z/2)
		tk1 = tk+(h/2)
		
		k2x, k2y, k2z = abc(xk1, yk1, zk1, tk1)
		
		xk2 = xk+(k2x/2)
		yk2 = yk+(k2y/2)
		zk2 = zk+(k2z/2)
		tk2 = tk+(h/2)
		
		k3x, k3y, k3z = abc(xk2, yk2, zk2, tk2)
		
		xk3 = xk+k3x
		yk3 = yk+k3y
		zk3 = zk+k3z
		tk3 = tk+h
		
		k4x, k4y, k4z = abc(xk3, yk3, zk3, tk3)
		
		xp1 = xk + ((1/6)*(k1x+2*k2x+2*k3x+k4x))
		yp1 = yk + ((1/6)*(k1y+2*k2y+2*k3y+k4y))
		zp1 = zk + ((1/6)*(k1z+2*k2z+2*k3z+k4z))
	
		return xp1, yp1, zp1

@jit(nopython = True)
def interp3_fast(xloc, yloc, zloc, val, x, y, z):
		
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
		
def rk4_data(xk, yk, zk, tk, abc, h, x, y, z, _uReshaped, _vReshaped, _wReshaped, sign):
	
	def uvw(tn, xk, yk, zk):
			
		u = interp3_fast(xk, yk, zk, _uReshaped, x, y, z)
		v = interp3_fast(xk, yk, zk, _vReshaped, x, y, z)
		w = interp3_fast(xk, yk, zk, _wReshaped, x, y, z)
		
		return u, v, w
		
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
