from __future__ import division
from numba import cuda
from numba import *
from math import *

@cuda.jit
def gpu_advected(_appendAdvectedVelocityx, _appendAdvectedVelocityy, _appendAdvectedVelocityz, XX, YY, ZZ, t, h):
	
	def abc(tn, x, y, z):
	
		q = 0.1
		omega = 2 * 3.141592653589793
		A0 = 1.7320508075688772
		B = 1.4142135623730951
		C = 1
		
		A = A0 + (1 - exp(-q * tn) * sin(omega * tn))
		
		u = A * sin(z) + C * cos(y)
		v = B * sin(x) + A * cos(z)
		w = C * sin(y) + B * cos(x)
		
		return u, v, w
		
			
	def rk4(tk, xk, yk, zk):
		
		sign = 1
			
		k1x, k1y, k1z = abc(tk, xk, yk, zk)
			
		xk1 = xk+sign*(k1x/2)
		yk1 = yk+sign*(k1y/2)
		zk1 = zk+sign*(k1z/2)
		tk1 = tk+sign*(h/2)
			
		k2x, k2y, k2z = abc(tk1, xk1, yk1, zk1)
		
		xk2 = xk+sign*(k2x/2)
		yk2 = yk+sign*(k2y/2)
		zk2 = zk+sign*(k2z/2)
		tk2 = tk+sign*(h/2)
			
		k3x, k3y, k3z = abc(tk2, xk2, yk2, zk2)
		
		xk3 = xk+sign*k3x
		yk3 = yk+sign*k3y
		zk3 = zk+sign*k3z
		tk3 = tk+sign*h
			
		k4x, k4y, k4z = abc(tk3, xk3, yk3, zk3)
			
		xp1 = xk + sign*((1/6)*(k1x+2*k2x+2*k3x+k4x))
		yp1 = yk + sign*((1/6)*(k1y+2*k2y+2*k3y+k4y))
		zp1 = zk + sign*((1/6)*(k1z+2*k2z+2*k3z+k4z))
		
		return xp1, yp1, zp1
	
	startX = cuda.grid(1)
	gridX = cuda.gridDim.x * cuda.blockDim.x;
	
	for i in range(startX, len(XX), gridX):
		
		x_new, y_new, z_new = rk4(t, XX[i], YY[i], ZZ[i])
			
		_appendAdvectedVelocityx[i] = x_new
		_appendAdvectedVelocityy[i] = y_new
		_appendAdvectedVelocityz[i] = z_new
