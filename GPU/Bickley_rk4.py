from __future__ import division
from numba import cuda
from numba import *
from math import *

@cuda.jit
def gpu_advected(_appendAdvectedVelocityx, _appendAdvectedVelocityy, XX, YY, t, h, sign):
	
	def psi(x, y, t):
	
		# Define values

		r0 = 6371e-3

		eps1 = 0.075
		eps2 = 0.4
		eps3 = 0.3

		U = 62.66e-6
		L = 1770e-3
		k1 = 2/r0
		k2 = 2*2/r0
		k3 = 2*3/r0

		c3 = 0.461*U
		c2 = 0.205*U
		c1 = c3 + ((sqrt(5.0)-1.0)/2.0)*(k2/k1)*(c2-c3)
		
		f1 = eps1 * cos(-k1 * c1 * t) * cos(k1 * x)
		f2 = eps2 * cos(-k2 * c2 * t) * cos(k2 * x)
		f3 = eps3 * cos(-k3 * c3 * t) * cos(k3 * x)
		
		psi0 = - U * L * tanh(y/L)
		psi1 = U * L * ((1.0/cosh(y/L))**2) * (f1 + f2 + f3)
		
		return psi0 + psi1
		
			
	def rk4(tk, xk, yk):
		
		# sign = 1
			
		delta = 0.0001
		
		k1x = h*(-(psi(xk, yk+delta, tk) - psi(xk, yk-delta, tk))/(2*delta))
		k1y = h*((psi(xk+delta, yk, tk) - psi(xk-delta, yk, tk))/(2*delta))
		
		xk1 = xk + sign*(k1x/2)
		yk1 = yk + sign*(k1y/2)
		tk1 = tk + sign*(h/2)
		
		k2x = h*(-(psi(xk1, yk1+delta, tk1) - psi(xk1, yk1-delta, tk1))/(2*delta))
		k2y = h*((psi(xk1+delta, yk1, tk1) - psi(xk1-delta, yk1, tk1))/(2*delta))
		
		xk2 = xk + sign*(k2x/2)
		yk2 = yk + sign*(k2y/2)
		tk2 = tk + sign*(h/2)
		
		k3x = h*(-(psi(xk2, yk2+delta, tk2) - psi(xk2, yk2-delta, tk2))/(2*delta))
		k3y = h*((psi(xk2+delta, yk2, tk2) - psi(xk2-delta, yk2, tk2))/(2*delta))
		
		xk3 = xk + sign*k3x
		yk3 = yk + sign*k3y
		tk3 = tk + sign*h
		
		k4x = h*(-(psi(xk3, yk3+delta, tk3) - psi(xk3, yk3-delta, tk3))/(2*delta))
		k4y = h*((psi(xk3+delta, yk3, tk3) - psi(xk3-delta, yk3, tk3))/(2*delta))
		
		xp1 = xk + sign*((1/6)*(k1x+2*k2x+2*k3x+k4x))
		yp1 = yk + sign*((1/6)*(k1y+2*k2y+2*k3y+k4y))
		
		return xp1, yp1
	
	startX = cuda.grid(1)
	gridX = cuda.gridDim.x * cuda.blockDim.x;
	
	for i in range(startX, len(XX), gridX):
		
		x_new, y_new = rk4(t, XX[i], YY[i])
			
		_appendAdvectedVelocityx[i] = x_new
		_appendAdvectedVelocityy[i] = y_new
