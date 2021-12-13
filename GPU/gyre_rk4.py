from __future__ import division
from numba import cuda
from numba import *
from math import *

@cuda.jit
def gpu_advected(_appendAdvectedVelocityx, _appendAdvectedVelocityy, XX, YY, t, h, eVal, sign):
	
	A = 0.1
	omega = (2*3.14)/10

	def fun(x, t, e):
		
		a_t = e*sin(omega*t)
		b_t = 1-(2*e*sin(omega*t))
		
		return a_t*(x**2) + b_t*x
		
	def psi(xk, yk, tk):
	
		e = eVal
		
		return A * sin(3.14 * fun(xk, tk, e)) * sin(3.14 * yk)
		
			
	def rk4(tk, xk, yk):
		
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
