from __future__ import division
import numpy as np

# Time dependent double gyre

# f(x,t) = a(t)*x^2 + b(t)*x
# a(t) = e*sin(omega*t)
# b(t) = 1-2*e*sin(omega*t)

# u = -np.pi*(A)*np.sin(np.pi*f(x))*cos(pi*y)
# v = np.pi*(A)*np.cos(np.pi*f(x))*sin(pi*y)*(df/dx)

# A = 0.1, omega = 2*pi/10 and e = 0.25, T = 10s

A = 0.1
omega = (2*np.pi)/10

def fun(x, t, e):
	
	a_t = e*np.sin(omega*t)
	b_t = 1-(2*e*np.sin(omega*t))
	
	return a_t*(x**2) + b_t*x
	
def psi_rk(xk, yk, tk, e):
	
	return A * np.sin(np.pi * fun(xk, tk, e)) * np.sin(np.pi * yk)

def psi(xk, yk, tk, eVal):
	
	e = eVal
	
	delta = 0.0001
	
	uvel = -(A*np.sin(np.pi*fun(xk, tk, e))*np.sin(np.pi*(yk + delta)) - A*np.sin(np.pi*fun(xk, tk, e))*np.sin(np.pi*(yk - delta))) / (2 * delta)
	vvel = (A*np.sin(np.pi*fun(xk + delta, tk, e))*np.sin(np.pi*yk) - A*np.sin(np.pi*fun(xk - delta, tk, e))*np.sin(np.pi*yk)) / (2 * delta)
	
	return uvel, vvel
