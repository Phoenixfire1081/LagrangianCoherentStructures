from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from cmath import exp

# Quasi-periodically perturbed Bickley jet

# psi(x, y, t) = -U * L * tanh (y/L) + U * L * sech(y/L)^2 * re(summ fn(t) exp (ikn * x))

# fn(t) = en * exp(-ikncnt)

def psi(x, y, t):
	
	# Define values

	r0 = 6371e-3

	eps1 = 0.0075
	eps2 = 0.4
	eps3 = 0.3

	U = 62.66e-6
	L = 1770e-3
	k1 = 2/r0
	k2 = 2*2/r0
	k3 = 2*3/r0

	c3 = 0.461*U
	c2 = 0.205*U
	c1 = c3 + ((np.sqrt(5)-1)/2)*(k2/k1)*(c2-c3)
	
	f1 = eps1 * exp(complex(imag = -k1 * c1 * t)) * exp(complex(imag = k1 * x))
	f2 = eps2 * exp(complex(imag = -k2 * c2 * t)) * exp(complex(imag = k2 * x))
	f3 = eps3 * exp(complex(imag = -k3 * c3 * t)) * exp(complex(imag = k3 * x))
	
	psi0 = - U * L * np.tanh(y/L)
	psi1 = U * L * ((1/np.cosh(y/L))**2) * np.real(f1 + f2 + f3)
	
	return psi0 + psi1
