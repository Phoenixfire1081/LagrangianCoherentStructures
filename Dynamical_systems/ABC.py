from __future__ import division
import numpy as np

def uvw(x, y, z, tn):
	
	q = 0.1
	omega = 2 * np.pi
	A0 = np.sqrt(3)
	B = np.sqrt(2)
	C = 1
	
	A = A0 + (1 - np.exp(-q * tn) * np.sin(omega * tn))
	
	u = A * np.sin(z) + C * np.cos(y)
	v = B * np.sin(x) + A * np.cos(z)
	w = C * np.sin(y) + B * np.cos(x)
	
	return u, v, w
