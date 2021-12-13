from __future__ import division
def euler_data(xk, yk, zk, tk, abc, h, XX, YY, ZZ, _uvwData):
	
	# if integration_type == 'Forward':
		
		_xx, _yy, _zz = abc(xk, yk, zk, tk, XX, YY, ZZ, _uvwData)
		
		xp1 = xk + (h * _xx)
		yp1 = yk + (h * _yy)
		zp1 = zk + (h * _zz)
		
	# elif integration_type == 'Backward':
	
		# h = 1
		
		# k1x, k1y, k1z = abc(tk, xk, yk, zk)
		
		# xk1 = xk-(k1x/2)
		# yk1 = yk-(k1y/2)
		# zk1 = zk-(k1z/2)
		# tk1 = tk-(h/2)
		
		# k2x, k2y, k2z = abc(tk1, xk1, yk1, zk1)
		
		# xk2 = xk-(k2x/2)
		# yk2 = yk-(k2y/2)
		# zk2 = zk-(k2z/2)
		# tk2 = tk-(h/2)
		
		# k3x, k3y, k3z = abc(tk2, xk2, yk2, zk2)
		
		# xk3 = xk-k3x
		# yk3 = yk-k3y
		# zk3 = zk-k3z
		# tk3 = tk-h
		
		# k4x, k4y, k4z = abc(tk3, xk3, yk3, zk3)
		
		# xp1 = xk - ((1/6)*(k1x+2*k2x+2*k3x+k4x))
		# yp1 = yk - ((1/6)*(k1y+2*k2y+2*k3y+k4y))
		# zp1 = zk - ((1/6)*(k1z+2*k2z+2*k3z+k4z))
		
	# else:
		
		# raise SystemError
		
		#print xp1, yp1, zp1
	
		return xp1, yp1, zp1
