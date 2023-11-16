from __future__ import division
from numpy import linalg as LA
import numpy as np
from numba import jit
import math

@jit(nopython=True)
def dotFast(mat1, mat2):
    s = 0
    mat = np.empty(shape=(mat1.shape[1], mat2.shape[0]), dtype=mat1.dtype)
    for r1 in range(mat1.shape[0]):
        for c2 in range(mat2.shape[1]):
            s = 0
            for j in range(mat2.shape[0]):
                s += mat1[r1,j] * mat2[j,c2]
            mat[r1,c2] = s
    return mat

@jit(nopython = True)
def FTLE_compute(xadvected_reshape, yadvected_reshape, XX_reshaped, YY_reshaped, res, intlen, _FTLEstep):
	
	counter = 0
	
	for i in range(res):
		for j in range(res):
			
			if i == 0:
				M11 = (xadvected_reshape[i+1,j] - xadvected_reshape[i,j]) / (XX_reshaped[i+1,j] - XX_reshaped[i,j])
				M21 = (yadvected_reshape[i+1,j] - yadvected_reshape[i,j]) / (XX_reshaped[i+1,j] - XX_reshaped[i,j])
			
			elif i == res-1:
				M11 = (xadvected_reshape[i,j] - xadvected_reshape[i-1,j]) / (XX_reshaped[i,j] - XX_reshaped[i-1,j])
				M21 = (yadvected_reshape[i,j] - yadvected_reshape[i-1,j]) / (XX_reshaped[i,j] - XX_reshaped[i-1,j])
			
			if j == 0:
				M12 = (xadvected_reshape[i,j+1] - xadvected_reshape[i,j]) / (YY_reshaped[i,j+1] - YY_reshaped[i,j])
				M22 = (yadvected_reshape[i,j+1] - yadvected_reshape[i,j]) / (YY_reshaped[i,j+1] - YY_reshaped[i,j])
			
			elif j == res-1:
				M12 = (xadvected_reshape[i,j] - xadvected_reshape[i,j-1]) / (YY_reshaped[i,j] - YY_reshaped[i,j-1])
				M22 = (yadvected_reshape[i,j] - yadvected_reshape[i,j-1]) / (YY_reshaped[i,j] - YY_reshaped[i,j-1])
			
			if i > 0 and i < res-1:
				M11 = (xadvected_reshape[i+1,j] - xadvected_reshape[i-1,j]) / (XX_reshaped[i+1,j] - XX_reshaped[i-1,j])
				M21 = (yadvected_reshape[i+1,j] - yadvected_reshape[i-1,j]) / (XX_reshaped[i+1,j] - XX_reshaped[i-1,j])
				
			if j > 0 and j < res-1:
				M12 = (xadvected_reshape[i,j+1] - xadvected_reshape[i,j-1]) / (YY_reshaped[i,j+1] - YY_reshaped[i,j-1])
				M22 = (yadvected_reshape[i,j+1] - yadvected_reshape[i,j-1]) / (YY_reshaped[i,j+1] - YY_reshaped[i,j-1])
			
			mat = np.array([[M11, M12],[M21, M22]])
			
			eigVal, eigVec = LA.eig(dotFast(mat.transpose(), mat))
			
			_FTLEstep[counter] = (1/intlen)*np.log(np.sqrt((np.max(eigVal))))
			
			counter += 1
			
	return _FTLEstep
