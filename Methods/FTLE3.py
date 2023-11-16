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
def FTLE_compute(xadvected_reshape, yadvected_reshape, zadvected_reshape, XX_reshaped, YY_reshaped, ZZ_reshaped, resx, resy, resz, intlen, _FTLEstep):
	
	counter = 0
	mat = np.zeros((3, 3), dtype = np.float32); eigVal = np.zeros((0), dtype = np.float32)

	for i in range(resx):
		for j in range(resy):
			for k in range(resz):
				
				if i == 0:
					M11 = (xadvected_reshape[i+1,j,k] - xadvected_reshape[i,j,k]) / (XX_reshaped[i+1,j,k] - XX_reshaped[i,j,k])
					M21 = (yadvected_reshape[i+1,j,k] - yadvected_reshape[i,j,k]) / (XX_reshaped[i+1,j,k] - XX_reshaped[i,j,k])
					M31 = (zadvected_reshape[i+1,j,k] - zadvected_reshape[i,j,k]) / (XX_reshaped[i+1,j,k] - XX_reshaped[i,j,k])
				
				elif i == resx-1:
					M11 = (xadvected_reshape[i,j,k] - xadvected_reshape[i-1,j,k]) / (XX_reshaped[i,j,k] - XX_reshaped[i-1,j,k])
					M21 = (yadvected_reshape[i,j,k] - yadvected_reshape[i-1,j,k]) / (XX_reshaped[i,j,k] - XX_reshaped[i-1,j,k])
					M31 = (zadvected_reshape[i,j,k] - zadvected_reshape[i-1,j,k]) / (XX_reshaped[i,j,k] - XX_reshaped[i-1,j,k])
				
				if j == 0:
					M12 = (xadvected_reshape[i,j+1,k] - xadvected_reshape[i,j,k]) / (YY_reshaped[i,j+1,k] - YY_reshaped[i,j,k])
					M22 = (yadvected_reshape[i,j+1,k] - yadvected_reshape[i,j,k]) / (YY_reshaped[i,j+1,k] - YY_reshaped[i,j,k])
					M32 = (zadvected_reshape[i,j+1,k] - zadvected_reshape[i,j,k]) / (YY_reshaped[i,j+1,k] - YY_reshaped[i,j,k])
				
				elif j == resy-1:
					M12 = (xadvected_reshape[i,j,k] - xadvected_reshape[i,j-1,k]) / (YY_reshaped[i,j,k] - YY_reshaped[i,j-1,k])
					M22 = (yadvected_reshape[i,j,k] - yadvected_reshape[i,j-1,k]) / (YY_reshaped[i,j,k] - YY_reshaped[i,j-1,k])
					M32 = (zadvected_reshape[i,j,k] - zadvected_reshape[i,j-1,k]) / (YY_reshaped[i,j,k] - YY_reshaped[i,j-1,k])
				
				if k == 0:
					M13 = (xadvected_reshape[i,j,k+1] - xadvected_reshape[i,j,k]) / (ZZ_reshaped[i,j,k+1] - ZZ_reshaped[i,j,k])
					M23 = (yadvected_reshape[i,j,k+1] - yadvected_reshape[i,j,k]) / (ZZ_reshaped[i,j,k+1] - ZZ_reshaped[i,j,k])
					M33 = (zadvected_reshape[i,j,k+1] - zadvected_reshape[i,j,k]) / (ZZ_reshaped[i,j,k+1] - ZZ_reshaped[i,j,k])
					
				elif k == resz-1:
					M13 = (xadvected_reshape[i,j,k] - xadvected_reshape[i,j,k-1]) / (ZZ_reshaped[i,j,k] - ZZ_reshaped[i,j,k-1])
					M23 = (yadvected_reshape[i,j,k] - yadvected_reshape[i,j,k-1]) / (ZZ_reshaped[i,j,k] - ZZ_reshaped[i,j,k-1])
					M33 = (zadvected_reshape[i,j,k] - zadvected_reshape[i,j,k-1]) / (ZZ_reshaped[i,j,k] - ZZ_reshaped[i,j,k-1])
					
				if i > 0 and i < resx-1:
					M11 = (xadvected_reshape[i+1,j,k] - xadvected_reshape[i-1,j,k]) / (XX_reshaped[i+1,j,k] - XX_reshaped[i-1,j,k])
					M21 = (yadvected_reshape[i+1,j,k] - yadvected_reshape[i-1,j,k]) / (XX_reshaped[i+1,j,k] - XX_reshaped[i-1,j,k])
					M31 = (zadvected_reshape[i+1,j,k] - zadvected_reshape[i-1,j,k]) / (XX_reshaped[i+1,j,k] - XX_reshaped[i-1,j,k])
					
				if j > 0 and j < resy-1:
					M12 = (xadvected_reshape[i,j+1,k] - xadvected_reshape[i,j-1,k]) / (YY_reshaped[i,j+1,k] - YY_reshaped[i,j-1,k])
					M22 = (yadvected_reshape[i,j+1,k] - yadvected_reshape[i,j-1,k]) / (YY_reshaped[i,j+1,k] - YY_reshaped[i,j-1,k])
					M32 = (zadvected_reshape[i,j+1,k] - zadvected_reshape[i,j-1,k]) / (YY_reshaped[i,j+1,k] - YY_reshaped[i,j-1,k])
				
				if k > 0 and k < resz-1:
					M13 = (xadvected_reshape[i,j,k+1] - xadvected_reshape[i,j,k-1]) / (ZZ_reshaped[i,j,k+1] - ZZ_reshaped[i,j,k-1])
					M23 = (yadvected_reshape[i,j,k+1] - yadvected_reshape[i,j,k-1]) / (ZZ_reshaped[i,j,k+1] - ZZ_reshaped[i,j,k-1])
					M33 = (zadvected_reshape[i,j,k+1] - zadvected_reshape[i,j,k-1]) / (ZZ_reshaped[i,j,k+1] - ZZ_reshaped[i,j,k-1])
				
				# mat = [[M11, M12, M13],[M21, M22, M23],[M31, M32, M33]]
				mat[0][0] = M11
				mat[0][1] = M12
				mat[0][2] = M13
				mat[1][0] = M21
				mat[1][1] = M22
				mat[1][2] = M23
				mat[2][0] = M31
				mat[2][1] = M32
				mat[2][2] = M33
				
				eigVal, eigVec = LA.eig(dotFast(mat.transpose(), mat))
				
				_FTLEstep[counter] = (1/intlen)*math.log(math.sqrt((max(eigVal))))
				counter += 1
			
	return _FTLEstep
