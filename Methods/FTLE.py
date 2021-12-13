from __future__ import division
from numpy import linalg as LA
import numpy as np

def FTLE_compute(xadvected_reshape, yadvected_reshape, XX_reshaped, YY_reshaped, res, intlen):
	
	FTLEstep = []

	for i in range(1, res-1):
		for j in range(1, res-1):
				
			M11 = (xadvected_reshape[i+1,j] - xadvected_reshape[i-1,j]) / (XX_reshaped[i+1,j] - XX_reshaped[i-1,j])
			M12 = (xadvected_reshape[i,j+1] - xadvected_reshape[i,j-1]) / (YY_reshaped[i,j+1] - YY_reshaped[i,j-1])
			
			M21 = (yadvected_reshape[i+1,j] - yadvected_reshape[i-1,j]) / (XX_reshaped[i+1,j] - XX_reshaped[i-1,j])
			M22 = (yadvected_reshape[i,j+1] - yadvected_reshape[i,j-1]) / (YY_reshaped[i,j+1] - YY_reshaped[i,j-1])
			
			mat = [[M11, M12],[M21, M22]]
			
			eigVal, eigVec = LA.eig(np.matmul(np.transpose(mat), mat))
			
			FTLEstep.append((1/intlen)*np.log(np.sqrt((np.max(eigVal)))))
			
	return FTLEstep
