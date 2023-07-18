import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
import time
import array
from struct import *
from copy import deepcopy
import os

from numba import cuda

#---------------------------------------------------------------------#

# ---------Analysis of Multiscale Data from the Geosciences---------- #

# Author: Abhishek Harikrishnan
# Email: abhishek.harikrishnan@fu-berlin.de
# Last updated: 13-12-2021

#---------------------------------------------------------------------#

# Append paths

sys.path.append('Dynamical_systems/')
sys.path.append('Integrator/')
sys.path.append('Methods/')

# Supported Dynamical Systems

# 1. Quasi periodically perturbed bickley jet (2D) - 'Bickley'
# 2. Time dependent double gyre (2D) - 'gyre_d'
# 3. Time independent double gyre (2D) - 'gyre_id'
# 4. ABC flow (3D) - 'ABC'

# Supported data types

# 1. Data (2D) - 'Data2'
# 2. Data (3D) - 'Data3'

# Start timer

start_time = time.time()

# Supported _system formats: Bickley, Double gyre, ABC, Data

_system = 'Bickley'

_integratorType = 'rk4'

_integrationType = 'forward'

_computeVelocity = False
_writeVelocityData = False
_advectParticles = True
_computeFTLE = True

# Write parameters

_writeData = False

_writeTecplotData = False
_writeAmiraASCII = False
_writeAmiraBinary = False

# Visualization parameters

_velocityVisualization = False
_advectionVisualization = False
_contourFTLE = True

# Parallelization

_enableMultiProcessing = False
_enableGPU = True

# Initialization

if _writeVelocityData:
	os.system('mkdir Velocity')

if _enableMultiProcessing:
	
	print ('Multiprocessing enabled...')
	
	import multiprocessing as mp
	from multiprocessing import Pool
	
	# Count the available cores
	
	_nCPU = mp.cpu_count()
	
	print ('Available number of cores:', _nCPU)
	
	_CPU = 12
	
	print ('Using', _CPU, 'cores...')
	
if _enableGPU:
	
	sys.path.append('GPU/')
	
	from numba import cuda
	from numba import *
	from math import *
	
	# Collect GPU information

	meminfo = cuda.current_context().get_memory_info()
	print("free: %s bytes, total, %s bytes" % (meminfo[0], meminfo[1]))
		
	cc_cores_per_SM_dict = {
	(2,0) : 32,
	(2,1) : 48,
	(3,0) : 192,
	(3,5) : 192,
	(3,7) : 192,
	(5,0) : 128,
	(5,2) : 128,
	(6,0) : 64,
	(6,1) : 128,
	(7,0) : 64,
	(7,5) : 64,
	(8,0) : 64,
	(8,6) : 128
	}
	
	
	device = cuda.get_current_device()
	my_sms = getattr(device, 'MULTIPROCESSOR_COUNT')
	my_cc = device.compute_capability
	cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
	total_cores = cores_per_sm*my_sms
	
	print("GPU compute capability: " , my_cc)
	print("GPU total number of SMs: " , my_sms)
	print("total cores: " , total_cores)
	print('MAX_THREADS_PER_BLOCK:', getattr(device, 'MAX_THREADS_PER_BLOCK'))
	print('MAX_BLOCK_DIM_X:', getattr(device, 'MAX_BLOCK_DIM_X'))
	print('MAX_BLOCK_DIM_Y:', getattr(device, 'MAX_BLOCK_DIM_Y'))
	print('MAX_BLOCK_DIM_Z:', getattr(device, 'MAX_BLOCK_DIM_Z'))
	print('MAX_GRID_DIM_X:', getattr(device, 'MAX_GRID_DIM_X'))
	print('MAX_GRID_DIM_Y:', getattr(device, 'MAX_GRID_DIM_Y'))
	print('MAX_GRID_DIM_Z:', getattr(device, 'MAX_GRID_DIM_Z'))
	print('MAX_SHARED_MEMORY_PER_BLOCK:', getattr(device, 'MAX_SHARED_MEMORY_PER_BLOCK') / 1024, 'MB')
	print('TOTAL_CONSTANT_MEMORY:', getattr(device, 'TOTAL_CONSTANT_MEMORY') / 1024, 'MB')
	print('WARP_SIZE:', getattr(device, 'WARP_SIZE'))
	print('MULTIPROCESSOR_COUNT:', getattr(device, 'MULTIPROCESSOR_COUNT'))

# Time information

if _system == 'Bickley':

	_startTime = 0 * 24 * 60 * 60

	_integrationLength = 11 * 24 * 60 * 60

	_integrationTimeSteps = 600
	
	if _integrationType == 'forward':

		_timeSpan = np.linspace(_startTime, _integrationLength, _integrationTimeSteps)
		sign = 1
	
	else:
		
		_timeSpan = np.linspace(_integrationLength, _startTime, _integrationTimeSteps)
		sign = -1

	_timeStepper = _timeSpan[1] - _timeSpan[0]
	
elif _system == 'ABC':
	
	_startTime = 1.01

	_integrationLength = 10.01

	_integrationTimeSteps = 10

	_timeSpan = np.linspace(_startTime, _integrationLength, _integrationTimeSteps)

	_timeStepper = _timeSpan[1] - _timeSpan[0]
	
	_dataFileName = '1.01'
	
elif _system == 'gyre_d' or _system == 'gyre_id':
	
	if _system == 'gyre_d':
		
		eVal = 0.25
	
	else:
		
		eVal = 0
	
	_startTime = 1

	_integrationLength = 15

	_integrationTimeSteps = 100

	if _integrationType == 'forward':

		_timeSpan = np.linspace(_startTime, _integrationLength, _integrationTimeSteps)
		sign = 1
	
	else:
		
		_timeSpan = np.linspace(_integrationLength, _startTime, _integrationTimeSteps)
		sign = -1

	_timeStepper = _timeSpan[1] - _timeSpan[0]
	
	_dataFileName = '1'
	
elif _system == 'Data':
	
	_startTime = 0

	_integrationLengthOneTS = 0.036125*2
	_integrationLength = 0.036125*2

	_integrationTimeSteps = 2
	
	if _integrationType == 'forward':

		_timeSpan = np.linspace(_startTime, _integrationLength, _integrationTimeSteps)
		sign = 1
		
	else:
		
		_timeSpan = np.linspace(_integrationLength, _startTime, _integrationTimeSteps)
		sign = -1
	
	if _integrationTimeSteps == 1:
		
		_timeStepper = _integrationLength
	
	else:
		
		_timeStepper = _timeSpan[1] - _timeSpan[0]
	
	_dataFileName = 13
	
# Domain parameters

if _system == 'Bickley':

	_resolution = 1000

	X = np.linspace(0, 20, _resolution)
	Y = np.linspace(-3, 3, _resolution)

	XX, YY = np.meshgrid(X, Y, indexing = 'ij')

	XX = np.ravel(XX)
	YY = np.ravel(YY)

elif _system == 'gyre_d' or _system == 'gyre_id':

	_resolution = 100

	X = np.linspace(0, 2, _resolution)
	Y = np.linspace(0, 1, _resolution)

	XX, YY = np.meshgrid(X, Y, indexing = 'ij')

	XX = np.ravel(XX)
	YY = np.ravel(YY)
	
elif _system == 'ABC':
	
	_resolution = 256
	
	_resolutionx = _resolution
	_resolutiony = _resolution
	_resolutionz = _resolution
	
	X = np.linspace(0, 2*np.pi, _resolution)
	Y = np.linspace(0, 2*np.pi, _resolution)
	Z = np.linspace(0, 2*np.pi, _resolution)
	
	XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing = 'ij')
	
	XX = np.ravel(XX)
	YY = np.ravel(YY)
	ZZ = np.ravel(ZZ)
	
elif _system == 'Data':
	
	from interpolationFast import interp3_CPU
	from interpolationFastGPU import interp3_GPU
	
	# _dataResolutionx = 600
	# _dataResolutiony = 352
	# _dataResolutionz = 600
	
	_dataResolutionx = 600
	_dataResolutiony = 385
	_dataResolutionz = 600
	
	_resolution = 50
	divVal = 4
	
	_fileNameStart = 13
	
	# data resolution for 100 x: 0 -> 1.0472, y: -1 -> -0.4318 and z: 0 -> 0.5235
	
	# dx = 0.010489457941869092
	# dy = 0.005698005698005715
	# dz = 0.005244728970934546
	
	# xRes = dx * _dataResolutionx
	# yRes = dy * _dataResolutiony
	# zRes = dz * _dataResolutionz
	
	Xd = np.linspace(0, 2*np.pi, _dataResolutionx)
	# Yd = np.linspace(-1, 1, _dataResolutiony)
	Yd = (1 - np.cos(np.pi * (np.array(range(_dataResolutiony + 1))) / (len(range(_dataResolutiony + 1)) - 1)))-1
	Zd = np.linspace(0, np.pi, _dataResolutionz)
	
	XXd, YYd, ZZd = np.meshgrid(Xd, Yd, Zd, indexing = 'ij')
	
	XXd = np.ravel(XXd)
	YYd = np.ravel(YYd)
	ZZd = np.ravel(ZZd)
	
	X = np.linspace(0, 2*np.pi, _resolution)
	Y = np.linspace(-1, 1, _resolution)
	Z = np.linspace(0, np.pi, _resolution)
	
	XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing = 'ij')
	
	XX = np.ravel(XX)
	YY = np.ravel(YY)
	ZZ = np.ravel(ZZ)

	_uvwData = np.zeros((_resolution**3, 3, _integrationTimeSteps))
	
	if _computeLVGT:
		_uvwDataAlt = np.zeros((_resolution**3, 3, _integrationTimeSteps))
	
	# Data is read in as zFastest

	for i in range(_integrationTimeSteps):
		
		if _integrationType == 'forward':
		
			_fileName = str(_fileNameStart + i).zfill(3)
			
		else:
			
			_fileName = str(_fileNameStart - i).zfill(3)
		
		_u = array.array('f')
		_u.fromfile(open('/home/abhishek02/Dokumente/Scripts/FTLE/Modified/data/' + _fileName + 'u.bin', 'rb'), (_dataResolutionx*_dataResolutiony*_dataResolutionz*8) // 8)
		
		cArr = np.zeros((len(XX)), dtype = np.float32)
		cArrGPU = cuda.to_device(cArr)
		val = interp3_GPU[int(np.ceil(len(XX)/256)), 256](XX, YY, ZZ, np.reshape(_u, [_dataResolutionx, _dataResolutiony, _dataResolutionz]), Xd, Yd, Zd, cArrGPU)
		_uN = cArrGPU.copy_to_host()
		
		if _computeLVGT:
			_uNalt = _ufun(np.column_stack([XXalt, YYalt, ZZalt]))
		
		del _u
		
		_v = array.array('f')
		_v.fromfile(open('/home/abhishek02/Dokumente/Scripts/FTLE/Modified/data/' + _fileName + 'v.bin', 'rb'), (_dataResolutionx*_dataResolutiony*_dataResolutionz*8) // 8)
		
		cArr = np.zeros((len(XX)), dtype = np.float32)
		cArrGPU = cuda.to_device(cArr)
		val = interp3_GPU[int(np.ceil(len(XX)/256)), 256](XX, YY, ZZ, np.reshape(_v, [_dataResolutionx, _dataResolutiony, _dataResolutionz]), Xd, Yd, Zd, cArrGPU)
		_vN = cArrGPU.copy_to_host()
		
		if _computeLVGT:
			_vNalt = _vfun(np.column_stack([XXalt, YYalt, ZZalt]))
		
		del _v
		
		_w = array.array('f')
		_w.fromfile(open('/home/abhishek02/Dokumente/Scripts/FTLE/Modified/data/' + _fileName + 'w.bin', 'rb'), (_dataResolutionx*_dataResolutiony*_dataResolutionz*8) // 8)
		
		cArr = np.zeros((len(XX)), dtype = np.float32)
		cArrGPU = cuda.to_device(cArr)
		val = interp3_GPU[int(np.ceil(len(XX)/256)), 256](XX, YY, ZZ, np.reshape(_w, [_dataResolutionx, _dataResolutiony, _dataResolutionz]), Xd, Yd, Zd, cArrGPU)
		_wN = cArrGPU.copy_to_host()
		
		if _computeLVGT:
			_wNalt = _wfun(np.column_stack([XXalt, YYalt, ZZalt]))
		
		del _w
		
		_uvwData[:,0,i] = np.float32(_uN)
		_uvwData[:,1,i] = np.float32(_vN)
		_uvwData[:,2,i] = np.float32(_wN)
		
			
	
		del Xd, Yd, Zd,_uN, _vN, _wN, XX, YY, ZZ
		
		# Choose data points for advecting the tracers
		
		_resolutionx = 50
		_resolutiony = 50
		_resolutionz = 50
		
		xchunk = 0
		# X1 = np.linspace(2*np.pi*xchunk/8, 2*np.pi*(xchunk+1)/8, _resolutionx)
		X1 = np.linspace(0, 2*np.pi, _resolutionx)
		Y1 = np.linspace(-1, 1, _resolutiony)
		Z1 = np.linspace(0, np.pi, _resolutionz)
		
		XX, YY, ZZ = np.meshgrid(X1, Y1, Z1, indexing = 'ij')
	
		XX = np.ravel(XX)
		YY = np.ravel(YY)
		ZZ = np.ravel(ZZ)
	

# Compute

if _system == 'Bickley':
	
	from Bickley import psi
	
elif _system == 'ABC':
	
	from ABC import uvw
	
elif _system == 'gyre_d' or _system == 'gyre_id':
	
	from Double_gyre import psi
	from Double_gyre import psi_rk
	
elif _system == 'Data':
	
	from data_read import uvw
	
if _integratorType == 'rk4':
	
	from RK4 import rk4_int
	from RK4 import rk4_gyre
	from RK4 import rk4_3
	from RK4 import rk4_data
	from Euler import euler_data
	
if _computeVelocity:
	
	if _velocityVisualization:
		
		plt.ion()
		
		if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
			
			ax = plt.figure().gca()
			
		elif _system == 'ABC' or _system == 'Data':
			
			ax = plt.figure().gca(projection = '3d')
	
	_fileCounter = 0
	
	for p in _timeSpan:
		
		_appendedVelocity = []
				
		if _system == 'Data':
			
			pass
			
		else:
			
			for i in range(len(XX)):
			
				if _system == 'Bickley':
				
					_u, _v = psi_vel(XX[i], YY[i], p)
					
					_appendedVelocity.append([_u, _v])
			
				elif _system == 'gyre_d' or _system == 'gyre_id':
				
					_u, _v = psi(XX[i], YY[i], p, eVal)
					
					_appendedVelocity.append([_u, _v])
					
				elif _system == 'ABC':
					
					_u, _v, _w = uvw(XX[i], YY[i], ZZ[i], p)
					
					_appendedVelocity.append([_u, _v, _w])
		
		if _writeVelocityData:
			
			_appendedVelocity = np.array(_appendedVelocity)
			
			if np.shape(_appendedVelocity)[1] == 2:
			
				fw = open('Velocity/' + str(_fileCounter) + 'u.bin', 'wb')
				for i in range(len(_appendedVelocity[:, 0])):
					fw.write(pack('f' , _appendedVelocity[:, 0][i]))
				fw.close()
			
				fw = open('Velocity/' + str(_fileCounter) + 'v.bin', 'wb')
				for i in range(len(_appendedVelocity[:, 1])):
					fw.write(pack('f' , _appendedVelocity[:, 1][i]))
				fw.close()
				
				_fileCounter += 1
			
		
		if _velocityVisualization:
			
			_appendedVelocity = np.array(_appendedVelocity)
			
			if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
			
				ax.quiver(XX, YY, _appendedVelocity[:,0], _appendedVelocity[:,1])
				
			elif _system == 'ABC':
				
				ax.quiver(XX, YY, ZZ, _appendedVelocity[:,0], _appendedVelocity[:,1], _appendedVelocity[:,2])
				
			elif _system == 'Data':
				
				ax.quiver(XX, YY, ZZ, _uvwData[:, 0, int(p)], _uvwData[:, 1, int(p)], _uvwData[:, 2, int(p)], length=0.05, normalize=True)
			
			plt.pause(0.2)
			ax.clear()

# Advect Particles

if _advectParticles:
	
	# Seed particles and advect in time
	
	_timeCounter = 0
	
	if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
	
		_finalAdvectedVelocity = np.zeros((len(XX), 2))
		
	elif _system == 'ABC' or _system == 'Data':
		
		_finalAdvectedVelocity = np.zeros((len(XX), 3))
	
	if _advectionVisualization:
		
		if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
		
			plt.ion()
			ax = plt.figure().gca()
			ax.scatter(XX, YY)
			
		elif _system == 'ABC' or _system == 'Data':
			
			plt.ion()
			ax = plt.figure().gca(projection = '3d')
			ax.scatter(XX, YY, ZZ)
	
	_finalAVcounter = 0
	
	for p in _timeSpan:
		
		print( 'Advecting: ', p)
		print('Time:', _timeCounter)
		
		if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
			
			if _enableGPU:
				
				_appendAdvectedVelocity = np.float32(np.zeros((len(XX), 2)))
				_appendAdvectedVelocityx = np.float32(np.zeros((len(XX))))
				_appendAdvectedVelocityy = np.float32(np.zeros((len(YY))))
				
				_localTime = time.time()
				
				_appendAdvectedVelocityxGPU = cuda.to_device(_appendAdvectedVelocityx)
				_appendAdvectedVelocityyGPU = cuda.to_device(_appendAdvectedVelocityy)
				XX = np.float32(XX)
				YY = np.float32(YY)
				
				print( 'Transferred data to GPU in', time.time() - _localTime, 's')
				
			else:
				
				_appendAdvectedVelocity = np.zeros((len(XX), 2))
		
		elif _system == 'ABC' or _system == 'Data':
			
			if _enableGPU:
				
				_appendAdvectedVelocity = np.float32(np.zeros((len(XX), 3)))
				_appendAdvectedVelocityx = np.float32(np.zeros((len(XX))))
				_appendAdvectedVelocityy = np.float32(np.zeros((len(YY))))
				_appendAdvectedVelocityz = np.float32(np.zeros((len(ZZ))))
				
				_localTime = time.time()
				
				_appendAdvectedVelocityxGPU = cuda.to_device(_appendAdvectedVelocityx)
				_appendAdvectedVelocityyGPU = cuda.to_device(_appendAdvectedVelocityy)
				_appendAdvectedVelocityzGPU = cuda.to_device(_appendAdvectedVelocityz)
				XX = np.float32(XX)
				YY = np.float32(YY)
				ZZ = np.float32(ZZ)
				
				print( 'Transferred data to GPU in', time.time() - _localTime, 's')
				
			else:
				
				_appendAdvectedVelocity = np.zeros((len(XX), 3))
				
				if _computeLVGT:
					
					_appendAdvectedVelocityAlt = np.zeros((len(XX), 3))
		
		if _enableMultiProcessing:
			
			if _system == 'Data':
			
				if p == _timeSpan[0]:
				
					_uReshaped = np.float32(np.reshape(_uvwData[:, 0, 0], [_resolution, _resolution, _resolution]))
					_vReshaped = np.float32(np.reshape(_uvwData[:, 1, 0], [_resolution, _resolution, _resolution]))
					_wReshaped = np.float32(np.reshape(_uvwData[:, 2, 0], [_resolution, _resolution, _resolution]))
				
				else:
					
					_uReshaped = np.float32(np.reshape(_uvwData[:, 0, _timeCounter], [_resolution, _resolution, _resolution]))
					_vReshaped = np.float32(np.reshape(_uvwData[:, 1, _timeCounter], [_resolution, _resolution, _resolution]))
					_wReshaped = np.float32(np.reshape(_uvwData[:, 2, _timeCounter], [_resolution, _resolution, _resolution]))
			
			def mpiAdvect(i):
				
				if p == _timeSpan[0]:
					
					if _system == 'Bickley':
				
						_uNew, _vNew = rk4_int(XX[i], YY[i], p, psi, _timeStepper)
					
					elif _system == 'gyre_d' or _system == 'gyre_id':
				
						_uNew, _vNew = rk4_gyre(XX[i], YY[i], p, psi_rk, _timeStepper, eVal)
						
					elif _system == 'ABC':
						
						_uNew, _vNew, _wNew = rk4_3(XX[i], YY[i], ZZ[i], p, uvw, _timeStepper)
						
					elif _system == 'Data':
						
						_uNew, _vNew, _wNew = rk4_data(XX[i], YY[i], ZZ[i], p, uvw, _timeStepper, X, Y, Z, _uReshaped, _vReshaped, _wReshaped, sign)
						
						if _computeLVGT:
							
							_uNewAlt, _vNewAlt, _wNewAlt = rk4_data(XXalt[i], YYalt[i], ZZalt[i], p, uvw, _timeStepper, Xalt, Y, Z, _uvwDataAlt, sign)
				
				else:
					
					if _system == 'Bickley':
					
						_uNew, _vNew = rk4_int(_xNew[i], _yNew[i], p, psi, _timeStepper)
					
					elif _system == 'gyre_d' or _system == 'gyre_id':
					
						_uNew, _vNew = rk4_gyre(_xNew[i], _yNew[i], p, psi_rk, _timeStepper, eVal)
						
					elif _system == 'ABC':
						
						_uNew, _vNew, _wNew = rk4_3(_xNew[i], _yNew[i], _zNew[i], p, uvw, _timeStepper)
						
					elif _system == 'Data':
						
						_uNew, _vNew, _wNew = rk4_data(_xNew[i], _yNew[i], _zNew[i], p, uvw, _timeStepper, X, Y, Z, _uReshaped, _vReshaped, _wReshaped, sign)
						
						if _computeLVGT:
							
							_uNewAlt, _vNewAlt, _wNewAlt = rk4_data(_xNewAlt[i], _yNewAlt[i], _zNewAlt[i], p, uvw, _timeStepper, Xalt, Y, Z, _uvwDataAlt, sign)
						
				if _system == 'Bickley' or _system == 'gyre_id' or _system == 'gyre_d':
					
					return _uNew, _vNew
				
				else:
					
					if _computeLVGT:
						
						return _uNew, _vNew, _wNew,_uNewAlt, _vNewAlt, _wNewAlt
					
					else:
						
						return _uNew, _vNew, _wNew
				
			pool = Pool(processes = _CPU)
			_val = pool.map(mpiAdvect, range(len(XX)))
			
			pool.close()
			
			_val = np.float32(_val)
			
			if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
		
				_appendAdvectedVelocity[:, 0] = _appendAdvectedVelocity[:, 0] + _val[:, 0]
				_appendAdvectedVelocity[:, 1] = _appendAdvectedVelocity[:, 1] + _val[:, 1]
				
				del _val
				
			elif _system == 'ABC' or _system == 'Data':
				
				if _computeLVGT:
					
					_appendAdvectedVelocity[:, 0] = _appendAdvectedVelocity[:, 0] + _val[:, 0]
					_appendAdvectedVelocity[:, 1] = _appendAdvectedVelocity[:, 1] + _val[:, 1]
					_appendAdvectedVelocity[:, 2] = _appendAdvectedVelocity[:, 2] + _val[:, 2]
					
					_appendAdvectedVelocityAlt[:, 0] = _appendAdvectedVelocityAlt[:, 0] + _val[:, 3]
					_appendAdvectedVelocityAlt[:, 1] = _appendAdvectedVelocityAlt[:, 1] + _val[:, 4]
					_appendAdvectedVelocityAlt[:, 2] = _appendAdvectedVelocityAlt[:, 2] + _val[:, 5]
					
				else:
				
					_appendAdvectedVelocity[:, 0] = _appendAdvectedVelocity[:, 0] + _val[:, 0]
					_appendAdvectedVelocity[:, 1] = _appendAdvectedVelocity[:, 1] + _val[:, 1]
					_appendAdvectedVelocity[:, 2] = _appendAdvectedVelocity[:, 2] + _val[:, 2]
				
				del _val
				
		elif _enableGPU:
			
			if p == _timeSpan[0]:
				
				if _system == 'ABC':
					
					from ABC_rk4 import gpu_advected
					
					_blockDim = (256)
					_gridDim = (int(np.ceil(len(XX)/256)))
					
					_localTime = time.time()
					
					gpu_advected[_gridDim, _blockDim](_appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, _appendAdvectedVelocityzGPU, XX, YY, ZZ, np.float32(p), np.float32(_timeStepper))
					
					print( 'GPU compute for ', p, 'completed in', time.time() - _localTime, 's')
					
					_localTime = time.time()
				
					_appendAdvectedVelocityx = _appendAdvectedVelocityxGPU.copy_to_host()
					_appendAdvectedVelocityy = _appendAdvectedVelocityyGPU.copy_to_host()
					_appendAdvectedVelocityz = _appendAdvectedVelocityzGPU.copy_to_host()
					
					print ('Transferred array to host in', time.time() - _localTime, 's')
					
					_appendAdvectedVelocity[:, 0] = _appendAdvectedVelocityx
					_appendAdvectedVelocity[:, 1] = _appendAdvectedVelocityy
					_appendAdvectedVelocity[:, 2] = _appendAdvectedVelocityz
					
					del _appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, _appendAdvectedVelocityzGPU
					
				elif _system == 'Data':
					
					from Data3_rk4 import gpu_advected
					
					_blockDim = (256)
					_gridDim = (int(np.ceil(len(XX)/256)))
				
					# print cuda_driver.mem_get_info()
					
					_uReshaped = np.ascontiguousarray(np.float32(np.reshape(_uvwData[:, 0, 0], [_resolution, _resolution, _resolution])))
					_vReshaped = np.ascontiguousarray(np.float32(np.reshape(_uvwData[:, 1, 0], [_resolution, _resolution, _resolution])))
					_wReshaped = np.ascontiguousarray(np.float32(np.reshape(_uvwData[:, 2, 0], [_resolution, _resolution, _resolution])))
					
					# _uReshaped2 = np.ascontiguousarray(np.float32(np.reshape(_uvwData[:, 0, 1], [_resolution, _resolution, _resolution])))
					# _vReshaped2 = np.ascontiguousarray(np.float32(np.reshape(_uvwData[:, 1, 1], [_resolution, _resolution, _resolution])))
					# _wReshaped2 = np.ascontiguousarray(np.float32(np.reshape(_uvwData[:, 2, 1], [_resolution, _resolution, _resolution])))
					
					_localTime = time.time()
					
					gpu_advected[_gridDim, _blockDim](_appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, _appendAdvectedVelocityzGPU, np.float32(X), np.float32(Y), np.float32(Z), np.float32(XX), np.float32(YY), np.float32(ZZ), np.float32(p), np.float32(_timeStepper), _uReshaped, _vReshaped, _wReshaped, sign)
					# gpu_advected[_gridDim, _blockDim](_appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, _appendAdvectedVelocityzGPU, np.float32(X), np.float32(Y), np.float32(Z), np.float32(XX), np.float32(YY), np.float32(ZZ), np.float32(p), np.float32(_timeStepper), _uReshaped, _vReshaped, _wReshaped, _uReshaped2, _vReshaped2, _wReshaped2)
					
					print ('GPU compute for ', p, 'completed in', time.time() - _localTime, 's')
					
					_localTime = time.time()
				
					_appendAdvectedVelocityx = _appendAdvectedVelocityxGPU.copy_to_host()
					_appendAdvectedVelocityy = _appendAdvectedVelocityyGPU.copy_to_host()
					_appendAdvectedVelocityz = _appendAdvectedVelocityzGPU.copy_to_host()
					
					print ('Transferred array to host in', time.time() - _localTime, 's')
					
					_appendAdvectedVelocity[:, 0] = _appendAdvectedVelocityx
					_appendAdvectedVelocity[:, 1] = _appendAdvectedVelocityy
					_appendAdvectedVelocity[:, 2] = _appendAdvectedVelocityz
					
					del _appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, _appendAdvectedVelocityzGPU, _uReshaped, _vReshaped, _wReshaped
					
					# raise SystemError
					
				elif _system == 'Bickley':
					
					from Bickley_rk4 import gpu_advected
					
					_blockDim = (256)
					_gridDim = (int(np.ceil(len(XX)/256)))
					
					_localTime = time.time()
					
					gpu_advected[_gridDim, _blockDim](_appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, XX, YY, np.float32(p), np.float32(_timeStepper), sign)
					
					print ('GPU compute for ', p, 'completed in', time.time() - _localTime, 's')
					
					_localTime = time.time()
					
					_appendAdvectedVelocityx = _appendAdvectedVelocityxGPU.copy_to_host()
					_appendAdvectedVelocityy = _appendAdvectedVelocityyGPU.copy_to_host()
					
					print ('Transferred array to host in', time.time() - _localTime, 's')
					
					_appendAdvectedVelocity[:, 0] = _appendAdvectedVelocityx
					_appendAdvectedVelocity[:, 1] = _appendAdvectedVelocityy
					
					del _appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU
					
				elif _system == 'gyre_id' or _system == 'gyre_d':
					
					from gyre_rk4 import gpu_advected
					
					_blockDim = (256)
					_gridDim = (int(np.ceil(len(XX)/256)))
					
					_localTime = time.time()
					
					gpu_advected[_gridDim, _blockDim](_appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, XX, YY, np.float32(p), np.float32(_timeStepper), np.float32(eVal), sign)
					
					print ('GPU compute for ', p, 'completed in', time.time() - _localTime, 's')
					
					_localTime = time.time()
					
					_appendAdvectedVelocityx = _appendAdvectedVelocityxGPU.copy_to_host()
					_appendAdvectedVelocityy = _appendAdvectedVelocityyGPU.copy_to_host()
					
					print ('Transferred array to host in', time.time() - _localTime, 's')
					
					_appendAdvectedVelocity[:, 0] = _appendAdvectedVelocityx
					_appendAdvectedVelocity[:, 1] = _appendAdvectedVelocityy
					
					del _appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU
					
					# raise SystemError
			
			else:
				
				if _system == 'ABC':
					
					_localTime = time.time()
					
					_xNew = np.ascontiguousarray(_xNew)
					_yNew = np.ascontiguousarray(_yNew)
					_zNew = np.ascontiguousarray(_zNew)
					
					gpu_advected[_gridDim, _blockDim](_appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, _appendAdvectedVelocityzGPU, _xNew, _yNew, _zNew, np.float32(p), np.float32(_timeStepper))
					
					print ('GPU compute for ', p, 'completed in', time.time() - _localTime, 's')
					
					_localTime = time.time()
				
					_appendAdvectedVelocityx = _appendAdvectedVelocityxGPU.copy_to_host()
					_appendAdvectedVelocityy = _appendAdvectedVelocityyGPU.copy_to_host()
					_appendAdvectedVelocityz = _appendAdvectedVelocityzGPU.copy_to_host()
					
					print ('Transferred array to host in', time.time() - _localTime, 's')
					
					_appendAdvectedVelocity[:, 0] = _appendAdvectedVelocityx
					_appendAdvectedVelocity[:, 1] = _appendAdvectedVelocityy
					_appendAdvectedVelocity[:, 2] = _appendAdvectedVelocityz
					
					del _appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, _appendAdvectedVelocityzGPU
					
				elif _system == 'Bickley':
					
					_localTime = time.time()
					
					_xNew = np.ascontiguousarray(_xNew)
					_yNew = np.ascontiguousarray(_yNew)
					
					gpu_advected[_gridDim, _blockDim](_appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, _xNew, _yNew, np.float32(p), np.float32(_timeStepper), sign)
					
					print ('GPU compute for ', p, 'completed in', time.time() - _localTime, 's')
					
					_localTime = time.time()
				
					_appendAdvectedVelocityx = _appendAdvectedVelocityxGPU.copy_to_host()
					_appendAdvectedVelocityy = _appendAdvectedVelocityyGPU.copy_to_host()
					
					print ('Transferred array to host in', time.time() - _localTime, 's')
					
					_appendAdvectedVelocity[:, 0] = _appendAdvectedVelocityx
					_appendAdvectedVelocity[:, 1] = _appendAdvectedVelocityy
					
					del _appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU
						
					# raise SystemError
					
				elif _system == 'gyre_d' or _system == 'gyre_id':
					
					_localTime = time.time()
					
					_xNew = np.ascontiguousarray(_xNew)
					_yNew = np.ascontiguousarray(_yNew)
					
					gpu_advected[_gridDim, _blockDim](_appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, _xNew, _yNew, np.float32(p), np.float32(_timeStepper), np.float32(eVal), sign)
					
					print ('GPU compute for ', p, 'completed in', time.time() - _localTime, 's')
					
					_localTime = time.time()
				
					_appendAdvectedVelocityx = _appendAdvectedVelocityxGPU.copy_to_host()
					_appendAdvectedVelocityy = _appendAdvectedVelocityyGPU.copy_to_host()
					
					print ('Transferred array to host in', time.time() - _localTime, 's')
					
					_appendAdvectedVelocity[:, 0] = _appendAdvectedVelocityx
					_appendAdvectedVelocity[:, 1] = _appendAdvectedVelocityy
					
					del _appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU
					
				elif _system == 'Data':
					
					from Data3_rk4 import gpu_advected
					
					_blockDim = (256) # threads per block
					_gridDim = (int(np.ceil(len(_xNew)/256))) # blocks per grid
					
					_localTime = time.time()
					
					# print cuda_driver.mem_get_info()
					
					_uReshaped = np.ascontiguousarray(np.float32(np.reshape(_uvwData[:, 0, _timeCounter], [_resolution, _resolution, _resolution])))
					_vReshaped = np.ascontiguousarray(np.float32(np.reshape(_uvwData[:, 1, _timeCounter], [_resolution, _resolution, _resolution])))
					_wReshaped = np.ascontiguousarray(np.float32(np.reshape(_uvwData[:, 2, _timeCounter], [_resolution, _resolution, _resolution])))
					
					# _uReshaped2 = np.ascontiguousarray(np.float32(np.reshape(_uvwData[:, 0, int(p)+1], [_resolution, _resolution, _resolution])))
					# _vReshaped2 = np.ascontiguousarray(np.float32(np.reshape(_uvwData[:, 1, int(p)+1], [_resolution, _resolution, _resolution])))
					# _wReshaped2 = np.ascontiguousarray(np.float32(np.reshape(_uvwData[:, 2, int(p)+1], [_resolution, _resolution, _resolution])))
					
					_xNew = np.ascontiguousarray(_xNew)
					_yNew = np.ascontiguousarray(_yNew)
					_zNew = np.ascontiguousarray(_zNew)
					
					gpu_advected[_gridDim, _blockDim](_appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, _appendAdvectedVelocityzGPU, np.float32(X), np.float32(Y), np.float32(Z), _xNew, _yNew, _zNew, np.float32(p), np.float32(_timeStepper), _uReshaped, _vReshaped, _wReshaped, sign)
					
					print ('GPU compute for ', p, 'completed in', time.time() - _localTime, 's')
					
					_localTime = time.time()
				
					_appendAdvectedVelocityx = _appendAdvectedVelocityxGPU.copy_to_host()
					_appendAdvectedVelocityy = _appendAdvectedVelocityyGPU.copy_to_host()
					_appendAdvectedVelocityz = _appendAdvectedVelocityzGPU.copy_to_host()
					
					print ('Transferred array to host in', time.time() - _localTime, 's')
					
					_appendAdvectedVelocity[:, 0] = _appendAdvectedVelocityx
					_appendAdvectedVelocity[:, 1] = _appendAdvectedVelocityy
					_appendAdvectedVelocity[:, 2] = _appendAdvectedVelocityz
					
					del _appendAdvectedVelocityxGPU, _appendAdvectedVelocityyGPU, _appendAdvectedVelocityzGPU, _uReshaped, _vReshaped, _wReshaped
					
					# raise SystemError
				
		else:
		
			for i in range(len(XX)):
				
				if p == _timeSpan[0]:
					
					if _system == 'Bickley':
				
						_uNew, _vNew = rk4_int(XX[i], YY[i], p, psi, _timeStepper)
					
					elif _system == 'gyre_d' or _system == 'gyre_id':
				
						_uNew, _vNew = rk4_int(XX[i], YY[i], p, psi_rk, _timeStepper)
						
					elif _system == 'ABC':
						
						_uNew, _vNew, _wNew = rk4_3(XX[i], YY[i], ZZ[i], p, uvw, _timeStepper)
						
					elif _system == 'Data':
						
						_uNew, _vNew, _wNew = euler_data(XX[i], YY[i], ZZ[i], p, uvw, _timeStepper, X, Y, Z, _uvwData)
						
						#raise SystemError
				
				else:
					
					if _system == 'Bickley':
					
						_uNew, _vNew = rk4_int(_xNew[i], _yNew[i], p, psi, _timeStepper)
					
					elif _system == 'gyre_d' or _system == 'gyre_id':
					
						_uNew, _vNew = rk4_int(_xNew[i], _yNew[i], p, psi_rk, _timeStepper)
						
					elif _system == 'ABC':
						
						_uNew, _vNew, _wNew = rk4_3(_xNew[i], _yNew[i], _zNew[i], p, uvw, _timeStepper)
						
					elif _system == 'Data':
						
						_uNew, _vNew, _wNew = euler_data(_xNew[i], _yNew[i], _zNew[i], p, uvw, _timeStepper, X, Y, Z, _uvwData)
			
				if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
				
					_appendAdvectedVelocity[i, 0] = _appendAdvectedVelocity[i, 0] + _uNew
					_appendAdvectedVelocity[i, 1] = _appendAdvectedVelocity[i, 1] + _vNew
					
				elif _system == 'ABC' or _system == 'Data':
					
					_appendAdvectedVelocity[i, 0] = _appendAdvectedVelocity[i, 0] + _uNew
					_appendAdvectedVelocity[i, 1] = _appendAdvectedVelocity[i, 1] + _vNew
					_appendAdvectedVelocity[i, 2] = _appendAdvectedVelocity[i, 2] + _wNew
			
		# _appendAdvectedVelocity = np.array(_appendAdvectedVelocity)
			
		_xNew = _appendAdvectedVelocity[:,0]
		_yNew = _appendAdvectedVelocity[:,1]
		
		if _system == 'ABC' or _system == 'Data':
			
			_zNew = _appendAdvectedVelocity[:,2]
			
		if _advectionVisualization:
			
			if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
			
				ax.scatter(_appendAdvectedVelocity[:,0], _appendAdvectedVelocity[:,1])
				
			elif _system == 'ABC' or _system == 'Data':
				
				ax.scatter(_appendAdvectedVelocity[:,0], _appendAdvectedVelocity[:,1], _appendAdvectedVelocity[:,2])
				
			plt.pause(0.4)
			ax.clear()
			
		_timeCounter += 1
			
	_finalAdvectedVelocity[:, 0] = _finalAdvectedVelocity[:, 0] + _appendAdvectedVelocity[:, 0]
	_finalAdvectedVelocity[:, 1] = _finalAdvectedVelocity[:, 1] + _appendAdvectedVelocity[:, 1]

	if _system == 'ABC' or _system == 'Data':
		
		_finalAdvectedVelocity[:, 2] = _finalAdvectedVelocity[:, 2] + _appendAdvectedVelocity[:, 2]
		
	else:
	
		del _appendAdvectedVelocity, _xNew, _yNew
	
if _computeFTLE:
	
	if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
	
		from FTLE import FTLE_compute
		
		XX_reshaped = np.reshape(XX, (_resolution, _resolution))
		YY_reshaped = np.reshape(YY, (_resolution, _resolution))
			
		_FTLE = []
		
		xadvected_reshape = np.reshape(_finalAdvectedVelocity[:,0],(_resolution, _resolution))
		yadvected_reshape = np.reshape(_finalAdvectedVelocity[:,1],(_resolution, _resolution))
		
		_FTLE.append(FTLE_compute(xadvected_reshape, yadvected_reshape, XX_reshaped, YY_reshaped, _resolution, _integrationLength))
	
	if _system == 'ABC' or _system == 'Data':
		
		from FTLE3 import FTLE_compute

		XX_reshaped = np.reshape(XX, (_resolutionx, _resolutiony, _resolutionz))
		YY_reshaped = np.reshape(YY, (_resolutionx, _resolutiony, _resolutionz))
		ZZ_reshaped = np.reshape(ZZ, (_resolutionx, _resolutiony, _resolutionz))

		xadvected_reshape = np.reshape(_finalAdvectedVelocity[:,0],(_resolutionx, _resolutiony, _resolutionz))
		yadvected_reshape = np.reshape(_finalAdvectedVelocity[:,1],(_resolutionx, _resolutiony, _resolutionz))
		zadvected_reshape = np.reshape(_finalAdvectedVelocity[:,2],(_resolutionx, _resolutiony, _resolutionz))

		_FTLEstep = np.zeros(((_resolutionx)*(_resolutiony)*(_resolutionz)), dtype = np.float32) # change dtype as np.complex128 for domain change error

		_FTLE = FTLE_compute(xadvected_reshape, yadvected_reshape, zadvected_reshape, XX_reshaped, YY_reshaped, ZZ_reshaped, _resolutionx, _resolutiony, _resolutionz, _integrationLength, _FTLEstep)
	
	if _contourFTLE:
		
		if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
		
			_xShortened, _yShortened = np.meshgrid(np.linspace(X[1], X[-2], _resolution-2), np.linspace(Y[1], Y[-2], _resolution-2), indexing ='ij')
			_FTLEreshaped = np.reshape(_FTLE, (_resolution-2, _resolution-2))
			
			plt.ion()
			ax = plt.figure().gca()
			fig = ax.contourf(_xShortened, _yShortened, _FTLEreshaped, cmap = cm.bone)
			plt.colorbar(fig, ax = ax)
		
		elif _system == 'ABC' or _system == 'Data':
			
			_xShortened, _yShortened, _zShortened = np.meshgrid(np.linspace(X[1], X[-1], _resolutionx), np.linspace(Y[0], Y[-1], _resolutiony), np.linspace(Z[0], Z[-1], _resolutionz), indexing ='ij')
			_FTLEreshaped = np.reshape(_FTLE, (_resolutionx, _resolutiony, _resolutionz))
			
			print('NOTE: Only a 2D slice is shown')
			plt.ion()
			ax = plt.figure().gca()
			fig = ax.contourf(_xShortened[:, :, 0], _yShortened[:, :, 0], _FTLEreshaped[:, :, 0], cmap = cm.bone)
			plt.colorbar(fig, ax = ax)

# Write data

if _writeData:
		
	if _writeTecplotData:
		
		if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
		
			_xShortened = _xShortened.ravel()
			_yShortened = _yShortened.ravel()
			_FTLE = np.array(_FTLE)
			
			f = open('Bickley.dat','w+')
			f.write('TITLE = "2D Field Plot" \nVARIABLES = "x""y""FTLE"\n')
			f.write('ZONE ' + 'I='+str(_resolution-2)+' J='+str(_resolution-2)+'\n')
			
			for i in range(len(_xShortened)):
				f.write(str(_xShortened[i]) + ' ' + str(_yShortened[i]) + ' ' + str(_FTLE[0,i]) + '\n')
			
			f.close()
		
		elif _system == 'ABC' or _system == 'Data':
			
			_xShortened = _xShortened.ravel()
			_yShortened = _yShortened.ravel()
			_zShortened = _zShortened.ravel()
			_FTLE = np.array(_FTLE)
			
			f = open(str(_dataFileName) + '.dat','w+')
			f.write('TITLE = "3D Field Plot" \nVARIABLES = "x""y""z""FTLE"\n')
			f.write('ZONE ' + 'I='+str(_resolution-2)+' J='+str(_resolution-2)+' K='+str(_resolution-2)+'\n')
			
			for i in range(len(_xShortened)):
				f.write(str(_xShortened[i]) + ' ' + str(_yShortened[i]) + ' ' + str(_zShortened[i]) + ' ' + str(_FTLE[i]) + '\n')
			
			f.close()
			
	if _writeAmiraASCII:
		
		if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
		
			_xShortened = _xShortened.ravel()
			_yShortened = _yShortened.ravel()
			_FTLE = np.array(_FTLEreshaped.ravel())
			
			amFile = open('Bickley.am','w+')

			amFile.write('define Lattice ' + str(_resolution-2) + ' ' + str(_resolution-2) + '\n')
			amFile.write('Parameters {\n\
				CoordType "curvilinear"\n\
			}\n')

			amFile.write('Lattice { float ScalarField } = @1\n\
			Lattice { float[2] Coordinates } = @2\n')

			amFile.write('@1\n')

			for i in range(len(_xShortened)):
				amFile.write(str(_FTLE[i]) + '\n')
				
			amFile.write('@2\n')

			for i in range(len(_xShortened)):
				amFile.write(str(_xShortened[i]) + ' ' + str(_yShortened[i]) + '\n')

			amFile.close()
		
		elif _system == 'ABC' or _system == 'Data':
			
			_xShortened = _xShortened.ravel()
			_yShortened = _yShortened.ravel()
			_zShortened = _zShortened.ravel()
			_FTLE = np.array(_FTLE)
			
			amFile = open(_dataFileName + '.am','w+')

			amFile.write('define Lattice ' + str(_resolution-2) + ' ' + str(_resolution-2) + ' ' + str(_resolution-2) + '\n')
			amFile.write('Parameters {\n\
				CoordType "curvilinear"\n\
			}\n')

			amFile.write('Lattice { float ScalarField } = @1\n\
			Lattice { float[3] Coordinates } = @2\n')

			amFile.write('@1\n')

			for i in range(len(_xShortened)):
				amFile.write(str(_FTLE[i]) + '\n')
				
			amFile.write('@2\n')

			for i in range(len(_xShortened)):
				amFile.write(str(_xShortened[i]) + ' ' + str(_yShortened[i]) + ' ' + str(_zShortened[i]) + '\n')

			amFile.close()

	if _writeAmiraBinary:
		
		if _system == 'Bickley' or _system == 'gyre_d' or _system == 'gyre_id':
		
			_xShortened = _xShortened.ravel()
			_yShortened = _yShortened.ravel()
			_FTLE = np.array(_FTLEreshaped.ravel())
			
			amFile = open('Bickley.am','wb')
			amFile.write('# AmiraMesh BINARY-LITTLE-ENDIAN 2.0\n'.encode('ascii'))
			text = 'define Lattice ' + str(_resolution-2) + ' ' + str(_resolution-2) + '\n'
			amFile.write(text.encode('ascii'))
			text = 'define Coordinates ' + str((_resolution-2)*2) + '\n'
			amFile.write(text.encode('ascii'))
			text = 'Parameters {\n\
				CoordType "rectilinear"\n\
			}\n'
			amFile.write(text.encode('ascii'))

			text = 'Lattice { float ScalarField } = @1\n\
			Coordinates { float xyz } = @2\n'
			amFile.write(text.encode('ascii'))

			amFile.write('@1\n'.encode('ascii'))

			for ii in range((_resolution-2)**2):
				amFile.write(pack('f', _FTLE[ii]))
				
			amFile.write('\n@2\n'.encode('ascii'))

			for ii in range(1, len(X)-1):
				amFile.write(pack('f', X[ii])) 
			for ii in range(1, len(Y)-1):
				amFile.write(pack('f', Y[ii]))
				
			amFile.close()
		
		elif _system == 'ABC' or _system == 'Data':
			
			# _xShortened = _xShortened.ravel()
			# _yShortened = _yShortened.ravel()
			# _zShortened = _zShortened.ravel()
			_FTLE = np.reshape(_FTLE, (_resolutionx, _resolutiony, _resolutionz))
			_FTLE = _FTLE.T.ravel()
			
			amFile = open(str(_dataFileName) + '.am','wb')
			amFile.write('# AmiraMesh BINARY-LITTLE-ENDIAN 2.0\n'.encode('ascii'))
			text = 'define Lattice ' + str(_resolutionx) + ' ' + str(_resolutiony) + ' ' + str(_resolutionz) + '\n'
			amFile.write(text.encode('ascii'))
			text = 'define Coordinates ' + str((_resolutionx)+(_resolutiony)+(_resolutionz)) + '\n'
			amFile.write(text.encode('ascii'))
			text = 'Parameters {\n\
				CoordType "rectilinear"\n\
			}\n'
			amFile.write(text.encode('ascii'))

			text = 'Lattice { float ScalarField } = @1\n\
			Coordinates { float xyz } = @2\n'
			amFile.write(text.encode('ascii'))

			amFile.write('@1\n'.encode('ascii'))

			for ii in range((_resolutionx)*(_resolutiony)*(_resolutionz)):
				amFile.write(pack('f', _FTLE[ii]))
				
			amFile.write('\n@2\n'.encode('ascii'))

			for ii in range(len(X)):
				amFile.write(pack('f', X[ii])) 
			for ii in range(len(Y)):
				amFile.write(pack('f', Y[ii]))
			for ii in range(len(Z)):
				amFile.write(pack('f', Z[ii]))
				
			amFile.close()
		
print ('Time:', time.time()-start_time)
