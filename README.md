# Lagrangian coherent structures with Finite-Time Lyaponov Exponent (FTLE)

This page has been setup to support the work presented in the poster, **_Lagrangian hairpins in atmospheric boundary layers_**. Please find the poster here: https://doi.org/10.1103/APS.DFD.2021.GFM.P0018.

This version of the code features all functions as described in the poster. However, reading 2D and 3D data needs to be made more user-friendly. This will be updated over time. 

As of now, one can play with the parameters of a _Bickley jet, time dependent and independent double gyre and ABC flow data_. The code structure takes a module based approach where one or several modules can be activated. For instance, to run a Bickley jet example, one needs to set the following parameters:

_\_system = 'Bickley'_\
_\_integrationType = 'forward' # or 'backward'_\
_\_computeVelocity = False_\
_\_writeVelocityData = False_\
_\_advectParticles = True_\
_\_computeFTLE = True_

Other parameters such as the start, end time and number of integration steps can be set within _if \_system == 'Bickley':_.\
Further parameters for the Bickley jet can be set in the file Dynamical_systems/Bickley.py for serial and multiprocessing modes. For GPU, navigate to GPU/Bickley_rk4.py.

If you'd like to have a visualization with python's matplotlib, the following needs to be set to True

_\_contourFTLE = True_

Finally, the code can be run with 3 modes: Serial, CPU-parallelized with multiprocessing and GPU-parallelized with numba-cuda. 

For CPU-parallelization, set _\_enableMultiProcessing = True_ and set _\_CPU = 'The number of CPUs you want'_.
For GPU-parallelization, set _\_enableGPU = True_. Block dimensions and grid dimensions can be set within the code if necessary. 
The default mode is serial when both _\_enableMultiProcessing_ and _\_enableGPU_ are False. 

Finally, the code feaures writing the result in either Tecplot ASCII, Amira ASCII or Amira binary formats. Support for RAW3D will be added soon. 

**Please cite the following work if this code is used**

_BibTex_

@article{harikrishnanp0018,\
  title={P0018: Lagrangian hairpins in atmospheric boundary layers},\
  author={Harikrishnan, Abhishek Paraswarar and Ernst, Natalia and Ansorge, Cedrick and Klein, Rupert and Vercauteren, Nikki},\
  publisher={APS}\
}

_APA_

Harikrishnan, A. P., Ernst, N., Ansorge, C., Klein, R., & Vercauteren, N. P0018: Lagrangian hairpins in atmospheric boundary layers.

# Gallery

## Bickley jet
The parameters used for this example are obtained from [1]. 
![Screenshot](Plots/Bickley_perturbed.png)

If we change the amplitudes, eps1 = 0.0075 and eps2 = 0.04 instead of eps1 = 0.075 eps2 = 0.4, then the following ensues:
![Screenshot](Plots/Bickley_steady.png)

## Double gyre
Similarly, the parameters for the double gyre are obtained from [2].
![Screenshot](Plots/Time_dependent_gyre.png)
![Screenshot](Plots/Time_independent_gyre.png)

## ABC flow
For the ABC flow, the parameters are obtained from [3].
![Screenshot](Plots/ABC_slice.png)
![Screenshot](Plots/ABC_3D.png)

# References
[1] Hadjighasem, A., Farazmand, M., Blazevski, D., Froyland, G., & Haller, G. (2017). A critical comparison of Lagrangian methods for coherent structure detection. Chaos: An Interdisciplinary Journal of Nonlinear Science, 27(5).
[2] https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html
[3] Haller, G. (2005). An objective definition of a vortex. Journal of fluid mechanics, 525, 1-26.
