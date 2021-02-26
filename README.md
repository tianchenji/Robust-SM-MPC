# Robust SM-MPC
Code for the manuscript "Robust Model Predictive Control with Set-Membership State Estimation".

## Prerequisites
* [CasADi](https://web.casadi.org/)

Tested using Python 3.7 and CasADi 3.5

## Description of the code
* `algorithms`
    * `LQR.py`: Implementation of LQR
    * `Mayne_MPC.py`: Implementation of Mayne's MPC
    * `nominal_MPC.py`: Implementation of nominal MPC
    * `SM_MPC_aggressive.py`: Implementation of Aggressive SM-MPC
    * `SM_MPC_safe.py`: Implementation of Safe SM-MPC
    * `SSE.py`: Implementation of ellipsoidal set-membership state estimation
    * `utils.py`: Contains some utility functions.

* `examples`
    * `cartpole`: Generates simulation results for the cartpole example.
    * `double_integrator`: Generates quantitative results in Table 1 and Table 2 for the double integrator example.
    * `robotic_walker`: Generates simulation results in Figure 3 for the robotic walker example.
    * `quadrotor`: Generates simulation results in Figure 4 for the quadrotor example.

* `figures`: Generates figures in the manuscript.

* `SM_MPC_safe_2d.py`: Generates qualitative results in Figure 2 for the double integrator example.

Note that we directly implemented the Switching SM-MPC for the double integrator example and did not include it in the `algorithm` folder.

## Useful notes
* The grid size used for searching the optimal parameters for the set-membership state estimation can be adjusted in `line 94-95` in `SSE.py`. We used a grid size of 0.1 for the double integrator example and of 0.01 for the other three examples.
* As mentioned in the last paragraph of Subsection B in supplementary material, different full-dimensional bounding sets can be used for the computation of the contraint tightening. We provide two such bounding sets in our implementation:
    * axis-aligned bounding box (`line 196-197` in `SM_MPC_safe.py`): used in the double integrator and quadrotor example
    * arbitrarily oriented bounding box (`line 200-206` in `SM_MPC_safe.py`): used in the cartpole and robotic walker example

## Example run
* Run Safe SM-MPC for subsystem (*i*) in the robotic walker example:  
  `python -m examples.robotic_walker.SM_MPC_safe_rw_y`
  
* Generate closed-loop trajectory for the double integrator in Figure 2:  
  `python -m figures.double_integrator.plot_di`
