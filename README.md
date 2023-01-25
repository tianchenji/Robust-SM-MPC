# Robust-SM-MPC
Code for our paper [Robust Output Feedback MPC with Reduced Conservatism under Ellipsoidal Uncertainty](https://ieeexplore.ieee.org/abstract/document/9992704) at [CDC 2022](https://cdc2022.ieeecss.org/).

We implemented three different algorithms: two tubes, single tube and the proposed SM MPC. Detailed descriptions of each method can be found in the paper.

## Prerequisites
* [CasADi](https://web.casadi.org/)

Tested using Python 3.7 and CasADi 3.5

## Description of the code
The files containing "cstr" only generate the constraint tightening, while the files without "cstr" generate the closed loop trajectories. "qr" in the file name means quadrotor. `SSE.py` is the implementation of set-membership state estimation. More detailed comments can be found in the code.

The `results` folder contains all necessary data for the results presented in the paper (i.e., Figure 2, 3, and 4).

## Citation
If you find the code useful, please consider citing our paper:
```
@inproceedings{ji2022robust,
  title={Robust Output Feedback MPC with Reduced Conservatism under Ellipsoidal Uncertainty},
  author={Ji, Tianchen and Geng, Junyi and Driggs-Campbell, Katherine},
  booktitle={2022 IEEE 61st Conference on Decision and Control (CDC)},
  pages={1782--1789},
  year={2022},
  organization={IEEE}
}
```

## Contact
Feel free to reach me at tj12@illinois.edu if you have any questions.
