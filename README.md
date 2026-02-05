## Fast Electrical Impedance Tomography with Hybrid Priors

Fast and accurate Electrical Impedance Tomography reconstruction using hybrid regularization methods.

## Quick Start

`runGLSL.m` for EIDORS dataset.

`runGLSLreal.m` for real-world dataset.

`runGLSLlung.m` for human breathing dataset.

## Available Methods

SL: Sparse Learning

**GLSL**: Graph Laplacian and Sparse Learning

### Plug-and-play method

SL-NLM: Sparse Learning with non-local means denoising

**GLSL-BM3D**: GLSL with BM3D denoising

**GLSL-ST**: Download the pretrained [weight](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth) of Swin-Transformer and put into `./swinir/`.

Run `python -data data pnpeit.py`


## Data Sources
### Human Lung Data
The clinical EIT data was obtained from studies on human subjects.
```
@article{heinrich2006body,
  title={Body and head position effects on regional lung ventilation in infants: an electrical impedance tomography study},
  author={Heinrich, Sina and Schiffmann, Holger and Frerichs, Alexander and Klockgether-Radke, Adelbert and Frerichs, In{\'e}z},
  journal={Intensive care medicine},
  volume={32},
  number={9},
  pages={1392--1398},
  year={2006},
  publisher={Springer}
}
@article{wolf2007regional,
  title={Regional lung volume changes in children with acute respiratory distress syndrome during a derecruitment maneuver},
  author={Wolf, Gerhard K and Grychtol, Bartlomiej and Frerichs, Inez and van Genderingen, Huibert R and Zurakowski, David and Thompson, John E and Arnold, John H},
  journal={Critical care medicine},
  volume={35},
  number={8},
  pages={1972--1978},
  year={2007},
  publisher={LWW}
}
@article{wolf2012reversal,
  title={Reversal of dependent lung collapse predicts response to lung recruitment in children with early acute lung injury},
  author={Wolf, Gerhard K and G{\'o}mez-Laberge, Camille and Kheir, John N and Zurakowski, David and Walsh, Brian K and Adler, Andy and Arnold, John H},
  journal={Pediatric Critical Care Medicine},
  volume={13},
  number={5},
  pages={509--515},
  year={2012},
  publisher={LWW}
}
@article{gomez2012unified,
  title={A unified approach for EIT imaging of regional overdistension and atelectasis in acute lung injury},
  author={Gomez-Laberge, Camille and Arnold, John H and Wolf, Gerhard K},
  journal={IEEE transactions on medical imaging},
  volume={31},
  number={3},
  pages={834--842},
  year={2012},
  publisher={IEEE}
}
```

### Water Tank Data
The experimental EIT data was sourced from an open-access archive:
```
@article{uef2017,
	title={Open 2D electrical impedance tomography data archive},
	author={Hauptmann, Andreas and Kolehmainen, Ville and Mach, Nguyet Minh and Savolainen, Tuomo and Sepp{\"a}nen, Aku and Siltanen, Samuli},
	journal={arXiv preprint arXiv:1704.01178},
	year={2017}
}
```
