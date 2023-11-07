# SSCAE_contrastive_physics

### Software requirement

| Package Requirement                        |
|--------------------------------------------|
| os                                         |                                      |
| numpy                                      |                                   |
| math                                       |
| matplotlib                                 |                           |
| Tensorflow (2.3.0 or higher)               |
| Keras (2.4.0 or higher)                    |

implementation of contrastive learning in computational physics problems (fluid dynamics and wildfire simulation))


Contrastive learning forms the basis of numerous modern data fusion techniques and 'large' deep learning models, yet it is still underutilized in the realm of computational physics. This research aims to elucidate and address the strengths and limitations of contrastive learning in computational science, with a particular focus on dynamical systems. The paper introduces a novel framework, the Semi-Supervised Contrastive Autoencoder (SSCAE), which integrates contrastive learning into an encoder-decoder structure. The study evaluates two fundamental tasks in computational physics, field prediction, and parameter identification.
This repository contains two test cases, namely a deterministic shallow water fluid mechanics and a stochastic wildfir simulator 

![SW](https://github.com/scheng1992/SSCAE_contrastive_physics/assets/28357071/461df689-f9b3-4ad4-bbdf-1167ab880eb4)

![Wildfire](https://github.com/scheng1992/SSCAE_contrastive_physics/assets/28357071/f33658a7-b803-4bba-bb01-fee1b509523e)
The training script and the trained models are provided.

Our current findings suggest that while contrastive pretraining can significantly improve the parameter estimatio, its contribution to the field prediction task is limited, in particular, for deterministic system. This study unveils potential opportunities for exploring more effective training approaches and architectural designs that can better harness the potential of contrastive learning in intricate, high-dimensional physical domains.
