# Deep Learning Implementations
This is my repository to keep track of the deep-learning models that I implemented.  
I decided exclude simple models like convnets or mlps as they do not really showcase anything sophisticated.

## 1. Variational autoencoder (./vae)  
This is by far my most favorite deep-learning architecture. I love it because it integrates probabilistic views and variational inference to deep-learning models. VAE is still one of state of the art models for generative modeling.

## 2. Integrated Gradients (./integrated_gradients)  
I implemented a module that can wrap around Keras models to explain their predictions. Take a look at my original repository (https://github.com/hiranumn/IntegratedGradients) for this for more thorough readmes. 

## 3. DRAW (./draw)  
The model combines differentiable gausian filter attention and recurrent variational autoencoders.

## 4. 3D voxel modeling with generative models - VAE vs. GAN (./voxel_modeling)
Implemented 3D generative models both in VAE and GAN, and comparing the two. (3d-GAN is based on http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf)

## 5. Simple DC-gan on MNIST (./gan)
Implemented simple DC-generative adversarial network on MNIST data for debugging purpose.

## 6. VRNN Variational Recurrent Neural Nets (./vrnn)
Implemented https://arxiv.org/abs/1506.02216.

## 7. Semi-Superivised VAE Regressor (./SemiSupervisedRegressionWithVAE)
I took M2 model from Max Welling's VAE paper and adapted to regression setting. The model is a modified version of "Semi-Supervised Learning with Deep Generative Models"(https://arxiv.org/abs/1406.5298)
