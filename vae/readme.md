# Variational Autoencode Implementation.

Auto-Encoding Variational Bayes by Diederik P Kingma, Max Welling [https://arxiv.org/abs/1312.6114] .  
Towards Deeper Understanding of Variational Autoencoding Models by Shengjia Zhao, Jiaming Song, Stefano Ermon [https://arxiv.org/abs/1702.08658]

This is a generic VAE implementation with fully-connected encoders and decoders. The number layers and hidden nodes can be specified through the arguments to the VAE constructor. You can parse through the ipynb for the MNIST example to understand the usage. You can look at "Variational Autoencoder Implementation.ipynb" for how to implement VAe.

Details and Features that are supported:
- L2 regularization on the fully-connected layers
- Batch normalizatoin
- Dropouts
- ELU as a default activation functions

Send me questions at hiranumn at cs.washington.edu
