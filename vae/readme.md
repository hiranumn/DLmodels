# Variational Autoencode Implementation.
Auto-Encoding Variational Bayes by Diederik P Kingma, Max Welling (https://arxiv.org/abs/1312.6114)

This is a generic VAE implementation with fully-connected encoders and decoders. The number layers and hidden nodes can be specified through the arguments to the VAE constructor. You can parse through the MNIST example notebook to understand the usage.

Details and Features that are supported:
- L2 regularization on the fully-connected layers
- Batch normalizatoin
- Dropouts
- ELU as a default activation functions

Send me questions at hiranumn at cs.washington.edu
