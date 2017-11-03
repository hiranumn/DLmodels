# DC-GAN simple implementation

## Architecture
2 conv layers for a generative network and 2 deconv layers for a discriminator.   
Both conv fitlers are 5 by 5 with strides.   
Dense layers are added to reshape tensors into desired dimentions.  

## Results
- Generated MNSIT samples 

![](figures/0.png)
![](figures/1.png)
![](figures/2.png)
![](figures/3.png)
![](figures/4.png)  

- Learning curves

![](figures/learning_curves.png)
