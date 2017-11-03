# DC-GAN simple implementation

## Architecture
2 conv layers for a generative network and 2 deconv layers for a discriminator.   
Both conv fitlers are 5 by 5 with strides.   
Dense layers are added to reshape tensors into desired dimentions.  

## Results
- generated MNSIT samples 

![](figures/0.png)
![](figures/1.png)
![](figures/2.png)
![](figures/3.png)
![](figures/4.png)
![](figures/5.png)
![](figures/6.png)
![](figures/7.png)
![](figures/8.png)
![](figures/9.png)

- learning curves
![](figures/learning_curves.png)
