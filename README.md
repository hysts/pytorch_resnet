# PyTorch Implementation of ResNet

## Usage

```
$ python main.py --block_type basic --depth 110 --outdir results
```

## Results on CIFAR-10

| Model      | Test Error (median of 3 runs) | Test Error (in paper)     | Training Time |
|:-----------|:-----------------------------:|:-------------------------:|--------------:|
| ResNet-110 | 6.52                          | 6.43 (best), 6.61 ± 0.16 |   3h06m       |

![](figures/ResNet-110_basic.png)

## References

* He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. [arXiv:1512.03385]( https://arxiv.org/abs/1512.03385 )


