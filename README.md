# Non-blind deconvolution methods benchmark

A benchmark for non-blind deconvolution methods: classical algorithms vs SOTA neural models.

---

## Installation

1. Install requirements (python >= 3.10):

```
make install
```

2. Download prepared data:

```
TODO
```

or

Download raw data:

```
TODO
```

and unpack it:

```
make prepare_raw_data
```

## Validation

TODO

---

## Sources of data

### Kernels:

1. Motion blur:

    1.1 Levin et al. Understanding and evaluating blind deconvolution algorithms. [Paper](https://ieeexplore.ieee.org/abstract/document/5206815), [data source](https://webee.technion.ac.il/people/anat.levin/). Total: 8 kernels.

    1.2 Sun et al. Edge-based Blur Kernel Estimation Using Patch Priors. [Paper & data source](https://cs.brown.edu/people/lbsun/deblur2013/deblur2013iccp.html). Total: 8 kernels (img_1_kernel{i}_OurNat5x5_kernel.png).

    1.3 Generated with simulator taken from [RGDN](https://github.com/donggong1/learn-optimizer-rgdn). Source code: `src/data/generate/motion_blur.py`. 


2. Eye PSF:

    2.1 Generated with our own simulator 

3. Gauss:

    3.1 Generated:


### Ground truth images

1. [BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/)