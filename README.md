# Non-blind deconvolution methods benchmark

A benchmark for non-blind deconvolution methods: classical algorithms vs SOTA neural models.

---

## **Installation**

1. Install requirements (python >= 3.9):

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

## **Validation**

Just simply run:
```
make test
```

---

## **Sources of data**

### Kernels:

1. Motion blur:

    1.1 Levin et al. Understanding and evaluating blind deconvolution algorithms. [Paper](https://ieeexplore.ieee.org/abstract/document/5206815), [data source](https://webee.technion.ac.il/people/anat.levin/). Total: 8 kernels.

    1.2 Sun et al. Edge-based Blur Kernel Estimation Using Patch Priors. [Paper & data source](https://cs.brown.edu/people/lbsun/deblur2013/deblur2013iccp.html). Total: 8 kernels (img_1_kernel{i}_OurNat5x5_kernel.png).

    1.3 Generated with simulator taken from [RGDN](https://github.com/donggong1/learn-optimizer-rgdn). Source code: `src/data/generate/motion_blur.py`. 


2. Eye PSF:

    2.1 Generated with our own simulator. Size: 256*256.

3. Gauss:

    3.1 Generated with [this script](https://github.com/birdievera/Anisotropic-Gaussian/blob/master/gaussian_filter.py).


### Ground truth images

1. [BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/)

2. [Sun et al](https://cs.brown.edu/people/lbsun/deblur2013/deblur2013iccp.html). Images are taken from [here](https://drive.google.com/drive/folders/1Mb_mhtLG6N7CwiCMBnBMlJZyaqxQM-Nl).


---

## **Models and algorithms**

1. Wiener filter (as baseline): source code in `src/deconv/classic/wiener.py`;

2. [USRNet](https://github.com/cszn/USRNet): source code in `src/deconv/neural/usrnet`;

3. [DWDN](https://github.com/dongjxjx/dwdn): source code in `src/deconv/neural/dwdn`;

4. [KerUnc](https://github.com/ysnan/NBD_KerUnc): source code in `src/deconv/neural/kerunc`;

5. [RGDN](https://github.com/donggong1/learn-optimizer-rgdn): source code in `src/deconv/neural/rgdn`.


Example of each model inference can be found [here](notebooks/models.ipynb).

---

## **Tips**

### SQL

- If you work in VS Code, you can use this [extention for SQLLite](https://marketplace.visualstudio.com/items?itemName=alexcvzz.vscode-sqlite) to make your work easier.

- To calculate statistics (e.g. std and median), [this extention](https://github.com/nalgeon/sqlean/blob/main/docs/install.md) is used here. Just download precompiled binaries suitable for your OS and unpack them to a folder (`sqlean` in my case). That's it!

- SQL queries to analyze benchmarking results can be found [here](results/queries.sql).

### Running the code

- If old torch version (we use 1.7.1 since we took the source code for neural models as is) is not compatible with your CUDA version, you can run this code in Docker container. Instructures are below.

---


## **How to run docker container**

1. Build image:

```
make build
```

2. Run container:

```
make run
```

3. Execute inside the container:

```
make exec
```

4. Run inside the container:

```
make test
```