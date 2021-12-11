<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/facenet"><img align="center" src="./imgs/facenet.png"></a></div>

<p align="center">
  «facenet» re-implements the paper <a href="https://arxiv.org/abs/1503.03832">FaceNet: A Unified Embedding for Face Recognition and Clustering</a>
<br>
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

Based on the [similarity](https://github.com/pytorch/vision/tree/main/references/similarity) implementation, the following functions have been added: 

1. Add configuration file module to support configurable training parameters;
2. Support Multi-GPU training  and  Mixed-Precision training;
3. Add a variety of preprocessing functions and training configuration.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

[facenet](https://arxiv.org/ABS/1503.03832) is an excellent face recognition paper, which innovatively puts forward a new training paradigm - `triplet loss training`. The core idea of triple loss is to reduce the Euclidean distance between similar faces and expand the distance between different classes as much as possible. Every training needs to collect training images (*anchor points*), positive samples of the same class and negative samples of different classes. 

[similarity](https://github.com/pytorch/vision/tree/main/references/similarity) provides an excellent `facenet`' training project. Before each round of training, distribute the data through the sampler to ensure that each batch of data contains `p` categories and each category has `k` training samples; After the forward calculation is completed, pairwise calculation is carried out on the same batch of data, and the positive and negative sample pairs that meet the definition of `semi-hard` (*the negative sample is not closer to the anchor point than the positive sample, but its distance is still within the boundary range*) are collected to participate in the loss function calculation. 

The deficiency of the above project lies in that it does not support multi-GPU training, mixed precision training and no good modular design, which leads to weak scalability and can not be directly applied to practical applications. In order to better train `facenet`, this warehouse has enhanced the operation based on [similarity](https://github.com/pytorch/vision/tree/main/references/similarity), providing a more friendly training implementation. 

## Installation

```
$ pip install -r requirements.txt
```

## Usage

1. Get data file. See [how to get data](./docs/how-to-get-data.md)
2. Training
   1. Single GPU training
      ```angular2html
      $ CUDA_VISIBLE_DEVICES=0 python tools/train.py -cfg=configs/lfw/r18_lfw_224_e2_adam_g1.yaml
      ```
   2. Multi GPU training
      ```angular2html
      $ CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -cfg=configs/lfw/r18_lfw_224_e2_adam_g4.yaml
      ```

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [ pytorch/vision](https://github.com/pytorch/vision)

```
@article{2015,
   title={FaceNet: A unified embedding for face recognition and clustering},
   url={http://dx.doi.org/10.1109/CVPR.2015.7298682},
   DOI={10.1109/cvpr.2015.7298682},
   journal={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
   publisher={IEEE},
   author={Schroff, Florian and Kalenichenko, Dmitry and Philbin, James},
   year={2015},
   month={Jun}
}
```

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/ZJCV/facenet/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2021 zjykzj