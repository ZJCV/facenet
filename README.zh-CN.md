<div align="right">
  Language:
    🇨🇳
  <a title="English" href="./README.md">🇺🇸</a>
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

基于[similarity](https://github.com/pytorch/vision/tree/main/references/similarity)实现，增加了如下功能：

1. 通过配置文件模块设置训练参数；
2. 支持多`GPU`训练/混合精度训练；
3. 添加多种预处理函数以及训练配置。

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

...

## Usage

...

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