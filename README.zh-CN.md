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

* 解析：[ FaceNet: A Unified Embedding for Face Recognition and Clustering](https://blog.zhujian.life/posts/d2015a83.html)

基于[similarity](https://github.com/pytorch/vision/tree/main/references/similarity)实现，增加了如下功能：

1. 通过配置文件模块设置训练参数；
2. 支持多`GPU`训练以及混合精度训练；
3. 添加多种预处理函数以及训练配置。

## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [安装](#安装)
- [用法](#用法)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

[facenet](https://arxiv.org/abs/1503.03832)是一篇非常优秀的人脸识别论文，它创新性的提出了一种新的训练范式 - 三元组损失（`triplet loss`）训练。三元组损失的核心思想在于缩小同类人脸之间的欧式距离的同时尽可能的的扩大不同类之间的距离，每次训练都需要采集训练图像（锚点）以及同类正样本和不同类负样本。

[similarity](https://github.com/pytorch/vision/tree/main/references/similarity)提供了一个非常棒的`facenet`训练工程。每轮训练开始之前先通过采样器对数据进行分配，保证每批次数据中包含`P`个类别以及每个类别拥有`K`个训练样本；完成前向计算后对同批次数据进行两两计算，采集符合`semi-hard`定义（*负样本并没有比正样本接近于锚点，但是其距离仍位于边界范围内*）的正负样本对参与损失函数计算。

上述工程的不足之处在于它并不支持多`GPU`训练、混合精度训练以及没有很好的模块化设计导致扩展性不强，不能够直接作用于实际应用。 为了更好的训练`facenet`，本仓库基于[similarity](https://github.com/pytorch/vision/tree/main/references/similarity)进行了增强操作，提供了更友好的训练实现。

## 安装

```
$ pip install -r requirements.txt
```

## 用法

1. 获取数据文件。查看[how to get data](./docs/how-to-get-data.md)
2. 训练
   1. 单`GPU`训练
      ```angular2html
      $ CUDA_VISIBLE_DEVICES=0 python tools/train.py -cfg=configs/lfw/r18_lfw_224_e2_adam_g1.yaml
      ```
   2. 多`GPU`训练
      ```angular2html
      $ CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -cfg=configs/lfw/r18_lfw_224_e2_adam_g4.yaml
      ```

## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 致谢

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

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/ZJCV/facenet/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2021 zjykzj