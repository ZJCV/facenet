
# README

## LFW

|   arch   |  dataset | optimizer | epochs | gpus |     eval type     | accuracy | thresold |
|:--------:|:--------:|:---------:|:------:|:----:|:-----------------:|:--------:|:--------:|
| ResNet18 | LFW |    ADAM   |    2   |   1  | Face Verficiation |  99.818% |   0.38   |

## CIFAR

|   arch   |  dataset | optimizer | epochs | gpus |     eval type     | accuracy | thresold |
|:--------:|:--------:|:---------:|:------:|:----:|:-----------------:|:--------:|:--------:|
| ResNet18 | CIFAR100 |    ADAM   |    2   |   1  | Face Verficiation |  99.012% |   0.33   |
| ResNet18 | CIFAR100 |    ADAM   |    2   |   2  | Face Verficiation |  99.011% |   0.33   |
| ResNet18 | CIFAR100 |    ADAM   |    2   |   4  | Face Verficiation |  99.011% |   0.36   |

## FMNIST

|   arch   |  dataset | optimizer | epochs | gpus |     eval type     | accuracy | thresold |
|:--------:|:--------:|:---------:|:------:|:----:|:-----------------:|:--------:|:--------:|
| ResNet50 | FMNIST |    ADAM   |   10   |   1  | Face Verficiation |  98.142% |   0.73  |