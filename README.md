# OSP
The implementation of CVPR2022 paper ["Out-of-Distributed Semantic Pruning for Robust Semi-Supervised Learning"]().
## Overview
> Recent advances in robust semi-supervised learning (SSL) typically filter out-of-distribution (OOD) information at the sample level. We argue that an overlooked problem of robust SSL is its corrupted information on semantic level, practically limiting the development of the field. In this paper, we take an initial step to explore and propose a unified framework termed OOD Semantic Pruning (OSP), which aims at pruning OOD semantics out from in-distribution (ID) features. Specifically, (i) we propose an aliasing OOD matching module to pair each ID sample with an OOD sample with semantic overlap. (ii) We design a soft orthogonality regularization, which first transforms each ID feature by suppressing its semantic component that is collinear with paired OOD sample. It then forces the predictions before and after soft orthogonality decomposition to be consistent. Being practically simple, our method shows a strong performance in OOD detection and ID classification on challenging benchmarks. In particular, OSP surpasses the previous state-of-the-art by 13.7% on accuracy for ID classification and 5.9% on AUROC for OOD detection on TinyImageNet dataset. Codes are available in the supplementary material.
>
![avatar](https://github.com/rain305f/OSP/blob/main/images/motivation_final.jpg)
![avatar](https://github.com/rain305f/OSP/blob/main/images/methodv4.jpg)

## Quick Start 
### Dataset Preparation
    Download CIFAR-100 dataset under the directory data.
### Training
Train the model by of CIFAR-100 dataset with 100 labeled images per class and 20,000 unlabeled samples. Here, first-50 classes as ID categories, the later-50 classes as OOD categories. The mismatch ratio $\gamma$ "args.ratio' is 0.3 or 0.6.
```shell
export CUDA_VISIBLE_DEVICES="1"
# stage 1
python ours_stage1.py \
--dataset cifar100 \
--arch wideresnet \
--batch-size 64 \
--expand-labels \
--seed 5 \
--ratio 3 \
--total-steps 50000 \
--eval-step 500 \
--out results/cifar100@100_r3/ours_stage1 \
##
# stage 1
python cifar100_ours_stage2.py \
--dataset cifar100 \
--ood 1 \
--arch wideresnet \
--batch-size 64 \
--mu 5 \
--expand-labels \
--seed 5 \
--ratio 3 \
--total-steps 100000 \
--eval-step 1000 \
--resume results/cifar100@100_r3/ours_stage1/model_best.pth.tar \
--out results/cifar100@100_r3/ours_stage2 \

```
# Acknowledgement
Our codes are based on [T2T](https://github.com/huangjk97/T2T)

# Citation
```
@articl{

}

```
