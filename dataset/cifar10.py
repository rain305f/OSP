import os
import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)
cifar10_mean = (0.5, 0.5, 0.5)
cifar10_std = (0.5, 0.5, 0.5)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = x_u_v_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(root, train_labeled_idxs, train=True, transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))
    
    if args.ood_dataset == 'TIN':
        ood_fn = 'Imagenet_resize.npy'
    elif args.ood_dataset == 'LSUN':
        ood_fn = 'LSUN_resize.npy'
    elif args.ood_dataset == 'Gaussian':
        ood_fn = 'Gaussian.npy'
    elif args.ood_dataset == 'Uniform':
        ood_fn = 'Uniform.npy'
    
    ood_data = np.load(os.path.join(root, 'ood', ood_fn))

    train_unlabeled_dataset.data = np.vstack((train_unlabeled_dataset.data, ood_data))
    train_unlabeled_dataset.targets = np.concatenate(
        (train_unlabeled_dataset.targets, -np.ones(ood_data.shape[0], dtype=np.int)))

    val_dataset = CIFAR10SSL(root, val_idxs, train=True, transform=transform_val)

    test_dataset = CIFAR10SSL(root, None, train=False, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


# def x_u_v_split(args, labels):
#     label_per_class = args.n_labels_per_cls # args.num_labeled // args.num_classes
#     val_per_class = args.num_val // args.num_classes
#     labels = np.array(labels)
#     labeled_idx = []
#     val_idx = []
#     # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
#     # unlabeled_idx = np.array(range(len(labels)))
#     for i in range(args.num_classes):
#         idx = np.where(labels == i)[0]
#         idx = np.random.choice(idx, label_per_class + val_per_class, False)
#         labeled_idx.extend(idx[:label_per_class])
#         val_idx.extend(idx[-val_per_class:])
#     labeled_idx = np.array(labeled_idx)
#     val_idx = np.array(val_idx)
#     unlabeled_idx = np.array([i for i in range(len(labels)) if i not in val_idx])
#     assert len(labeled_idx) == args.num_labeled
#     assert len(val_idx) == args.num_val

#     if args.expand_labels or args.num_labeled < args.batch_size:
#         num_expand_x = math.ceil(
#             args.batch_size * args.eval_step / args.num_labeled)
#         labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
#     np.random.shuffle(labeled_idx)
#     return labeled_idx, unlabeled_idx, val_idx

def x_u_v_split(args, labels): 
    tot_class= args.tot_class
    n_unlabels = args.n_unlabels
    labels = np.array(labels)
    classes = np.unique(labels)
    n_labels = args.n_labels_per_cls * tot_class
    n_unlabels_per_cls = int(n_unlabels * (1.0 - args.ratio)) // tot_class 

    if (tot_class < len(classes)):
        n_unlabels_shift = (n_unlabels - (n_unlabels_per_cls * tot_class)) // (len(classes) - tot_class)

    labeled_idx = []
    unlabeled_idx = []
    val_idx = []
    for i in classes[:tot_class]:
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, args.n_labels_per_cls + n_unlabels_per_cls + args.n_val_per_class, True)
        labeled_idx.extend(idx[:args.n_labels_per_cls])
        unlabeled_idx.extend(idx[args.n_labels_per_cls:args.n_labels_per_cls + n_unlabels_per_cls])
        val_idx.extend(idx[-args.n_val_per_class:])
    
    if args.ood is not None:
        for i in classes[tot_class:]:
            idx = np.where(labels == i)[0]
            idx = np.random.choice(idx, n_unlabels_shift, False)
            unlabeled_idx.extend(idx[:n_unlabels_shift])

    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    val_idx = np.array(val_idx)

    if args.expand_labels:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / n_labels)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx, val_idx


DATASET_GETTERS = {'cifar10': get_cifar10}
