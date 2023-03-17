import os
import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)
cifar10_mean = (0.5, 0.5, 0.5)
cifar10_std = (0.5, 0.5, 0.5)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

class CIFAR100FIX(datasets.CIFAR100):
    def __init__(self, root, num_super=10, train=True, transform=None,
                 target_transform=None, download=False, return_idx=True):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])  #父类的标签，0-9:ID（55个子类）;
        self.course_labels = coarse_labels[self.targets]

        # course_labels 用来做什么
        self.targets = np.array(self.targets)

        labels_unknown = self.targets[np.where(self.course_labels > num_super)[0]]
        labels_known = self.targets[np.where(self.course_labels <= num_super)[0]]
        unknown_categories = np.unique(labels_unknown)
        known_categories = np.unique(labels_known)

        num_unknown = len(unknown_categories)
        num_known = len(known_categories)
#         print("number of unknown categories %s"%num_unknown)
#         print("number of known categories %s"%num_known)
        assert num_known + num_unknown == 100
        #new_category_labels = list(range(num_known))
        self.targets_new = np.zeros_like(self.targets)
        for i, known in enumerate(known_categories):
            ind_known = np.where(self.targets==known)[0]
            self.targets_new[ind_known] = i
        for i, unknown in enumerate(unknown_categories):
            ind_unknown = np.where(self.targets == unknown)[0]
            self.targets_new[ind_unknown] = num_known

        self.targets = self.targets_new
        assert len(np.where(self.targets == num_known)[0]) == len(labels_unknown)
        assert len(np.where(self.targets < num_known)[0]) == len(labels_known)
        self.num_known_class = num_known


    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



class CIFAR100SSL(datasets.CIFAR100): #(CIFAR100FIX):
    def __init__(self, root, indexs,  train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=True):
        super().__init__(root,train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.return_idx = return_idx
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

        self.set_index()
    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.targets_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets


    def __getitem__(self, index):
        img, target = self.data_index[index], self.targets_index[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.data_index)



def get_cifar100(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])

 
    base_dataset = datasets.CIFAR100(root, train=True, download=True)  #CIFAR100FIX(root, train=True, download=True, num_super=args.num_super)

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = x_u_v_split(args, base_dataset.targets ) #x_u_split(args, base_dataset.targets  )

    train_labeled_dataset = CIFAR100SSL(root, train_labeled_idxs, train=True, transform=transform_labeled)
    
    train_unlabeled_dataset = CIFAR100SSL(root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar10_std))

    val_dataset = CIFAR100SSL(root, val_idxs, train=True, transform=transform_val)
    
    test_dataset = CIFAR100SSL(root, None , train=False, transform=transform_val)
    test_label = np.array(test_dataset.targets)
    test_idx = np.where(test_label < args.tot_class)[0]
 
    test_dataset = CIFAR100SSL(root, test_idx , train=False, transform=transform_val)


    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset



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
    
    # if args.ood is not None:
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




def x_u_split(args, labels ):
    label_per_class = args.num_labeled
    val_per_class = args.num_val
#     test_per_class = args.num_test
    
    labels = np.array(labels)
    labeled_idx = []
    val_idx = []
    unlabeled_idx = []
    
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    for i in range(args.num_id_classes): # split all the id classes
        idx = np.where(labels == i)[0]
        unlabeled_idx.extend(idx)
        idx = np.random.choice(idx, label_per_class + val_per_class ,False)
        labeled_idx.extend(idx[:label_per_class])
        val_idx.extend(idx[label_per_class:])
        

    labeled_idx = np.array(labeled_idx)
    val_idx = np.array(val_idx)

    assert len(labeled_idx) == args.num_labeled * args.num_id_classes
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)

    # if args.ood is not None: # 如果包含OOD，那么把其他ood类加上
    unlabeled_idx = np.array(range(len(labels)))

        
    unlabeled_idx = [idx for idx in unlabeled_idx if idx not in labeled_idx]
    unlabeled_idx = [idx for idx in unlabeled_idx if idx not in val_idx]
    print("labeled" , len(labeled_idx))
    print("unlabeled" , len(unlabeled_idx))
    print("val",len(val_idx))
    return labeled_idx, unlabeled_idx, val_idx 

DATASET_GETTERS = {'cifar100': get_cifar100}
