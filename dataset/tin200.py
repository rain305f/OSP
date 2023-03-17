import os
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

import numpy as np
import cv2
from torchvision import transforms
from .randaugment import RandAugmentMC


def load_tiny_imagenet(root):
    labels_dict = {} # (key: str : value: int ): 将label (n02124075)转换成 0-200的类别标签
    class_dict = {} # (key: int  : value: str )
    train_names = [] # train set中所有image的名称

    val_labels = []
    val_names = []

    # labels_dict： str-> int
    with open(root+'/wnids.txt') as wnid:
        i = 0 
        for line in wnid:
            # labels_t.append(line.strip('\n'))
            labels_dict[line.strip('\n')] = i 
            class_dict[i] = line.strip('\n')
            i = i + 1 

    
    # read train set data 
    for label in labels_dict.keys():
        txt_path = root+ '/train/'+label+'/'+label+'_boxes.txt'
        image_name = []
        with open(txt_path) as txt:
            for line in txt:
                image_name.append(line.strip('\n').split('\t')[0])
        train_names.append(image_name)

    # read val set data 
    with open(root+'val/val_annotations.txt') as txt:
        for line in txt:
            if labels_dict[line.strip('\n').split('\t')[1]] < 100 : # val也是id类
                val_names.append(line.strip('\n').split('\t')[0])
                val_labels.append(labels_dict[line.strip('\n').split('\t')[1]])

    return train_names , val_names , val_labels , class_dict


def split_x_u_ood(tot_class,train_images,labeled_num_per_class , unlabeled_num_per_class, ood_num_per_class):
    ood_idxs = []
    labeled_idxs = []
    unlabeled_idxs = []
    for i in range(tot_class):
        image_idx = np.random.choice(train_images[i] , labeled_num_per_class + unlabeled_num_per_class  , False)  # list of str,图片的名称
        labeled_idxs.append(image_idx[:labeled_num_per_class ])
        unlabeled_idxs.append(image_idx[labeled_num_per_class:])
        
    for i in range(tot_class,200):
        image_idx = np.random.choice(train_images[i] ,ood_num_per_class  , False)  
        ood_idxs.append(image_idx)
    return ood_idxs , labeled_idxs, unlabeled_idxs
        

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

tin200_mean = (0.480245, 0.44807, 0.397548) 
tin200_std = (0.27172, 0.265269, 0.273970)

def get_tin200( args, root ):
    tot_class = args.tot_class 
    labeled_num_per_class = args.n_labels_per_cls
    n_unlabels = args.n_unlabels 
    unlabeled_num_per_class = int(n_unlabels * (1.0 - args.ratio)) // tot_class 
    ood_num_per_class = int(n_unlabels * args.ratio) //(200-tot_class)
    
    print(labeled_num_per_class,"labeled_num_per_class" )
    print( unlabeled_num_per_class," unlabeled_num_per_class")
    print( ood_num_per_class," ood_num_per_class")
    
    
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=tin200_mean, std=tin200_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=tin200_mean, std=tin200_std)
    ])


    train_images  , val_images , val_labels , class_dict = load_tiny_imagenet(root)
    
#     print("val_labels", np.unique(val_labels) )
    
    ood_idxs , labeled_idxs, unlabeled_idxs = split_x_u_ood(tot_class,train_images,labeled_num_per_class , unlabeled_num_per_class, ood_num_per_class)
    
    # labeled data
    train_labeled_dataset = TIN('labeled', root, tot_class = tot_class,image_idx= labeled_idxs, class_dict = class_dict, transform=transform_labeled)
    
    # ood and in unlabeled data 
    train_unlabeled_dataset = TIN('unlabeled',root, tot_class = tot_class,image_idx= unlabeled_idxs , ood_idx= ood_idxs,class_dict = class_dict, transform=TransformFixMatch(mean=tin200_mean, std=tin200_std))
    
    val_dataset = TIN('val',root,tot_class = tot_class ,image_idx= val_images,class_dict = class_dict, transform=transform_val,val_labels=val_labels)

    print(len(train_labeled_dataset),"labeled_num_per_class" )
    print( len(train_unlabeled_dataset)," unlabeled_num_per_class")
    print( len( val_dataset) ,"  val_dataset")
    
    
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset


class TIN(Dataset):
    def __init__(self, type, root, tot_class, image_idx, class_dict,ood_idx=None, transform=None,  target_transform=None, val_labels = None , return_idx=True):
        self.type = type
        self.class_dict = class_dict # (key: int  : value: str )
        if type == 'labeled': # ood ,labeled_data , unlabeled_data
            self.images = []
            self.labels = []

            for i in range(0, tot_class): # 从id 类里选择labeled_num_per_class 个样本
                for image_name in image_idx[i]:
                    image_path = os.path.join(root+'/train', self.class_dict[i] , 'images', image_name) 
                    self.images.append(cv2.imread(image_path))
                    self.labels.append(i)
            self.labels = np.array(self.labels)
            self.images = np.array(self.images)

        elif type == 'unlabeled':
            assert ood_idx != None
            self.images = []
            self.labels = []
            for i in range(0, tot_class): # 从id 类里选择labeled_num_per_class 个样本
                for image_name in image_idx[i]:
                    image_path = os.path.join(root+'/train', self.class_dict[i] , 'images', image_name) 
                    self.images.append(cv2.imread(image_path))
                    self.labels.append(i)

            # ood 
            for i in range(tot_class,200):
                 for image_name in ood_idx[i-tot_class]:
                    image_path = os.path.join(root+'/train', self.class_dict[i] , 'images', image_name) 
                    self.images.append(cv2.imread(image_path))
                    self.labels.append(i)               

            self.labels = np.array(self.labels)
#             print("unlabeled: ", np.unique(self.labels))
            self.images = np.array(self.images)        

        elif type == 'val':
            assert val_labels != None
            self.val_images = []
            for val_image in image_idx:
                val_image_path = os.path.join(root+'/val' , 'images', val_image ) 
                self.val_images.append(cv2.imread(val_image_path))
            self.images = np.array(self.val_images)
            self.labels = np.array(val_labels)

        self.return_idx = return_idx  
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.fromarray(img)
        target = self.labels[index] 
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.return_idx:
            return img, target
        else:
            return img, target, index


    def __len__(self): 
        return self.images.shape[0]

DATASET_GETTERS = {'TIN': get_tin200}