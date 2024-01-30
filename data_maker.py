import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from math import ceil
from PIL import Image

'''
# Codes are built based on
# https://github.com/ucr-optml/AutoBalance and https://github.com/kaidic/LDAM-DRW/tree/master
Reference:
    [1] Mingchen Li, Xuechen Zhang, Christos Thrampoulidis, Jiasi Chen, and Samet Oymak. Autobalance: Optimized
    loss functions for imbalanced data. Advances in Neural Information Processing Systems, 34:3163–3177, 2021.
    [2] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. Learning imbalanced datasets with
    label-distribution-aware margin loss. Advances in neural information processing systems, 32, 2019.
    to compare results in the same data setting
'''

def make_data_loaders(args):
    total_size = args.per_class_size
    val_size = args.per_class_size * args.validation_ratio
    train_size = total_size - val_size
    imbalance_ratio = args.imbalance_ratio
    num_classes = args.num_classes
    num_total_samples=[]
    num_train_samples=[]
    num_val_samples=[]

    train_data_set = args.train_data_set
    test_data_set = args.test_data_set
    train_x, train_y = np.array(train_data_set.data), np.array(train_data_set.targets)

    if args.imbalance_type == "step":
        train_mu = imbalance_ratio
        val_mu = imbalance_ratio
        for cls_idx in range(num_classes // 2):
            num_total_samples.append(int(ceil(total_size * train_mu)))
            num_train_samples.append(int(ceil(train_size * train_mu)))
            num_val_samples.append(int(ceil(val_size * val_mu)))
        for cls_idx in range( num_classes - (num_classes // 2) ):
            num_total_samples.append(int(total_size))
            num_train_samples.append(int(train_size))
            num_val_samples.append(int(val_size))

    elif args.imbalance_type == "LT":
        train_mu = imbalance_ratio ** (1. / (args.num_classes-1.))
        val_mu = imbalance_ratio ** (1. / (args.num_classes-1.))
        for i in range(num_classes):
            num_total_samples.append(int(ceil(total_size * (train_mu ** i))))
            num_train_samples.append(int(ceil(train_size * (train_mu ** i))))
            num_val_samples.append(int(ceil(val_size * (val_mu ** i))))

    else:
        raise ValueError("imbalance type does not match")

    train_index = []
    val_index = []

    for i in range(num_classes):
        train_index.extend(np.where(train_y == i)[0][:num_train_samples[i]])
        val_index.extend(np.where(train_y == i)[0][-num_val_samples[i]:])

    total_index=[]
    total_index.extend(train_index)
    total_index.extend(val_index)
    total_index=list(set(total_index))
    random.shuffle(total_index)
    train_x, train_y= train_x[total_index], train_y[total_index]

    # To make different train/val split from time to time

    train_index = []
    val_index = []
    # print(train_x,train_y)
    print("-----------------------------------")
    print("num_total_samples, num_train_samples, num_val_samples")
    print(num_total_samples, num_train_samples, num_val_samples)
    print("-----------------------------------")

    class_sample_num_dict = {}
    class_sample_num_dict["num_total_samples"] = num_total_samples
    class_sample_num_dict["num_train_samples"] = num_train_samples
    class_sample_num_dict["num_val_samples"] = num_val_samples


    for i in range(num_classes):
        train_index.extend(np.where(train_y == i)[0][:num_train_samples[i]])
        val_index.extend(np.where(train_y == i)[0][-num_val_samples[i]:])

    random.shuffle(train_index)
    random.shuffle(val_index)

    train_data, train_targets = train_x[train_index], train_y[train_index]
    val_data, val_targets = train_x[val_index], train_y[val_index]
    total_data, total_targets = train_x, train_y

    # make custom data set
    train_transform = args.transform_train
    test_transform = args.transform_test

    total_dataset = CustomDataset(total_data, total_targets, train_transform)
    train_dataset = CustomDataset(train_data,train_targets,train_transform)
    val_dataset = CustomDataset(val_data,val_targets,train_transform)

    total_eval_dataset = CustomDataset(total_data, total_targets, test_transform)
    train_eval_dataset = CustomDataset(train_data,train_targets,test_transform)
    val_eval_dataset = CustomDataset(val_data,val_targets,test_transform)

    # make data_loader
    batch_size = args.batch_size
    num_workers = args.num_workers

    total_loader = DataLoader(total_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=False, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=True, drop_last=False, pin_memory=True)


    eval_total_loader = DataLoader(total_eval_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=False, drop_last=False, pin_memory=True)
    eval_train_loader = DataLoader(train_eval_dataset, batch_size=batch_size, num_workers=num_workers,
                                   shuffle=False, drop_last=False, pin_memory=True)
    eval_val_loader = DataLoader(val_eval_dataset, batch_size=batch_size, num_workers=num_workers,
                                 shuffle=False, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=False, pin_memory=True)

    data_load_dict = {}
    data_load_dict["total_loader"] = total_loader
    data_load_dict["train_loader"] = train_loader
    data_load_dict["val_loader"] = val_loader

    data_load_dict["eval_total_loader"] = eval_total_loader
    data_load_dict["eval_train_loader"] = eval_train_loader
    data_load_dict["eval_val_loader"] = eval_val_loader
    data_load_dict["test_loader"] = test_loader

    return data_load_dict, class_sample_num_dict



class CustomDataset(Dataset):
    """CustomDataset with support of transforms.
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    def __len__(self):
        return len(self.data)


