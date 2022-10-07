import os
import torch
from torchvision import transforms
from .data_list import return_classlist
import pickle
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from PIL import Image
import random
from .data_list import Imagelists_labeled, Imagelists_unlabeled, Imagelists_unlabeled_test, return_classlist, make_dataset_fromlist

from .randaugment import RandAugmentMC

def get_color_distortion(s=1.0):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

def load_pickle():
    with open('dict_path2img.pickle', 'rb') as config_dictionary_file:
        dict_path2img = pickle.load(config_dictionary_file)

    return dict_path2img


class RandomIdentitySampler_alignedreid(Sampler):
    def __init__(self, num_of_class, source_label, num_per_class_src):

        self.num_instances = num_per_class_src
        self.num_pids_per_batch = num_of_class

        self.index_dic = defaultdict(list)
        for index, pid in enumerate(source_label):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def return_dataset_DARK(args):

    base_path = './data/txt/%s' % args.dataset
    root = args.root
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % args.num)
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % args.num)

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    # torchvision.transforms.RandomResizedCrop
    data_transforms = {
        'label': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'str': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(crop_size),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'weak': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    src_imgs, src_labels = make_dataset_fromlist(image_set_file_s)
    trg_train_imgs, trg_train_labels = make_dataset_fromlist(image_set_file_t)

    len_source = len(src_imgs)
    len_target = len(trg_train_imgs)
    print("len_source: %d" % len_source)
    print("len_target: %d" % len_target)

    # breakpoint()

    labeled_imgs = np.concatenate((src_imgs, trg_train_imgs))
    labels = np.concatenate((src_labels, trg_train_labels))

    labeled_dataset = Imagelists_labeled(labeled_imgs, labels, root=root, transform=data_transforms['label'])
    target_dataset_val = Imagelists_unlabeled_test(image_set_file_t_val, root=root, transform=data_transforms['test'])
    target_dataset_test = Imagelists_unlabeled_test(image_set_file_unl, root=root, transform=data_transforms['test'])
    target_dataset_unl = Imagelists_unlabeled(image_set_file_unl, root=root, transform=data_transforms['weak'], transform2=data_transforms['str'])

    class_list = return_classlist(image_set_file_s)

    n_class = len(class_list)
    print("%d classes in this dataset" % n_class)

    bs = args.ways * args.src_shots
    nw = 3

    labeled_data_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.ways * (args.src_shots + args.trg_shots),
                                                      num_workers=nw, shuffle=False, drop_last=True, sampler=RandomIdentitySampler(num_of_class=args.ways,
                                                                                                                                   source_label=src_labels, target_label=trg_train_labels,
                                                                                                                                   num_per_class_src=args.src_shots, num_per_class_trg=args.trg_shots,
                                                                                                                                   ways=args.ways))
    target_loader_val = torch.utils.data.DataLoader(target_dataset_val, batch_size=min(bs, len(target_dataset_val)),
                                                    num_workers=nw, shuffle=True, drop_last=True)

    target_loader_test = torch.utils.data.DataLoader(target_dataset_test, batch_size=bs, num_workers=nw,
                                                     shuffle=True, drop_last=False)

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                                    batch_size=bs, num_workers=nw, shuffle=True, drop_last=True)

    return labeled_data_loader, target_loader_val, target_loader_test, target_loader_unl, class_list


def read_target_split(image_list):
    with open(image_list) as f:
        image_index = [int(x) for x in f.readlines()]
    return image_index


class RandomIdentitySampler(Sampler):
    def __init__(self, num_of_class, source_label, target_label, num_per_class_src, num_per_class_trg, ways):

        self.index_dic_src = defaultdict(list)
        for index, pid in enumerate(source_label):
            self.index_dic_src[pid].append(index)
        num_of_all_source = len(source_label)

        self.index_dic_trg = defaultdict(list)
        for index, pid in enumerate(target_label):
            self.index_dic_trg[pid].append(index + num_of_all_source)

        self.num_per_class_src = num_per_class_src
        self.num_per_class_trg = num_per_class_trg
        self.num_of_class = num_of_class
        self.classes = list(self.index_dic_src.keys())
        self.num_identities = len(self.classes)
        self.ways = ways

    def __iter__(self):

        class_list = list(range(self.num_identities))
        random.shuffle(class_list)

        ret = []
        for j in class_list:
            src_pid = self.index_dic_src[j]
            trg_pid = self.index_dic_trg[j]

            replace1 = False if len(trg_pid) >= self.num_per_class_trg else True
            replace2 = False if len(src_pid) >= self.num_per_class_src else True

            src_t = np.random.choice(src_pid, size=self.num_per_class_src, replace=replace2)
            trg_t = np.random.choice(trg_pid, size=self.num_per_class_trg, replace=replace1)

            ret.extend(src_t)
            ret.extend(trg_t)

        return iter(ret)

    def __len__(self):

        return self.num_identities * (self.num_per_class_trg + self.num_per_class_src)
