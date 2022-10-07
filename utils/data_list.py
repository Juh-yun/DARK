import numpy as np
import os
import os.path
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]

    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)

    image_index = image_index[selected_list]
    # print(image_index)
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class Imagelists(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False, dict_path2img=None):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)

class Imagelists_aug(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, transform2=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.transform2 = transform2
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img1 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform2 is not None:
            img2 = self.transform2(img)
            return img1, img2, target

        if not self.test:
            return img1, target
        else:
            return img1, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


class Imagelists_unlabeled_test(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


class Imagelists_labeled(object):
    def __init__(self, image_list, label_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False):

        imgs, labels = image_list, label_list

        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


class Imagelists_unlabeled(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, transform2=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.transform2 = transform2
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img1 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform2 is not None:
            img2 = self.transform2(img)
            return img1, img2, target

        if not self.test:
            return img1, target
        else:
            return img1, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)
