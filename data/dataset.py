import torchvision


# train_dataset_path = {
#         'miniImageNet': 'dataset/miniImagenet/base',
#         'tieredImageNet': 'dataset/tieredImageNet/base',
#         'CIFAR-FS': 'dataset/cifar100/base',
#         'FC100': 'dataset/FC100_hd/base',
#     }
#
# val_dataset_path = {
#         'miniImageNet': 'dataset/miniImagenet/val',
#         'tieredImageNet': 'dataset/tieredImageNet/val',
#         'CIFAR-FS': 'dataset/cifar100/val',
#         'FC100': 'dataset/FC100_hd/val',
#     }
#
# test_dataset_path = {
#         'miniImageNet': 'dataset/miniImagenet/novel',
#         'tieredImageNet': 'dataset/tieredImageNet/novel',
#         'CIFAR-FS': 'dataset/cifar100/novel',
#         'FC100': 'dataset/FC100_hd/novel',
#     }
#
# dataset_path = {
#         'miniImageNet': ['dataset/miniImagenet/base', 'dataset/miniImagenet/val', 'dataset/miniImagenet/novel'],
#         'tieredImageNet': ['dataset/tieredImageNet/base', 'dataset/tieredImageNet/val', 'dataset/tieredImageNet/novel'],
#         'CIFAR-FS': ['dataset/cifar100/base', 'dataset/cifar100/val', 'dataset/cifar100/novel'],
#         'FC100': ['dataset/FC100_hd/base', 'dataset/FC100_hd/val', 'dataset/FC100_hd/novel'],
# }

train_dataset_path = {
        'miniImageNet': '/HDD/liuyuanyuan/mini_imagenet/miniImagenet/base',#../dataset/miniImagenet/base
        'tieredImageNet': '/HDD/liuyuanyuan/tired/NewFolder/train',
        'CUB': '/HDD/liuyuanyuan/CUB_FSL/train',
        'CIFAR-FS': '/HDD/liuyuanyuan/cifar100/base',
        'FC100': '/HDD/liuyuanyuan/FC100_hd/base',
        'FC100_hd': '/HDD/liuyuanyuan/FC100_hd/base',
        'CropDisease': '../dataset/CD-FSL/CropDisease/base'
    }

val_dataset_path = {
        'miniImageNet': '/HDD/liuyuanyuan/mini_imagenet/miniImagenet/val',#../dataset/miniImagenet/val
        'tieredImageNet': '/HDD/liuyuanyuan/tired/val',
        'CUB': '/HDD/liuyuanyuan/CUB_FSL/val',
        'CIFAR-FS': '/HDD/liuyuanyuan/cifar100/val',
        'FC100': '/HDD/liuyuanyuan/FC100_hd/val',
        'CropDisease': '../dataset/CD-FSL/CropDisease/val'
    }

test_dataset_path = {
        'miniImageNet': '/HDD/liuyuanyuan/mini_imagenet/miniImagenet/novel',#../dataset/miniImagenet/novel
        'tieredImageNet': '/HDD/liuyuanyuan/tired/test',
        'CUB': '/HDD/liuyuanyuan/CUB_FSL/test',
        'CIFAR-FS': '/HDD/liuyuanyuan/cifar100/novel',
        'FC100': '/HDD/liuyuanyuan/FC100_hd/novel',
        'FC100_hd': '/HDD/liuyuanyuan/FC100_hd/novel',
        'EuroSAT': '../dataset/CD-FSL/EuroSAT/2750',
        'CropDisease': '../dataset/CD-FSL/CropDisease/novel'
    }

dataset_path = {
        'miniImageNet': ['../dataset/miniImagenet/base', '../dataset/miniImagenet/val', '../dataset/miniImagenet/novel'],
        'tieredImageNet': ['../dataset/tieredImageNet/base', '../dataset/tieredImageNet/val', '../dataset/tieredImageNet/novel'],
        'CUB': ['../dataset/CUB/base', '../dataset/CUB/val', '../dataset/CUB/novel'],
        'CIFAR-FS': ['../dataset/cifar100/base', '../dataset/cifar100/val', '../dataset/cifar100/novel'],
        'FC100': ['../dataset/FC100/base', '../dataset/FC100/val', '../dataset/FC100/novel'],
        'CropDisease': ['../dataset/CD-FSL/CropDisease/train'],
        'EuroSAT': ['../dataset/CD-FSL/EuroSAT/2750']
}


class DatasetWithTextLabel(object):
    def __init__(self, dataset_name, aug, split='test'):
        self.dataset_name = dataset_name
        if split == 'train':
            dataset_path = train_dataset_path[dataset_name]
        elif split == 'val':
            dataset_path = val_dataset_path[dataset_name]
        elif split == 'test':
            dataset_path = test_dataset_path[dataset_name]
        self.dataset = torchvision.datasets.ImageFolder(dataset_path, aug)
        self.idx2text = {}
        self.idx2att={}
        if dataset_name == 'miniImageNet' or dataset_name == 'tieredImageNet':
            with open('data/ImageNet_idx2text.txt', 'r') as f:
                for line in f.readlines():
                    idx, _, text = line.strip().split()
                    text = text.replace('_', ' ')
                    self.idx2text[idx] = text
        elif dataset_name == 'FC100':
            with open('data/cifar100_idx2text.txt', 'r') as f:
                for line in f.readlines():
                    idx, text = line.strip().split()# 删除字符串两端的空格,将字符串按照空格进行分割，生成一个列表
                    idx = idx.strip(':')# # 移除头尾通过.进行切片
                    text = text.replace('_', ' ')
                    self.idx2text[idx] = text# idx2text[0]=apple
        elif dataset_name == 'CIFAR-FS':
            for idx in self.dataset.classes:
                text = idx.replace('_', ' ')
                self.idx2text[idx] = text
        elif dataset_name == 'CUB':
            # for idx in self.dataset.classes:
            #     id,_ = idx.split('.')
            #     self.idx2text[id] = text
            with open('data/attributes_names.txt', 'r') as f:
                for line in f.readlines():
                    idx, text = line.strip().split()  # 删除字符串两端的空格,将字符串按照空格进行分割，生成一个列表 002   has_bill_shape::dagger
                    textb = text.replace('_', ' ')  # has_bill_shape::dagger
                    textb = textb.replace('::', ' ')
                    self.idx2att[idx] = textb
                    # print("idx:", idx, "idx2att[idx]=", textb)

            with open('data/classes.txt', 'r') as f:
                for line in f.readlines():
                    idx, text = line.strip().split()# 删除字符串两端的空格,将字符串按照空格进行分割，生成一个列表  002   002.Laysan_Albatross
                    id,textb = text.split('.')#002 Laysan_Albatross
                    textb = textb.replace('_', ' ')#Laysan_Albatross
                    textb+=" "
                    # print("idx",idx)
                    # print("self.idx2att[idx]:",self.idx2att[idx])
                    textb+=self.idx2att[id]
                    self.idx2text[text] = textb  # idx2text[002.Laysan_Albatross]=Laysan Albatross
                    # print("text:",text,"idx2text[text]=",textb)#共200个
        # print("cub_dataset:",self.dataset.classes)


    def __getitem__(self, i):
        image, label = self.dataset[i]
        text = self.dataset.classes[label]
        text = self.idx2text[text]
        # text prompt: A photo of a {label}
        text = 'A photo of a ' + text
        return image, label, text

    def __len__(self):
        return len(self.dataset)
