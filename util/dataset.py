from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import config as conf




def load_data(conf, training=True):  #필요한 데이터 로드 후 토치에 사용할 수 있는 형태로 만들어줌
    if training:
        dataroot = conf.train_root
        transform = conf.train_transform
        batch_size = conf.train_batch_size
    else:
        dataroot = conf.test_root
        transform = conf.test_transform
        batch_size = conf.test_batch_size

    data = ImageFolder(dataroot, transform=transform)
    class_num = len(data.classes)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, 
        pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader, class_num