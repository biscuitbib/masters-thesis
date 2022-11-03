
from thesisproject.data.image_pair_dataset import ImagePairDataset
from thesisproject.data.image_queue import ImageQueue
from thesisproject.data.slice_loader import SliceLoader
from thesisproject.data.loading_pool import LoadingPool


def get_slice_loaders(path, volume_transform=None):
    train_data = ImagePairDataset(path + "train", image_transform=volume_transform, label_transform=volume_transform)
    #train_queue = ImageQueue(train_data)
    train_loader = SliceLoader(train_data, slices_per_epoch=2000)

    val_data = ImagePairDataset(path + "val", image_transform=volume_transform, label_transform=volume_transform)
    #val_queue = ImageQueue(val_data)
    val_loader = SliceLoader(val_data, slices_per_epoch=1000)

    return train_loader, val_loader