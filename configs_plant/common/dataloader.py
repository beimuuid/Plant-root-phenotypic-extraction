from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from detectron2.config import LazyCall as L
from torch.utils.data.distributed import DistributedSampler

from data import ImageFileTrainAM2K, DataGeneratorAM2K

# Dataloader
train_dataset = DataGeneratorAM2K(
    data=ImageFileTrainAM2K(
        trimap_dir="/home/bingxing2/ailab/zhangshufei/Plant-root-phenotypic-extraction/data/plant-root/train/trimap",
        image_dir="/home/bingxing2/ailab/zhangshufei/Plant-root-phenotypic-extraction/data/plant-root/train/original",
        # alpha_ext='.png',
        trimap_ext='.png',
        image_ext='.png',
    ),
    phase='train'
)

dataloader = OmegaConf.create()
dataloader.train = L(DataLoader)(
    dataset=train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=L(DistributedSampler)(
        dataset=train_dataset,
    ),
    drop_last=True
)

