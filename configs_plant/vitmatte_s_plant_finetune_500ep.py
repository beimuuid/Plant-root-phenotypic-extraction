from .common.train import train
from .common.model import model
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.dataloader import dataloader

train.max_iter = int(174 / 8 / 2 * 500)
train.checkpointer.period = int(174 / 8 / 2 * 10)

optimizer.lr=5e-5
lr_multiplier.scheduler.values=[1.0, 0.1, 0.05]
lr_multiplier.scheduler.milestones=[int(174 / 8 / 2 * 300), int(174 / 8 / 2 * 450)]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

train.init_checkpoint = './output_of_train/vitmatte_s_am2k_100ep_data-aug_trimap/model_final.pth'
train.output_dir = './output_of_train/vitmatte_s_plant_finetune_500ep_data-aug_trimap'

dataloader.train.batch_size=16
dataloader.train.num_workers=4

train.eval_period = 50
