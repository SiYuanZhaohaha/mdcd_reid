import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset, PairImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler_IdUniform
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .mm import MM
from .mixed_market1501 import Mixed_Market1501
from .mixed_msmt17 import Mixed_MSMT17
from .hazy_market1501 import Hazy_Market1501
from .multiple_market1501 import Multiple_Market1501
from .hhazy_market1501 import HHazy_Market1501
from .rrain_market1501 import RRain_Market1501
from .lt_market1501 import LT_Market1501_H, LT_Market1501_M, LT_Market1501_L
import torch.multiprocessing as mp

__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'mm': MM,
    "mixed_msmt17":Mixed_MSMT17,
    "mixed_market1501":Mixed_Market1501,
    "hazy_market1501":Hazy_Market1501,
    "mutltiple_market1501":Multiple_Market1501,
    "hhazy_market1501": HHazy_Market1501,
    "rrain_market1501": RRain_Market1501,
    'lt_market1501_h': LT_Market1501_H,
    'lt_market1501_m': LT_Market1501_M,
    'lt_market1501_l': LT_Market1501_L,

}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def corrupt_train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, imgs_f,pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), torch.stack(imgs_f, dim=0), pids, camids, viewids

def corrupt_val_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, imgs_f,pids, camids, viewids , img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
   
    return torch.stack(imgs, dim=0), torch.stack(imgs_f, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.NAMES == 'ourapi':
        dataset = OURAPI(root_train=cfg.DATASETS.ROOT_TRAIN_DIR, root_val=cfg.DATASETS.ROOT_VAL_DIR, config=cfg)
    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'img_triplet']:
        print('using img_triplet sampler')
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER in ['id_triplet', 'id']:
        print('using ID sampler')
        train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler_IdUniform(dataset.train, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn, drop_last = True,
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num


from .restoration import ImageRestorationTransform
def make_dataloader_new(cfg):
    mp.set_start_method('spawn', force=True)
    #dehazy = ImageRestorationTransform()
    train_transforms = T.Compose([ 
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    """ val_transforms_dehazy = T.Compose([
        dehazy,
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ]) """
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = PairImageDataset(dataset.train, dataset.c_train,train_transforms)
    
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=corrupt_train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=corrupt_train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=corrupt_train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    # val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    query_set = ImageDataset(dataset.query, val_transforms)
    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    corrupted_query_set = ImageDataset(dataset.c_query, val_transforms)
    corrupted_gallery_set = ImageDataset(dataset.c_gallery, val_transforms)

    val_set = torch.utils.data.ConcatDataset([query_set, gallery_set])
    corrupted_val_set = torch.utils.data.ConcatDataset([corrupted_query_set, corrupted_gallery_set])

    train_val_set = PairImageDataset(dataset.query + dataset.gallery, dataset.c_query+dataset.c_gallery,val_transforms)

    corrupted_query_set = torch.utils.data.ConcatDataset([corrupted_query_set, gallery_set])
    corrupted_gallery_set = torch.utils.data.ConcatDataset([query_set, corrupted_gallery_set])
    batch_size=cfg.TEST.IMS_PER_BATCH

    train_val_loader = DataLoader(
        train_val_set, batch_size=64, shuffle=False, num_workers=num_workers,
        collate_fn=corrupt_val_collate_fn
    )

    val_loader = DataLoader(
        val_set, batch_size=64, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    corrupted_val_loader = DataLoader(corrupted_val_set,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      collate_fn=val_collate_fn)
    corrupted_query_loader = DataLoader(corrupted_query_set,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        collate_fn=val_collate_fn)
    corrupted_gallery_loader = DataLoader(corrupted_gallery_set,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          collate_fn=val_collate_fn)
    
    # return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
    return train_loader, train_val_loader, val_loader, corrupted_val_loader, corrupted_query_loader, corrupted_gallery_loader, len(
        dataset.query), num_classes, cam_num, view_num
