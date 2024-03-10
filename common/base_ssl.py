import os
import os.path as osp
import math
import time
import glob
import abc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from common.timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from common.my_vq_model_ssl2 import get_model
from dataloader.dataset import MultipleDatasets
from common.utils.augmentation import GaussianBlur,Solarization
# dataset_list = ['CrowdPose', 'Human36M', 'MSCOCO', 'MuCo', 'PW3D']
dataset_list = ['CrowdPose_SSL', 'Human36M_SSL', 'My_MuCo_SSL', 'My_MSCOCO_SSL', 'PW3D_SSL']
for i in range(len(dataset_list)):
    exec('from dataloader.' + dataset_list[i] + ' import ' + dataset_list[i])

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        if dist.get_rank() == 0:
            self.logger = colorlogger(cfg, log_name=log_name)
        else:
            self.logger = None

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class DataAugmentationDINO(object):
    def __init__(self):
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.ToPILImage(),
            color_jitter,
            GaussianBlur(1.0),
            Solarization(0.2),
            normalize,
        ])
        # second global crop
        # self.global_transfo2 = transforms.Compose([
        #     color_jitter,
        #     GaussianBlur(0.1),
        #     Solarization(0.2),
        #     normalize,
        # ])
        # # transformation for the local small crops
        # self.local_crops_number = local_crops_number
        # self.local_transfo = transforms.Compose([
        #     color_jitter,
        #     GaussianBlur(p=0.5),
        #     normalize,
        # ])

    def __call__(self, image):
        image = self.global_transfo1(image)
        # crops.append(self.global_transfo2(image))
        # crops.append(self.local_transfo(image))
        return image

class Trainer_SSL(Base):
    def __init__(self, cfg):
        self.cfg = cfg
        super(Trainer_SSL, self).__init__(cfg.log_dir, log_name='train_logs.txt')

    def get_optimizer(self, model):
        base_params = list(map(id, model.module.backbone.parameters()))
        other_params = filter(lambda p: id(p) not in base_params, model.module.parameters())
        optimizer = torch.optim.AdamW([
            {'params': model.module.backbone.parameters(), 'lr': self.cfg.lr_backbone},
            {'params': other_params, },
            {'params': self.awl.parameters(), 'weight_decay': 0}
        ],
        lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(self.cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        if dist.get_rank() == 0:
            self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        # model_file_list = glob.glob(osp.join(self.cfg.model_dir,'*.pth.tar'))
        # cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        cur_epoch = 0
        ckpt_path = osp.join(self.cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path, map_location='cpu') 
        # start_epoch = ckpt['epoch'] + 1
        backbonedict = {}
        # for k,v in ckpt['network'].items():
        #     if "vqvae" not in k: # and "cascade_pose_out" not in k:
        #         backbonedict[k]=v
        #     if "backbone" not in k and "pose2feat" not in k and "down_linear" not in k and "conv2d_to_3d" not in k:
        #         backbonedict[k]=v
        # ckpt = backbonedict
        # ckpt.pop("module.spose_shape_cam_param",None)

        try:
            ckpt = ckpt['network']
        except KeyError:
            pass
        
        for k,v in ckpt.items():
            if "vqvae" not in k: # and "cascade_pose_out" not in k:
                backbonedict[k]=v
        ckpt = backbonedict
        # ckpt.pop("token_mlp.weight",None)
        # ckpt.pop("token_mlp.bias",None)
        # ckpt.pop("decoder_token_mlp.weight",None)
        # ckpt.pop("module.spose_shape_cam_param",None)
        start_epoch = 1
        infoa,infob = model.load_state_dict(ckpt, strict=False)
        print("missing",infoa)
        print("unexpected",infob)
        # self.logger.info(info)
        if cur_epoch != 0:
            self.awl.load_state_dict(ckpt['awl'])
        if dist.get_rank() == 0:
            self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model.cuda(), optimizer

    def set_lr(self, epoch):
        for e in self.cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < self.cfg.lr_dec_epoch[-1]:
            idx = self.cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg.lr / (self.cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg.lr / (self.cfg.lr_dec_factor ** len(self.cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        if dist.get_rank() == 0:
            self.logger.info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(self.cfg.trainset_3d)):
            trainset3d_loader.append(eval(self.cfg.trainset_3d[i])(DataAugmentationDINO(), "train"))
        trainset2d_loader = []
        for i in range(len(self.cfg.trainset_2d)):
            trainset2d_loader.append(eval(self.cfg.trainset_2d[i])(DataAugmentationDINO(), "train"))

        if len(trainset3d_loader) > 0 and len(trainset2d_loader) > 0:
            self.vertex_num = trainset3d_loader[0].vertex_num
            self.joint_num = trainset3d_loader[0].joint_num
            trainset3d_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
            trainset2d_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
            trainset_loader = MultipleDatasets([trainset3d_loader, trainset2d_loader], make_same_len=True)
        elif len(trainset3d_loader) > 0:
            self.vertex_num = trainset3d_loader[0].vertex_num
            self.joint_num = trainset3d_loader[0].joint_num
            trainset_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
        elif len(trainset2d_loader) > 0:
            self.vertex_num = trainset2d_loader[0].vertex_num
            self.joint_num = trainset2d_loader[0].joint_num
            trainset_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
        else:
            assert 0, "Both 3D training set and 2D training set have zero length."
            
        self.itr_per_epoch = math.ceil(len(trainset_loader) / self.cfg.num_gpus / self.cfg.train_batch_size)
        if self.cfg.distributed:
            self.sampler = DistributedSampler(trainset_loader)
        else:
            self.sampler = None
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=self.cfg.train_batch_size, 
                shuffle=(self.sampler is None), num_workers=self.cfg.num_thread, pin_memory=True,collate_fn=self._collate_fn,
                sampler=self.sampler)
                
    def _collate_fn(self,batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
        

    def _make_model(self):
        # prepare network
        if dist.get_rank() == 0:
            self.logger.info("Creating graph and optimizer...")
        if not hasattr(self, 'joint_num'):
            self.joint_num = 30
        teacher = get_model(self.joint_num, 'train', self.cfg)
        student = get_model(self.joint_num, 'train', self.cfg)
        awl = AutomaticWeightedLoss(9)
 
        if self.cfg.distributed:
            teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher.cuda())
            student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student.cuda())
            if self.cfg.is_local:
                teacher = DDP(teacher, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=True)
                student = DDP(student, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=True)
                awl = DDP(awl.cuda(), device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=True)
            else:
                teacher = DDP(teacher, find_unused_parameters=True)
                student = DDP(student, find_unused_parameters=True)
                awl = DDP(awl.cuda(), find_unused_parameters=True)
        else:
            teacher = DataParallel(teacher).cuda()
            student = DataParallel(student).cuda()
        self.awl = awl
        optimizer = self.get_optimizer(student)
        if self.cfg.continue_train:
            start_epoch, student, optimizer = self.load_model(student, optimizer)
        else:
            start_epoch = 0
        teacher.load_state_dict(student.state_dict())
        for tparams in teacher.parameters():
            tparams.requires_grad = False
        
        student.train()

        self.start_epoch = start_epoch
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer

    # def _init_ddp(self):
    #     torch.cuda.set_device(f'cuda:{self.cfg.local_rank}')
    #     dist.init_process_group(backend='nccl')
    #     assert dist.is_initialized(), "Distributed backend not initialized."





class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=21):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True, dtype=torch.float32)
        self.params = nn.Parameter(params)

    def forward(self, loss_dict):
        if not hasattr(self, 'keys'):
            self.keys = sorted(list(loss_dict.keys()))
        for i, key in enumerate(self.keys):
            loss_dict[key] = 0.5 / (self.params[i] ** 2) * loss_dict[key] + torch.log(1 + self.params[i] ** 2)
        return loss_dict