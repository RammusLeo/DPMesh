import argparse
import os
import torch
import torch.distributed as dist
import sys
from config import cfg
from common.base import Trainer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader.VideoMulti import VideoMulti
import sys
import numpy as np
import random
sys.path.insert(0, os.path.join(cfg.root_dir, 'common'))
from common.utils.dir import make_folder

def setup_seed(seed=42):
    seed += dist.get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = False  
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.enabled = True

best_dict = {
    'demo': {
        'best_MPJPE': 1e10,
    }
}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--resume_ckpt', type=str, default='', help='for resuming train')
    parser.add_argument('--amp', dest='use_mixed_precision', action='store_true', help='use automatic mixed precision training')
    parser.add_argument('--init_scale', type=float, default=1024., help='initial loss scale')
    parser.add_argument('--cfg', type=str, default='', help='experiment configure file name')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--exp_id', type=str, default='debug', help='experiment configure file name')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--with_contrastive', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-4, help='learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--inter_weight', type=float, default=0.1)
    parser.add_argument('--intra_weight', type=float, default=0.1)
    parser.add_argument('--total_steps', type=int, default=1e10)
    parser.add_argument('--vqtokens', default=24)
    parser.add_argument('--vqclasses', default=256)
    parser.add_argument('--save_folder', default='output')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.distributed:
        torch.cuda.set_device(args.local_rank) 
        dist.init_process_group(backend='nccl', init_method='env://')
        assert dist.is_initialized(), "distributed is not initialized"
    if dist.get_rank() == 0:
        make_folder(cfg.model_dir)
        make_folder(cfg.vis_dir)
        make_folder(cfg.log_dir)
        make_folder(cfg.result_dir)
        dirs = [cfg.model_dir, cfg.vis_dir, cfg.log_dir, cfg.result_dir]
    else:
        dirs = [None, None, None, None]
    dist.broadcast_object_list(dirs, src=0)
    cfg.model_dir, cfg.vis_dir, cfg.log_dir, cfg.result_dir = dirs
    setup_seed()
    if dist.get_rank() == 0:
        cfg.set_args(args.continue_train, resume_ckpt=args.resume_ckpt)
    if args.cfg:
        yml_cfg = cfg.update(args)
    trainer = Trainer(cfg)
    trainer._make_model()
    trainer.model.eval()
    img_list = [name[:-4] for name in os.listdir(os.path.join(cfg.root_dir, 'demo', cfg.renderimg, "annotations"))]
    for img_name in img_list:

        testset_loader = VideoMulti(transforms.ToTensor(),img_name=img_name)
        if cfg.distributed:
            testset_sampler = torch.utils.data.distributed.DistributedSampler(testset_loader,shuffle=False)
        else:
            testset_sampler = None
        test_batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.test_batch_size, 
                    shuffle=False, num_workers=cfg.num_thread, pin_memory=True,
                    sampler=testset_sampler
                    )
        eval(0, trainer, img_name, testset_loader, test_batch_generator)
        

def eval(epoch, trainer, dataset_name, testset_loader, test_batch_generator):
    trainer.model.eval()
    outlist = []
    print("start eval")
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(test_batch_generator)):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}
        with torch.no_grad():
            out = trainer.model(inputs, targets, meta_info, 'test')
        out = {k: v.cpu().numpy() for k,v in out.items()}
        key = list(out.keys())[0]
        batch_size = out[key].shape[0]
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)] # batch_size * dict
        outlist+=out
    testset_loader.evaluate(outlist)



if __name__ == "__main__":
    main()
