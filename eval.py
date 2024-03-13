import argparse
import os
import torch
import torch.distributed as dist
import sys
from config import cfg
from common.base_v2 import Trainer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader.PW3D import PW3D
# from dataloader.CMU_Panotic import CMU_Panotic
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
    '3dpw': {
        'best_MPJPE': 1e10,
    },
    '3dpw-crowd':{
        'best_MPJPE': 1e10,
    },
    '3dpw-pc':{
        'best_MPJPE': 1e10,
    },
    '3dpw-oc':{
        'best_MPJPE': 1e10,
    },
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
    trainer._make_model(is_eval=True)
    test_dataset_dict = {}
    for dataset_name in best_dict:
        if '3dpw' in dataset_name:
            testset_loader = PW3D(transforms.ToTensor(), data_name=dataset_name)
        if cfg.distributed:
            testset_sampler = torch.utils.data.distributed.DistributedSampler(testset_loader)
        else:
            testset_sampler = None
        test_batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.test_batch_size, 
                    shuffle=False, num_workers=cfg.num_thread, pin_memory=True,
                    sampler=testset_sampler
                    )
        test_dataset_dict[dataset_name] = {
            'loader': test_batch_generator,
            'dataset': testset_loader
        }
    for data_name in best_dict.keys():
        ckpt_path = os.path.join('./checkpoints', '{}_best_ckpt.pth.tar'.format(data_name))
        ckpt = torch.load(ckpt_path, map_location='cpu')
        trainer.model.load_state_dict(ckpt)
        trainer.model.eval()
        eval(0, trainer, data_name, test_dataset_dict[data_name]['dataset'], test_dataset_dict[data_name]['loader'])

def eval(epoch, trainer, dataset_name, testset_loader, test_batch_generator):
    trainer.model.eval()
    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(test_batch_generator)):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                out = trainer.model(inputs, targets, meta_info, 'test')
        out = {k: v.cpu().numpy() for k,v in out.items()}
        key = list(out.keys())[0]
        batch_size = out[key].shape[0]
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)] # batch_size * dict
        # if not dist.is_initialized():
        cur_eval_result = testset_loader.evaluate(out, cur_sample_idx, meta_info) # dict of list
        for k,v in cur_eval_result.items():
            if k in eval_result: 
                eval_result[k] += v
            else: 
                eval_result[k] = v
        cur_sample_idx += len(out)

    mpjpe = torch.tensor(np.mean(eval_result['mpjpe'])).float().cuda().flatten()
    pa_mpjpe = torch.tensor(np.mean(eval_result['pa_mpjpe'])).float().cuda().flatten()
    mpvpe = torch.tensor(np.mean(eval_result['mpvpe'])).float().cuda().flatten()
    samples = torch.tensor(len(eval_result['mpjpe'])).float().cuda().flatten()
    
    dist.barrier()
    gather_list = [torch.zeros_like(mpjpe) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, mpjpe)
    mpjpe_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, pa_mpjpe)
    pa_mpjpe_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, mpvpe)
    mpvpe_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, samples)
    samples_pre_rank = torch.stack(gather_list).flatten()

    all_samples = samples_pre_rank.sum()
    all_mpjpe = mpjpe_pre_rank * samples_pre_rank
    all_pa_mpjpe = pa_mpjpe_pre_rank * samples_pre_rank
    all_mpvpe = mpvpe_pre_rank * samples_pre_rank

    mean_mpjpe = all_mpjpe.sum() / all_samples
    mean_pa_mpjpe = all_pa_mpjpe.sum() / all_samples
    mean_mpvpe = all_mpvpe.sum() / all_samples

    result_dict = {
        'mpjpe': mean_mpjpe.item(),
        'pa_mpjpe': mean_pa_mpjpe.item(),
        'mpvpe': mean_mpvpe.item(),
    }
        
    if dist.get_rank() == 0:
        print('{} {}'.format(dataset_name, epoch))
        for k,v in result_dict.items():
            print(f'{k}: {v:.2f}')
        
        message = [f'{k}: {v:.2f}' for k, v in result_dict.items()]
        trainer.logger.info('{} '.format(dataset_name) + ' '.join(message))         
        
    dist.barrier()

def eval_occ(epoch, trainer, dataset_name, testset_loader, test_batch_generator):
    trainer.model.eval()
    eval_result = {}
    cur_sample_idx = 0
    print("start eval")
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(test_batch_generator)):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                out = trainer.model(inputs, targets, meta_info, 'test')
        out = {k: v.cpu().numpy() for k,v in out.items()}
        key = list(out.keys())[0]
        batch_size = out[key].shape[0]
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)] # batch_size * dict
        # if not dist.is_initialized():
        cur_eval_result = testset_loader.evaluate(out, cur_sample_idx, meta_info) # dict of list
        for k,v in cur_eval_result.items():
            if k in eval_result: 
                eval_result[k] += v
            else: 
                eval_result[k] = v
        cur_sample_idx += len(out)

    mpjpe = torch.tensor(np.mean(eval_result['mpjpe'])).float().cuda().flatten()
    pa_mpjpe = torch.tensor(np.mean(eval_result['pa_mpjpe'])).float().cuda().flatten()
    mpvpe = torch.tensor(np.mean(eval_result['mpvpe'])).float().cuda().flatten()
    samples = torch.tensor(len(eval_result['mpjpe'])).float().cuda().flatten()
    
    mpjpe_visable = torch.tensor(np.mean(eval_result['mpjpe_visable'])).float().cuda().flatten()
    pa_mpjpe_visable = torch.tensor(np.mean(eval_result['pa_mpjpe_visable'])).float().cuda().flatten()
    mpjpe_invisable = torch.tensor(np.mean(eval_result['mpjpe_invisable'])).float().cuda().flatten()
    pa_mpjpe_invisable = torch.tensor(np.mean(eval_result['pa_mpjpe_invisable'])).float().cuda().flatten()
    
    dist.barrier()
    gather_list = [torch.zeros_like(mpjpe) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, mpjpe)
    mpjpe_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, pa_mpjpe)
    pa_mpjpe_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, mpvpe)
    mpvpe_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, samples)
    samples_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, mpjpe_visable)
    mpjpe_visable_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, pa_mpjpe_visable)
    pa_mpjpe_visable_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, mpjpe_invisable)
    mpjpe_invisable_pre_rank = torch.stack(gather_list).flatten()
    dist.all_gather(gather_list, pa_mpjpe_invisable)
    pa_mpjpe_invisable_pre_rank = torch.stack(gather_list).flatten()


    all_samples = samples_pre_rank.sum()
    all_mpjpe = mpjpe_pre_rank * samples_pre_rank
    all_pa_mpjpe = pa_mpjpe_pre_rank * samples_pre_rank
    all_mpvpe = mpvpe_pre_rank * samples_pre_rank

    all_mpjpe_visable = mpjpe_visable_pre_rank * samples_pre_rank
    all_pa_mpjpe_visable = pa_mpjpe_visable_pre_rank * samples_pre_rank
    all_mpjpe_invisable = mpjpe_invisable_pre_rank * samples_pre_rank
    all_pa_mpjpe_invisable = pa_mpjpe_invisable_pre_rank * samples_pre_rank

    mean_mpjpe = all_mpjpe.sum() / all_samples
    mean_pa_mpjpe = all_pa_mpjpe.sum() / all_samples
    mean_mpvpe = all_mpvpe.sum() / all_samples

    mean_mpjpe_visable = all_mpjpe_visable.sum() / all_samples
    mean_pa_mpjpe_visable = all_pa_mpjpe_visable.sum() / all_samples
    mean_mpjpe_invisable = all_mpjpe_invisable.sum() / all_samples
    mean_pa_mpjpe_invisable = all_pa_mpjpe_invisable.sum() / all_samples
    
    result_dict = {
        'mpjpe': mean_mpjpe.item(),
        'pa_mpjpe': mean_pa_mpjpe.item(),
        'mpvpe': mean_mpvpe.item(),
        'mpjpe_visable': mean_mpjpe_visable.item(),
        'pa_mpjpe_visable': mean_pa_mpjpe_visable.item(),
        'mpjpe_invisable': mean_mpjpe_invisable.item(),
        'pa_mpjpe_invisable': mean_pa_mpjpe_invisable.item(),
    }
        
    if dist.get_rank() == 0:
        print('{} {}'.format(dataset_name, epoch))
        for k,v in result_dict.items():
            print(f'{k}: {v:.2f}')
        
        message = [f'{k}: {v:.2f}' for k, v in result_dict.items()]
        trainer.logger.info('{} '.format(dataset_name) + ' '.join(message))         
        
    dist.barrier()

if __name__ == "__main__":
    main()
