import argparse
import os
import torch
import torch.distributed as dist
import sys
from config import cfg
from common.base import Trainer
import torch.cuda.amp as amp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader.PW3D import PW3D
# from dataloader.CMU_Panotic import CMU_Panotic
from tensorboardX import SummaryWriter
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
    # '3dpw-crowd':{
    #     'best_MPJPE': 1e10,
    # },
    # '3dpw-pc':{
    #     'best_MPJPE': 1e10,
    # },
    # '3dpw-oc':{
    #     'best_MPJPE': 1e10,
    # },
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
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--save_folder', default='output')

    args = parser.parse_args()
    return args

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()

def main():
    args = parse_args()
    if args.cfg:
        yml_cfg = cfg.update(args)
    if args.distributed:
        # init_distributed_mode(args)
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(dist.get_rank()) 
        # 
        # assert dist.is_initialized(), "distributed is not initialized"
        
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
    trainer = Trainer(cfg)
    if dist.get_rank() == 0:
        os.system('nproc')
        message = '\n'.join(['{}:{}'.format(k, v) for k, v in yml_cfg.items()])
        trainer.logger.info('Experiment ID: {}'.format(args.exp_id))
        trainer.logger.info('logdir: {}'.format(cfg.log_dir))
        trainer.logger.info('atgs: ' + ' '.join(sys.argv))
        trainer.logger.info('work_size: {}'.format(dist.get_world_size()))
        trainer.logger.info('yml_cfg: \n{}'.format(message))
        os.system('find ./ -name \"*.yml\" -or -name \"*.py\" | xargs tar --exclude=\"*chumpy*\" --exclude=\"./data\" -zcf {}/code.tar.gz'.format(cfg.output_dir))

    trainer._make_batch_generator()
    trainer._make_model()
    test_dataset_dict = {}
    for dataset_name in best_dict:
        if '3dpw' in dataset_name:
            testset_loader = PW3D(transforms.ToTensor(), data_name=dataset_name)
        else:
            testset_loader = CMU_Panotic()
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
    

    scaler = amp.GradScaler(init_scale=args.init_scale, enabled=args.use_mixed_precision)

    if dist.get_rank() == 0:
        trainer.writer = SummaryWriter(logdir=cfg.output_dir)

    global_step = 0

    for epoch in range(trainer.start_epoch, cfg.end_epoch):      
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        trainer.model.train()
        if dist.is_initialized():
            assert trainer.sampler is not None, 'sampler is none'
        if trainer.sampler is not None:
            trainer.sampler.set_epoch(epoch)
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            if global_step > cfg.total_steps + 10:
                exit()
            inputs = {k: v.cuda() for k, v in inputs.items()}
            targets = {k: v.cuda() for k, v in targets.items()}
            meta_info = {k: v.cuda() for k, v in meta_info.items()}
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            with amp.autocast(args.use_mixed_precision):
                loss = trainer.model(inputs, targets, meta_info, 'train')

                loss = {k: loss[k].mean() for k in loss}
                loss = trainer.awl(loss)
                _loss = sum(loss[k] for k in loss)
            with amp.autocast(False):
                _loss = scaler.scale(_loss)
                _loss.backward()
                if cfg.max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainer.model.module.parameters(), cfg.max_norm)
                scaler.step(trainer.optimizer)
            scaler.update(args.init_scale)

            trainer.gpu_timer.toc()


            global_step += 1
            # if global_step % 200 == 0 and global_step != 0 and cfg.with_contrastive:
            #     # eavl ft model
            #     trainer.model.eval()
            #     for data_name in best_dict.keys():
            #         eval(global_step, trainer, data_name, test_dataset_dict[data_name]['dataset'], test_dataset_dict[data_name]['loader'])
            #     trainer.model.eval()

            if itr % 200 == 0 and dist.get_rank() == 0:
                screen = [
                    'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, len(trainer.batch_generator)),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * len(trainer.batch_generator)),
                    ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
                trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

            if itr % 1000 == 0 and itr != 0:
                trainer.model.eval()
                for data_name in best_dict.keys():
                    eval(epoch, trainer, data_name, test_dataset_dict[data_name]['dataset'], test_dataset_dict[data_name]['loader'])
                
            #     trainer.save_model({
            #         'epoch': epoch,
            #         'network': trainer.model.state_dict(),
            #         'optimizer': trainer.optimizer.state_dict(),
            #         'awl': trainer.awl.state_dict(),
            #     }, epoch)
                trainer.model.train()
        # eval model
        for data_name in best_dict.keys():
            eval(epoch, trainer, data_name, test_dataset_dict[data_name]['dataset'], test_dataset_dict[data_name]['loader'])
        dist.barrier()

    
    # To be done: Test
     
def eval(epoch, trainer, dataset_name, testset_loader, test_batch_generator):
    trainer.model.eval()
    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(test_batch_generator)):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}
        with torch.no_grad():
            out = trainer.model(inputs, targets, meta_info, 'test')
        out = {k: v.cpu().numpy() for k,v in out.items()}
        key = list(out.keys())[0]
        batch_size = out[key].shape[0]
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)] # batch_size * dict
        if not dist.is_initialized():
            cur_eval_result = testset_loader.evaluate(out, cur_sample_idx) # dict of list
            for k,v in cur_eval_result.items():
                if k in eval_result: 
                    eval_result[k] += v
                else: 
                    eval_result[k] = v
            cur_sample_idx += len(out)
        else:
            index_list = meta_info['idx'].flatten().long().tolist()
            cur_eval_result = testset_loader.random_idx_eval(out, index_list)
            for k,v in cur_eval_result.items():
                if k in eval_result: 
                    eval_result[k] += v
                else: 
                    eval_result[k] = v
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
            trainer.writer.add_scalar(f'test/epoch/{dataset_name}_{k}', v, epoch)
            print(f'{k}: {v:.2f}')
        
        message = [f'{k}: {v:.2f}' for k, v in result_dict.items()]
        # message = ' '.join(message)
        trainer.logger.info('{} '.format(dataset_name) + ' '.join(message))
        if result_dict['mpjpe'] < best_dict[dataset_name]['best_MPJPE']:
            best_dict[dataset_name]['best_MPJPE'] = result_dict['mpjpe']
            trainer.logger.info('best model: {}, best mpjpe: {:.2f}'.format(epoch, result_dict['mpjpe']))
            torch.save(trainer.model.state_dict(), os.path.join(cfg.model_dir, '{}_best_ckpt.pth.tar'.format(dataset_name)))            
        
    dist.barrier()

if __name__ == "__main__":
    main()
