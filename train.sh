# gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',')
# gpu_ids=${gpu_ids%?}  # 
# # gpu_ids="0" # only use one gpu for debug
# gpu_num=$(echo $gpu_ids | tr ',' '\n' | wc -l)


lr=1e-4 
wd=1e-4 
exp_id="main_train" 
TORCH_USE_CUDA_DSA=1 \
CUDA_VISIBLE_DEVICES=3,7 \
torchrun \
--master_port 29521 \
--nproc_per_node 2 \
train_vpd.py \
--cfg ./assets/yaml/main_train_3dpw.yml \
--distributed \
--lr 2e-4 \
--lr_backbone 2e-4 \
--weight_decay 0.00001 \
--exp_id "vpd_5" \
--resume_ckpt 'output/vpd_useattn_finetune_1/checkpoint/3dpw-oc_best_ckpt.pth.tar' \
--continue 

CUDA_VISIBLE_DEVICES=6,7 \
torchrun \
--master_port 29531 \
--nproc_per_node 2 \
train_vpd.py \
--cfg ./assets/yaml/main_train_v2.yml \
--distributed \
--lr 2e-5 \
--lr_backbone 2e-5 \
--weight_decay 0.00001 \
--exp_id "vpd_5" \
--resume_ckpt 'output/vpd_useattn_finetune_1/checkpoint/3dpw-oc_best_ckpt.pth.tar' \
--continue

# --amp \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
torchrun \
--master_port 29511 \
--nproc_per_node 7 \
train_vpd.py \
--cfg ./assets/yaml/main_train_3dpw.yml \
--distributed \
--lr 4e-5 \
--lr_backbone 4e-5 \
--weight_decay 0.00001 \
--exp_id "vpd_2" \
--resume_ckpt 'output/simple_vpd_1/checkpoint/3dpw_best_ckpt.pth.tar' \
--continue


export NCCL_DEBUG=INFO
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
torchrun \
--master_port 29511 \
--nproc_per_node 7 \
train.py \
--cfg ./assets/yaml/fine_tune.yml \
--distributed \
--lr 5e-5 \
--lr_backbone 5e-5 \
--weight_decay 0.0001 \
--exp_id "fine_tune" \
--vqtokens 48 \
--vqclasses 2048 \
--resume_ckpt output/normal_novqloss_2048x48_augment_cliff/checkpoint/3dpw_best_ckpt.pth.tar \
--continue \
--save_folder 'JOTR_vqvae_2048x48_mask'

CUDA_VISIBLE_DEVICES=4,5 \
python3 -m torch.distributed.launch \
--master_port 29511 \
--nproc_per_node 2 \
train_teacher.py \
--cfg ./assets/yaml/fine_tune_ssl.yml \
--distributed \
--lr 5e-5 \
--lr_backbone 5e-5 \
--weight_decay 0.0001 \
--exp_id "fine_tune" \
--vqtokens 48 \
--vqclasses 2048 \
--resume_ckpt output/train_teacher_swin/checkpoint/3dpw_best_ckpt.pth.tar \
--continue
# lr=5e-5
# wd=1e-3
# exp_id="fine_tune"
# CUDA_VISIBLE_DEVICES=$gpu_ids \
# python3 -m torch.distributed.launch \
# --master_port 29521 \
# --nproc_per_node $gpu_num \
# train.py \
# --cfg ./assets/yaml/fine_tune.yml \
# --distributed \
# --amp \
# --with_contrastive \
# --total_steps 10000 \
# --lr $lr \
# --lr_backbone $lr \
# --weight_decay $wd \
# --exp_id $exp_id \
# --resume_ckpt JOTR/output/JOTR/checkpoint/snapshot_17.pth.tar

CUDA_VISIBLE_DEVICES=4,5 \
python -m torch.distributed.launch \
--master_port 29521 \
--nproc_per_node 2 \
test_allgather.py

CUDA_VISIBLE_DEVICES=1,3 \
python3 -m torch.distributed.launch \
--master_port 29511 \
--nproc_per_node 2 \
train.py \
--cfg ./assets/yaml/fine_tune.yml \
--distributed \
--lr 1e-4 \
--lr_backbone 1e-4 \
--weight_decay 0.0001 \
--exp_id "fine_tune" \
--vqtokens 40 \
--vqclasses 2048 \
--resume_ckpt output/normal_novqloss_2048x48_augment_cliff/checkpoint/3dpw_best_ckpt.pth.tar \
--continue 

--resume_ckpt output/vit_debug/checkpoint/3dpw-pc_best_ckpt.pth.tar \
--continue

CUDA_VISIBLE_DEVICES=6,7 \
python3 -m torch.distributed.launch \
--master_port 29521 \
--nproc_per_node 2 \
train_distill.py \
--cfg ./assets/yaml/fine_tune_ssl.yml \
--distributed \
--lr 2e-5 \
--lr_backbone 2e-5 \
--weight_decay 0.00001 \
--exp_id "fine_tune" \
--vqtokens 48 \
--vqclasses 2048 \
--resume_ckpt output/distill_swin_swin_simclr_3/checkpoint/3dpw-pc_best_ckpt.pth.tar \
--continue