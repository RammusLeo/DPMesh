# gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',')
# gpu_ids=${gpu_ids%?}  # 
# # gpu_ids="0" # only use one gpu for debug
# gpu_num=$(echo $gpu_ids | tr ',' '\n' | wc -l)


MESA_GL_VERSION_OVERRIDE==4.1 \
CUDA_VISIBLE_DEVICES=1 \
torchrun \
--master_port 29591 \
--nproc_per_node 1 \
eval_occ.py \
--cfg ./assets/yaml/main_train_v2_eval.yml \
--exp_id="main_train" \
--distributed \
--resume_ckpt output/vpd_useattn_finetune_1/checkpoint/3dpw_best_ckpt.pth.tar \
--continue \
--vqtokens 48 \
--vqclasses 2048

MESA_GL_VERSION_OVERRIDE==4.1 \
CUDA_VISIBLE_DEVICES=3 \
python3 -m torch.distributed.launch \
--master_port 29501 \
--nproc_per_node 1 \
eval.py \
--cfg ./assets/yaml/main_train.yml \
--exp_id="main_train" \
--distributed \
--resume_ckpt checkpoint/3dpw_best_ckpt.pth.tar \
--continue \
--vqtokens 48 \
--vqclasses 2048

CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch \
--master_port 29500 \
--nproc_per_node 1 \
eval_distill.py \
--cfg ./assets/yaml/main_train.yml \
--exp_id="main_train" \
--distributed \
--resume_ckpt output/distill_swin_swin_2/checkpoint/3dpw-pc_best_ckpt.pth.tar \
--continue \
--vqtokens 48 \
--vqclasses 2048

CUDA_VISIBLE_DEVICES=6 \
python3 -m torch.distributed.launch \
--master_port 29500 \
--nproc_per_node 1 \
eval_distill_flag.py \
--cfg ./assets/yaml/main_train.yml \
--exp_id="main_train" \
--distributed \
--resume_ckpt output/distill_swin_swin_2/checkpoint/3dpw-oc_best_ckpt.pth.tar \
--continue \
--vqtokens 48 \
--vqclasses 2048