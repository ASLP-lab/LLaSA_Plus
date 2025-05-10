

# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=2,3,4,5
export HF_ENDPOINT="https://hf-mirror.com"
# export CUDA_LAUNCH_BLOCKING=1

torchrun --nproc_per_node=4 --master_port=49501 finetune_offline_mtp2_notcat.py