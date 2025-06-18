#!/bin/bash
#SBATCH --job-name==res_frac_coords_predict            # 作业名称
#SBATCH --nodes=1                      # 使用1个节点
#SBATCH --ntasks=1                     # 启动1个任务
#SBATCH --cpus-per-task=16            # 每个任务分配16个CPU
#SBATCH --gres=gpu:2                   # 请求2个GPU
#SBATCH --time=24:00:00                # 任务运行最长时间为10小时
#SBATCH -A ai4phys      
#SBATCH -p vip_gpu_ailab                
#SBATCH --output=logs/train_output_%j.log    # 输出文件 (%j 会被作业ID替代)
#SBATCH --error=logs/train_error_%j.log      # 错误日志文件

CUDA_VISIBLE_DEVICES=0,1 python main.py \
        --config-file configs_plant/vitmatte_s_plant_finetune_500ep.py \
        --num-gpus 2 \
        --num-machines 1 \
        --dist-url "tcp://127.0.0.1:1417"

