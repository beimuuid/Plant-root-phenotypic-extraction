CUDA_VISIBLE_DEVICES=0,1 python main.py \
        --config-file configs_plant/vitmatte_s_plant_finetune_500ep.py \
        --num-gpus 2 \
        --num-machines 1 \
        --dist-url "tcp://127.0.0.1:1417"

