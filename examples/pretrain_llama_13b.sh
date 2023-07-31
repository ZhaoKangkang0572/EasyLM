#! /bin/bash

# This is the example script to pretrain a 7B LLaMA model on a TPU v4-512 pod.
# These hyperparameters are the ones we used to train the OpenLLaMA 7B model on
# the RedPajama dataset. To use this on TPU pod, you need to run this
# script on every hosts in a TPU pod.

# Put your WANDB API key here to enable logging to wandb.
export WANDB_API_KEY='8d90ffa40a01a3bb4700d686b7821fdccda82631'

# TPU specific flags to improve training throughput
# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'
easylm_dir="/opt/ml/EasyLM"
export PYTHONPATH="${easylm_dir}:$PYTHONPATH"
echo "The current PYTHONPATH is: $PYTHONPATH"

mkdir data_train
chmod +x /s5cmd
/s5cmd sync s3://pretrained/llama2/* /

/s5cmd sync s3://pretrained/data/train/* /data_train/


/s5cmd sync s3://pretrained/data/test/* /data_train/

mkdir /train_logs
mkdir /train_logs/checkpoint

python -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,64,1' \
    --dtype='bf16' \
    --total_steps=250000 \
    --log_freq=50 \
    --save_model_freq=0 \
    --save_milestone_freq=2500 \
    --load_llama_config='13b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --tokenizer.vocab_file='/llama2/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
    --optimizer.adamw_optimizer.lr_decay_steps=250000 \
    --train_dataset.type='json' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.json_dataset.path='/data_train/train' \
    --train_dataset.json_dataset.seq_length=4096 \
    --train_dataset.json_dataset.batch_size=32 \
    --train_dataset.json_dataset.tokenizer_processes=16 \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.prefix='EasyLM' \
    --logger.project="llama2_13b_pretrain" \
    --logger.output_dir="/train_logs/checkpoint" \
    --logger.wandb_dir="$HOME/experiment_output/open_llama_7b"
|& tee $HOME/output.txt


chmod +x /s5cmd
/s5cmd sync /train_logs/checkpoint s3://demo-instance-kang/lm_demo/output/$(date +%Y-%m-%d-%H-%M-%S)/

