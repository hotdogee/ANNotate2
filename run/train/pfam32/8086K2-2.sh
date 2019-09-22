CARDID=1
HOSTSCRIPT=$(basename ${BASH_SOURCE%.*})
DATADIR=/home/hotdogee/datasets3/Pfam32.0
DATASET=pfam32-regions-d0-s0
TFSCRIPT=v5-BiRnn
BATCHSIZE=1
TOTAL_INSTALLED_GPUS=2
CARDTYPE=TITANV
# _TITANV_8086K2-2.2
MODELDIR=/data12/checkpoints/${DATASET}/${TFSCRIPT}
PYTHON=/home/hotdogee/venv/tf37/bin/python
export CUDA_VISIBLE_DEVICES=${CARDID}; ${PYTHON} ${TFSCRIPT}.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam32-regions-d0-s0-train.tfrecords --eval_data=${DATADIR}/pfam32-regions-d0-s0-train.tfrecords --metadata_path=${DATADIR}/pfam32-regions-d0-s0-meta.json --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.2 --conv_1_prenet_dropout=0.2 --conv_1_dropout=0.2 --rnn_dropout=0.0 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=False --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=35000 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.065 --batch_size=${BATCHSIZE} --model_dir=${MODELDIR}/Gru512x4-hw512_Pfam32_TITANV_8086K2-2.2
# /data12/checkpoints/pfam32-regions-d0-s0/v5-BiRnn
# Epoch 0:   0%|   | 3816/88966235 [01:19<516:11:54, 47.87seq/s]
# 2019-09-21 13:56:56,225 Gru512x4-hw512 8086K2-2.2[6247] INFO accuracy:  0.01714%, loss:  9.81872, lr:  0.00006, step:       0
# 2019-09-21 14:01:01,990 Gru512x4-hw512 8086K2-2.2[6247] INFO global_step/sec: 0.813785
# 2019-09-21 14:01:01,990 Gru512x4-hw512 8086K2-2.2[6247] INFO accuracy: 21.24207%, loss:  8.51614, lr:  0.00007, step:     200 (245.765 sec)
# 2019-09-21 14:05:08,752 Gru512x4-hw512 8086K2-2.2[6247] INFO global_step/sec: 0.810496
# 2019-09-21 14:05:08,754 Gru512x4-hw512 8086K2-2.2[6247] INFO accuracy: 40.46223%, loss:  6.64726, lr:  0.00007, step:     400 (246.763 sec)
