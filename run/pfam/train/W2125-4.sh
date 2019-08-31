CARDID=3
HOSTSCRIPT=$(basename ${BASH_SOURCE%.*})
DATADIR=/home/hotdogee/datasets3
DATASET=pfam-regions-d0-s0
TFSCRIPT=v4-BiRnn
BATCHSIZE=1
TOTAL_INSTALLED_GPUS=4
CARDTYPE=2080Ti
# _TITANV_W2125-1.4
MODELDIR=/data12/checkpoints/${DATASET}/${TFSCRIPT}
PYTHON=/home/hotdogee/venv/tf37/bin/python
export CUDA_VISIBLE_DEVICES=${CARDID}; ${PYTHON} ${TFSCRIPT}.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s0-meta.json --num_classes=16714 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.2 --conv_1_prenet_dropout=0.2 --conv_1_dropout=0.2 --rnn_dropout=0.0 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=512 --attention_value_channels=512 --attention_hidden_size=512 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=True --rnn_attention_hidden_size=512 --use_rnn_attention_batch_norm=True --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=35000 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.065 --batch_size=${BATCHSIZE} --model_dir=${MODELDIR}/FullGru512x4_hw512_attpost_${CARDTYPE}_${HOSTSCRIPT}.${TOTAL_INSTALLED_GPUS}
