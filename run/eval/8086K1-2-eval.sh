CARDID=1
TFSCRIPT=v4-BiRnn
HOSTSCRIPT=$(basename ${BASH_SOURCE%.*})
DATADIR=/home/hotdogee/datasets2
DATASET=pfam-regions-d0-s20
BATCHSIZE=1
TOTAL_INSTALLED_GPUS=2
CARDTYPE=2080Ti
MODELDIR=/data12/checkpoints/${DATASET}/${TFSCRIPT}/Gru256x4_step2192725_ws0_lr0.001_2080Ti_8086K2-1.2
CHECKPOINT=epoch-13-9195287

export CUDA_VISIBLE_DEVICES=${CARDID}; python ${TFSCRIPT}.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s20-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s20-test.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s20-meta.json --num_classes=16715 --batch_size=${BATCHSIZE} --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.9 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --conv_1_filters=32 --conv_1_bank_size=16 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=32 --use_conv_1_prenet=False --conv_1_prenet_dropout=0.2 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=False --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=None --warmup_schedule=exp --warmup_steps=1 --rnn_num_units=[256,256,256,256] --rnn_cell_type=CudnnGRU --warmup_repeat_steps=0 --warmup_start_lr=0.5 --learning_rate=0.001 --model_dir=${MODELDIR} --job=eval --predict_top_k=5 --eval_format=msgpack --eval_level=min --eval_checkpoint=${MODELDIR}/${CHECKPOINT}
