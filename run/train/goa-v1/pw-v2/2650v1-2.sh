# cd /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2
# chmod a+x ./run/train/goa-v1/*.sh
# source /opt/intel/parallel_studio_xe_2019/bin/psxevars.sh
# source ~/venv/tf15/bin/activate
# ./run/train/goa-v1/2650v1-2.sh
CARDID=1
HOSTSCRIPT=$(basename ${BASH_SOURCE%.*})
DATADIR=/data12/goa/goa-20191015
DATASET=goa-20191015
TFSCRIPT=goa-v1
BATCHSIZE=1
TOTAL_INSTALLED_GPUS=2
CARDTYPE=1080Ti
# _1080Ti_2650v1-2.2
MODELDIR=/data12/checkpoints/${DATASET}/${TFSCRIPT}
PYTHON=/home/hotdogee/venv/tf15/bin/python
# export CUDA_VISIBLE_DEVICES=${CARDID}; ${PYTHON} ${TFSCRIPT}.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/goa-20191015-train.tfrecords --eval_data=${DATADIR}/goa-20191015-train.tfrecords --metadata_path=${DATADIR}/goa-20191015-meta.json --save_summary_steps=200 --log_step_count_steps=20 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.2 --conv_1_prenet_dropout=0.2 --conv_1_dropout=0.2 --rnn_dropout=0.0 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=False --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --use_output_highway=True --output_highway_depth=1 --output_highway_units=1024 --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[256,256,256,256] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=1300 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.03 --loss_pos_weight=3.3 --batch_size=${BATCHSIZE} --model_dir=${MODELDIR}/Gru256x4-pw3.3f_goa_1080Ti_2650v1-2.2
# 12864 steps per epoch
# scan/pw-v2
export CUDA_VISIBLE_DEVICES=${CARDID}; ${PYTHON} ${TFSCRIPT}.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/goa-20191015-train.tfrecords --eval_data=${DATADIR}/goa-20191015-train.tfrecords --metadata_path=${DATADIR}/goa-20191015-meta.json --save_summary_steps=200 --log_step_count_steps=20 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.2 --conv_1_prenet_dropout=0.2 --conv_1_dropout=0.2 --rnn_dropout=0.0 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=False --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --use_output_highway=True --output_highway_depth=1 --output_highway_units=1024 --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[256,256,256,256] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=1300 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.03 --loss_pos_weight=2.5 --batch_size=${BATCHSIZE} --model_dir=${MODELDIR}/pw-v2/Gru256x4-pw2.5_goa_1080Ti_2650v1-2.2
