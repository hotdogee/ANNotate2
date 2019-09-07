# export CUDA_VISIBLE_DEVICES=${CARDID}; python ${TFSCRIPT}.py --training_data=${DATADIR}/pfam-regions-d0-s20-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s20-test.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s20-meta.json --num_classes=16715 --batch_size=${BATCHSIZE} --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.9 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --rnn_cell_type=CudnnGRU --use_conv_1_bank=False --conv_1_filters=32 --conv_1_bank_size=16 --conv_1_strides=1 --use_conv_1_residual=False --use_conv_1_highway=False --conv_1_highway_depth=3 --conv_1_highway_units=32 --use_conv_1_prenet=False --conv_1_prenet_dropout=0.2 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=False --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --optimizer=Adam --learning_rate_decay_fn=noisy_linear_cosine_decay --warmup_start_lr=0.001 --warmup_schedule=exp --warmup_steps=0 --learning_rate=0.03 --model_dir=${MODELDIR}/Bench-tf1.10.1-n410.48-c10.0-d7.3.0-u18.04-20180921-${CARDTYPE}-${HOSTSCRIPT}
# export CUDA_VISIBLE_DEVICES=${CARDID}; python ${TFSCRIPT}.py --training_data=${DATADIR}/pfam-regions-d0-s20-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s20-test.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s20-meta.json --num_classes=16715 --batch_size=${BATCHSIZE} --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.9 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --rnn_cell_type=CudnnLSTM --use_conv_1_bank=True --conv_1_filters=32 --conv_1_bank_size=16 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=32 --use_conv_1_prenet=False --conv_1_prenet_dropout=0.2 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=True --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=None --warmup_start_lr=0.001 --warmup_schedule=exp --warmup_steps=35000 --learning_rate=0.008 --model_dir=${MODELDIR}/RnnAttOnly-lr0.008-ws35000-${CARDTYPE}-${HOSTSCRIPT}
# export CUDA_VISIBLE_DEVICES=${CARDID}; python ${TFSCRIPT}.py --training_data=${DATADIR}/pfam-regions-d0-s20-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s20-test.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s20-meta.json --num_classes=16715 --batch_size=${BATCHSIZE} --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.9 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --rnn_cell_type=CudnnGRU --use_conv_1_bank=True --conv_1_filters=32 --conv_1_bank_size=16 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=32 --use_conv_1_prenet=False --conv_1_prenet_dropout=0.2 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=True --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=None --warmup_start_lr=0.001 --warmup_schedule=exp --warmup_steps=35000 --learning_rate=0.03 --model_dir=${MODELDIR}/RnnAttGru-lr0.03-ws35000-${CARDTYPE}-${HOSTSCRIPT}
# export CUDA_VISIBLE_DEVICES=${CARDID}; python ${TFSCRIPT}.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s20-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s20-test.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s20-meta.json --num_classes=16715 --batch_size=${BATCHSIZE} --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.9 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --rnn_cell_type=CudnnLSTM --use_conv_1_bank=True --conv_1_filters=32 --conv_1_bank_size=16 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=32 --use_conv_1_prenet=False --conv_1_prenet_dropout=0.2 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=False --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=None --warmup_start_lr=0.001 --warmup_schedule=exp --warmup_steps=35000 --learning_rate=0.09 --model_dir=${MODELDIR}/NoAtt_lr0.09_${CARDTYPE}_${HOSTSCRIPT}.${TOTAL_INSTALLED_GPUS}
# export CUDA_VISIBLE_DEVICES=${CARDID}; python ${TFSCRIPT}.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s20-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s20-test.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s20-meta.json --num_classes=16715 --batch_size=${BATCHSIZE} --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.9 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --conv_1_filters=32 --conv_1_bank_size=16 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=32 --use_conv_1_prenet=False --conv_1_prenet_dropout=0.2 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=False --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=None --warmup_start_lr=0.001 --warmup_schedule=exp --warmup_steps=35000 --learning_rate=0.13 --rnn_num_units=[256,256] --rnn_cell_type=CudnnGRU --model_dir=${MODELDIR}/Gru256x2_lr0.13_${CARDTYPE}_${HOSTSCRIPT}.${TOTAL_INSTALLED_GPUS}
# export CUDA_VISIBLE_DEVICES=${CARDID}; python ${TFSCRIPT}.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s20-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s20-test.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s20-meta.json --num_classes=16715 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.9 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --conv_1_filters=32 --conv_1_bank_size=16 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=32 --use_conv_1_prenet=False --conv_1_prenet_dropout=0.2 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=False --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=None --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=35000 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.065 --batch_size=${BATCHSIZE} --model_dir=${MODELDIR}/Gru512x4_b1_lr0.065_${CARDTYPE}_${HOSTSCRIPT}.${TOTAL_INSTALLED_GPUS}
# export CUDA_VISIBLE_DEVICES=${CARDID}; python ${TFSCRIPT}.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s20-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s20-test.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s20-meta.json --num_classes=16715 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.9 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --conv_1_filters=32 --conv_1_bank_size=16 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=32 --use_conv_1_prenet=False --conv_1_prenet_dropout=0.2 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=False --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=None --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=1 --warmup_repeat_steps=0 --warmup_start_lr=0.5 --learning_rate=0.0325 --batch_size=${BATCHSIZE} --model_dir=${MODELDIR}/Gru512x4_step494346_b1_ws0_lr0.0325_${CARDTYPE}_${HOSTSCRIPT}.${TOTAL_INSTALLED_GPUS}
# CARDID=1
# TFSCRIPT=v4-BiRnn
# HOSTSCRIPT=$(basename ${BASH_SOURCE%.*})
# DATADIR=/home/hotdogee/datasets2
# DATASET=pfam-regions-d0-s20
# BATCHSIZE=1
# TOTAL_INSTALLED_GPUS=2
# CARDTYPE=TITANV
# # _TITANV_8086K1-2.2
# # cp -r ../prep/Gru512x4_b1_lr0.065_TITANV_8086K1-2.2_step494346/ ./Gru512x4_step494346_b1_ws0_lr0.0325_TITANV_8086K1-2.2
# # cp -r ../prep/Gru512x4_step494346_b1_ws0_lr0.0325_TITANV_8086K1-2.2_step1264123/ ./Gru512x4_step1264123_b1_ws0_lr0.00325_TITANV_8086K1-2.2
# MODELDIR=/data2r1/checkpoints/${DATASET}/${TFSCRIPT}
# export CUDA_VISIBLE_DEVICES=${CARDID}; python ${TFSCRIPT}.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s20-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s20-test.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s20-meta.json --num_classes=16715 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.9 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --conv_1_filters=32 --conv_1_bank_size=16 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=32 --use_conv_1_prenet=False --conv_1_prenet_dropout=0.2 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=False --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=None --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=1 --warmup_repeat_steps=0 --warmup_start_lr=0.5 --learning_rate=0.00325 --batch_size=${BATCHSIZE} --model_dir=${MODELDIR}/Gru512x4_step1264123_b1_ws0_lr0.00325_${CARDTYPE}_${HOSTSCRIPT}.${TOTAL_INSTALLED_GPUS}
CARDID=1
HOSTSCRIPT=$(basename ${BASH_SOURCE%.*})
DATADIR=/home/hotdogee/datasets3
DATASET=pfam-regions-d0-s0
TFSCRIPT=v4-BiRnn
BATCHSIZE=1
TOTAL_INSTALLED_GPUS=2
CARDTYPE=TITANRTX
# FullGru512x4_hw512_attpost_rnndrop_TITANRTX_8086K1-2.2
MODELDIR=/data12/checkpoints/${DATASET}/${TFSCRIPT}
export CUDA_VISIBLE_DEVICES=${CARDID}; python ${TFSCRIPT}.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s0-meta.json --num_classes=16714 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.2 --conv_1_prenet_dropout=0.2 --conv_1_dropout=0.2 --rnn_dropout=0.2 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=512 --attention_value_channels=512 --attention_hidden_size=512 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=512 --attention_filter_width=256 --use_conv_1_attention_batch_norm=False --use_rnn_attention=True --rnn_attention_hidden_size=512 --use_rnn_attention_batch_norm=True --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=35000 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.065 --batch_size=${BATCHSIZE} --model_dir=${MODELDIR}/FullGru512x4_hw512_attpost_rnndrop_TITANRTX_8086K1-2.2 --job=export --export_checkpoint=${MODELDIR}/FullGru512x4_hw512_attpost_rnndrop_TITANRTX_8086K1-2.2/epoch-18-15916590
# 95.91% - 167d - FullGru512x4_hw512_attpost_rnndrop_TITANRTX_8086K1-2.2 - epoch-18-15916590
# /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_rnndrop_TITANRTX_8086K1-2.2/export/1567787563
