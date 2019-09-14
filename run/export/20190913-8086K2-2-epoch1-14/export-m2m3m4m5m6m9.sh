# chmod a+x /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/export/20190913-8086K2-2-epoch1-14/*.sh
# /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/export/20190913-8086K2-2-epoch1-14/export-m2m3m4m5m6m9.sh
DATADIR=/data12/datasets3
# FullGru512x4_hw512_attpost_2080Ti_W2125-4.4
MODELDIR=/data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_2080Ti_W2125-4.4
# ls -1 /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_2080Ti_W2125-4.4/epoch*meta
CHECKPOINTS=(
epoch-1-884255
epoch-2-1768510
epoch-3-2652765
epoch-4-3537020
epoch-5-4421275
epoch-6-5305530
)
for i in "${CHECKPOINTS[@]}"
do
    export CUDA_VISIBLE_DEVICES=1; /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/v4-BiRnn.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s0-meta.json --num_classes=16714 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.2 --conv_1_prenet_dropout=0.2 --conv_1_dropout=0.2 --rnn_dropout=0.0 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=512 --attention_value_channels=512 --attention_hidden_size=512 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=True --rnn_attention_hidden_size=512 --use_rnn_attention_batch_norm=True --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=35000 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.065 --batch_size=1 --model_dir=${MODELDIR} --job=export --export_checkpoint=${MODELDIR}/${i} --export_dir=${MODELDIR}/export_epoch
done
# ls -1 /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_2080Ti_W2125-4.4/export_epoch
# 1568424545
# 1568424552
# 1568424557
# 1568424563
# 1568424570
# 1568424576

# FullGru512x4_hw512_attpost_highdrop_2080Ti_W2125-3.4
MODELDIR=/data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_highdrop_2080Ti_W2125-3.4
# ls /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_highdrop_2080Ti_W2125-3.4/epoch*meta
CHECKPOINTS=(
epoch-1-884255
epoch-2-1768510
epoch-3-2652765
epoch-3-3537017
epoch-4-4421272
epoch-5-5305527
epoch-6-6189782
epoch-7-7074037
epoch-8-7958292
epoch-9-8842547
epoch-10-9726802
epoch-11-10611057
epoch-12-11495312
# epoch-13-12379567
)
for i in "${CHECKPOINTS[@]}"
do
    export CUDA_VISIBLE_DEVICES=1; /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/v4-BiRnn.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s0-meta.json --num_classes=16714 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.5 --conv_1_prenet_dropout=0.5 --conv_1_dropout=0.5 --rnn_dropout=0.0 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=512 --attention_value_channels=512 --attention_hidden_size=512 --attention_num_heads=1 --attention_dropout=0.2 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=True --rnn_attention_hidden_size=512 --use_rnn_attention_batch_norm=True --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=35000 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.065 --batch_size=1 --model_dir=${MODELDIR} --job=export --export_checkpoint=${MODELDIR}/${i} --export_dir=${MODELDIR}/export_epoch
done
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_highdrop_2080Ti_W2125-3.4/export/* /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_highdrop_2080Ti_W2125-3.4/export_epoch/
# ls -1 /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_highdrop_2080Ti_W2125-3.4/export_epoch
# 1568424583
# 1568424589
# 1568424596
# 1568424602
# 1568424609
# 1568424615
# 1568424622
# 1568424629
# 1568424635
# 1568424642
# 1568424648
# 1568424655
# 1568424662
# 1567786205

# FullGru512x4_hw512_apo_hd_lr0.055_ws33k_1080Ti_8086K2-1.2
MODELDIR=/data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.055_ws33k_1080Ti_8086K2-1.2
# ls /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.055_ws33k_1080Ti_8086K2-1.2/epoch*meta
CHECKPOINTS=(
epoch-1-884255
epoch-2-1768510
epoch-3-2652765
epoch-4-3537020
epoch-5-4421275
epoch-6-5305530
epoch-7-6189785
epoch-8-7074040
epoch-9-7958295
epoch-10-8842550
epoch-11-9726805
epoch-12-10611060
epoch-13-11495315
# epoch-14-12379570
)
for i in "${CHECKPOINTS[@]}"
do
    export CUDA_VISIBLE_DEVICES=1; /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/v4-BiRnn.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s0-meta.json --num_classes=16714 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.5 --conv_1_prenet_dropout=0.5 --conv_1_dropout=0.5 --rnn_dropout=0.0 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=512 --attention_value_channels=512 --attention_hidden_size=512 --attention_num_heads=1 --attention_dropout=0.2 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=True --rnn_attention_hidden_size=512 --use_rnn_attention_batch_norm=True --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=33051 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.055 --batch_size=1 --model_dir=${MODELDIR} --job=export --export_checkpoint=${MODELDIR}/${i} --export_dir=${MODELDIR}/export_epoch
done
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.055_ws33k_1080Ti_8086K2-1.2/export/* /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.055_ws33k_1080Ti_8086K2-1.2/export_epoch/
# ls -1 /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.055_ws33k_1080Ti_8086K2-1.2/export_epoch
# 1568424669
# 1568424675
# 1568424682
# 1568424689
# 1568424695
# 1568424702
# 1568424709
# 1568424715
# 1568424722
# 1568424729
# 1568424735
# 1568424742
# 1568424749
# 1567786591

# FullGru512x4_hw512_apo_hd_lr0.075_ws18k_2080Ti_W2125-2.4
MODELDIR=/data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.075_ws18k_2080Ti_W2125-2.4
# ls /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.075_ws18k_2080Ti_W2125-2.4/epoch*meta
CHECKPOINTS=(
epoch-1-884255
epoch-2-1768510
epoch-3-2652765
epoch-4-3537020
epoch-5-4421275
epoch-6-5305530
epoch-7-6189785
epoch-8-7074040
# epoch-9-7958295
epoch-10-8842550
)
for i in "${CHECKPOINTS[@]}"
do
    export CUDA_VISIBLE_DEVICES=1; /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/v4-BiRnn.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s0-meta.json --num_classes=16714 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.5 --conv_1_prenet_dropout=0.5 --conv_1_dropout=0.5 --rnn_dropout=0.0 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=512 --attention_value_channels=512 --attention_hidden_size=512 --attention_num_heads=1 --attention_dropout=0.2 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=True --rnn_attention_hidden_size=512 --use_rnn_attention_batch_norm=True --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=18335 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.075 --batch_size=1 --model_dir=${MODELDIR} --job=export --export_checkpoint=${MODELDIR}/${i} --export_dir=${MODELDIR}/export_epoch
done
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.075_ws18k_2080Ti_W2125-2.4/export/* /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.075_ws18k_2080Ti_W2125-2.4/export_epoch/
# ls -1 /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.075_ws18k_2080Ti_W2125-2.4/export_epoch
# 1568424755
# 1568424762
# 1568424768
# 1568424775
# 1568424782
# 1568424789
# 1568424795
# 1568424802
# 1567786835
# 1568424808

# FullGru512x4_hw512_apo_hd_lr0.035_ws27k_2080Ti_W2125-1.4
MODELDIR=/data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.035_ws27k_2080Ti_W2125-1.4
# ls /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.035_ws27k_2080Ti_W2125-1.4/epoch*meta
CHECKPOINTS=(
epoch-1-884255
epoch-2-1768510
epoch-3-2652765
# epoch-4-3537020
)
for i in "${CHECKPOINTS[@]}"
do
    export CUDA_VISIBLE_DEVICES=1; /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/v4-BiRnn.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s0-meta.json --num_classes=16714 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.5 --conv_1_prenet_dropout=0.5 --conv_1_dropout=0.5 --rnn_dropout=0.0 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=512 --attention_value_channels=512 --attention_hidden_size=512 --attention_num_heads=1 --attention_dropout=0.2 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=True --rnn_attention_hidden_size=512 --use_rnn_attention_batch_norm=True --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=27778 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.035 --batch_size=1 --model_dir=${MODELDIR} --job=export --export_checkpoint=${MODELDIR}/${i} --export_dir=${MODELDIR}/export_epoch
done
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.035_ws27k_2080Ti_W2125-1.4/export/* /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.035_ws27k_2080Ti_W2125-1.4/export_epoch/
# ls -1 /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.035_ws27k_2080Ti_W2125-1.4/export_epoch
# 1568424815
# 1568424822
# 1568424828
# 1567787010

# FullGru512x4_hw512_apo_hd_lr0.045_ws30k_1080Ti_4960X-2.2
MODELDIR=/data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.045_ws30k_1080Ti_4960X-2.2
# ls /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.045_ws30k_1080Ti_4960X-2.2/epoch*meta
CHECKPOINTS=(
epoch-1-884255
)
for i in "${CHECKPOINTS[@]}"
do
    export CUDA_VISIBLE_DEVICES=1; /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/v4-BiRnn.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s0-meta.json --num_classes=16714 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.5 --conv_1_prenet_dropout=0.5 --conv_1_dropout=0.5 --rnn_dropout=0.0 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=512 --attention_value_channels=512 --attention_hidden_size=512 --attention_num_heads=1 --attention_dropout=0.2 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=True --rnn_attention_hidden_size=512 --use_rnn_attention_batch_norm=True --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=30710 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.045 --batch_size=1 --model_dir=${MODELDIR} --job=export --export_checkpoint=${MODELDIR}/${i} --export_dir=${MODELDIR}/export_epoch
done
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.045_ws30k_1080Ti_4960X-2.2/export/* /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.045_ws30k_1080Ti_4960X-2.2/export_epoch/
# ls -1 /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.045_ws30k_1080Ti_4960X-2.2/export_epoch
# 1568424835

# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_2080Ti_W2125-4.4/export_epoch/* /home/hotdogee/models/pfam3/
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_highdrop_2080Ti_W2125-3.4/export_epoch/* /home/hotdogee/models/pfam3/
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.055_ws33k_1080Ti_8086K2-1.2/export_epoch/* /home/hotdogee/models/pfam3/
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.075_ws18k_2080Ti_W2125-2.4/export_epoch/* /home/hotdogee/models/pfam3/
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.035_ws27k_2080Ti_W2125-1.4/export_epoch/* /home/hotdogee/models/pfam3/
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_apo_hd_lr0.045_ws30k_1080Ti_4960X-2.2/export_epoch/* /home/hotdogee/models/pfam3/

docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -p 8501:8501 --mount type=bind,source=/home/hotdogee/models,target=/models -t tensorflow/serving:latest-gpu --model_config_file=/models/models_config_pfam3_all.proto
