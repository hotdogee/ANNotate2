DATADIR=/data12/datasets3
# m10: Gru512x8_Pfam31_TITANRTX_8086K1-2.2
MODELDIR=/data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn/Gru512x8_Pfam31_TITANRTX_8086K1-2.2
CHECKPOINTS=(
step-88234
step-176574
step-264745
step-352965
step-441289
step-529290
step-617036
step-704828
step-792789
epoch-1-884255
)
for i in "${CHECKPOINTS[@]}"
do
    export CUDA_VISIBLE_DEVICES=0; /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/v5-BiRnn.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s0-meta.json --data_version=v1 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.2 --conv_1_prenet_dropout=0.2 --conv_1_dropout=0.2 --rnn_dropout=0.0 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=32 --attention_value_channels=32 --attention_hidden_size=32 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=128 --attention_filter_width=64 --use_conv_1_attention_batch_norm=False --use_rnn_attention=False --rnn_attention_hidden_size=128 --use_rnn_attention_batch_norm=False --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[512,512,512,512,512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=35000 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.065 --batch_size=1 --model_dir=${MODELDIR} --job=export --export_checkpoint=${MODELDIR}/${i} --export_dir=${MODELDIR}/export_epoch1
done

# chmod a+x /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/predict/20191013-512x8/*.sh
# /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/predict/20191013-512x8/8086K1-export-m10.sh
# ls -1 /data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn/Gru512x8_Pfam31_TITANRTX_8086K1-2.2/export_epoch1/
# 1570954345
# 1570954352
# 1570954359
# 1570954367
# 1570954374
# 1570954382
# 1570954390
# 1570954397
# 1570954405
# 1570954413
# mkdir /home/hotdogee/models/pfam-m10e1
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn/Gru512x8_Pfam31_TITANRTX_8086K1-2.2/export_epoch1/* /home/hotdogee/models/pfam-m10e1
# ll /home/hotdogee/models/pfam-m10e1
# vi /home/hotdogee/models/models_config_pfam_m10e1.proto
# model_config_list {
#   config {
#     name: 'pfam'
#     base_path: '/models/pfam-m10e1/'
#     model_platform: 'tensorflow'
#     model_version_policy {
#         all {}
#     }
#   }
# }
# docker run --gpus '"device=0"' -p 8501:8501 --mount type=bind,source=/home/hotdogee/models,target=/models -t tensorflow/serving:latest-gpu --model_config_file=/models/models_config_pfam_m10e1.proto --file_system_poll_wait_seconds=60
# docker run --gpus '"device=1"' -p 8601:8501 --mount type=bind,source=/home/hotdogee/models,target=/models -t tensorflow/serving:latest-gpu --model_config_file=/models/models_config_pfam_m10e1.proto --file_system_poll_wait_seconds=60
# docker run --gpus '"device=2"' -p 8701:8501 --mount type=bind,source=/home/hotdogee/models,target=/models -t tensorflow/serving:latest-gpu --model_config_file=/models/models_config_pfam_m10e1.proto --file_system_poll_wait_seconds=60
# docker run --gpus '"device=3"' -p 8801:8501 --mount type=bind,source=/home/hotdogee/models,target=/models -t tensorflow/serving:latest-gpu --model_config_file=/models/models_config_pfam_m10e1.proto --file_system_poll_wait_seconds=60

# curl -d '{"instances": ["FLIM","VP","AWGST"]}' -X POST http://192.168.1.33:8601/v1/models/pfam:predict
# curl -d '{"instances": ["FLIM","VP","AWGST"]}' -X POST http://192.168.1.33:8601/v1/models/pfam/versions/1570954345:predict
# curl -d '{"instances": ["FLIM","VP","AWGST"]}' -X POST http://localhost:8501/v1/models/pfam/versions/1570954345:predict
# curl -d '{"instances": ["FLIM","VP","AWGST"]}' -X POST http://192.168.1.35:8501/v1/models/pfam/versions/1570954345:predict

# cp -r /data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn/Gru512x8_Pfam31_TITANRTX_8086K1-2.2/export_epoch1/1570954359 /home/hotdogee/models/pfam-m10e1
# rm -rf /home/hotdogee/models/pfam-m10e1/1570954345

# cp -r /data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn/Gru512x8_Pfam31_TITANRTX_8086K1-2.2/export_epoch1/1570954367 /home/hotdogee/models/pfam-m10e1
# rm -rf /home/hotdogee/models/pfam-m10e1/1570954352

# cp -r /data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn/Gru512x8_Pfam31_TITANRTX_8086K1-2.2/export_epoch1/1570954374 /home/hotdogee/models/pfam-m10e1
# rm -rf /home/hotdogee/models/pfam-m10e1/1570954359

# cp -r /data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn/Gru512x8_Pfam31_TITANRTX_8086K1-2.2/export_epoch1/1570954382 /home/hotdogee/models/pfam-m10e1
# rm -rf /home/hotdogee/models/pfam-m10e1/1570954367

# cp -r /data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn/Gru512x8_Pfam31_TITANRTX_8086K1-2.2/export_epoch1/1570954390 /home/hotdogee/models/pfam-m10e1
# rm -rf /home/hotdogee/models/pfam-m10e1/1570954374

# cp -r /data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn/Gru512x8_Pfam31_TITANRTX_8086K1-2.2/export_epoch1/1570954397 /home/hotdogee/models/pfam-m10e1
# rm -rf /home/hotdogee/models/pfam-m10e1/1570954382

# cp -r /data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn/Gru512x8_Pfam31_TITANRTX_8086K1-2.2/export_epoch1/1570954405 /home/hotdogee/models/pfam-m10e1
# rm -rf /home/hotdogee/models/pfam-m10e1/1570954390

cp -r /data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn/Gru512x8_Pfam31_TITANRTX_8086K1-2.2/export_epoch1/1570954413 /home/hotdogee/models/pfam-m10e1
rm -rf /home/hotdogee/models/pfam-m10e1/1570954397
