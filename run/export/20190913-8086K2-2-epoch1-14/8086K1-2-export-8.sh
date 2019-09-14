DATADIR=/data12/datasets3
# FullGru512x4_hw512_attpost_rnndrop_TITANRTX_8086K1-2.2
MODELDIR=/data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_rnndrop_TITANRTX_8086K1-2.2
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
epoch-14-12379570
epoch-15-13263825
epoch-16-14148080
epoch-17-15032335
# epoch-18-15916590
)
for i in "${CHECKPOINTS[@]}"
do
    export CUDA_VISIBLE_DEVICES=1; /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/v4-BiRnn.py --experiment_name=PARSE --host_script_name=PARSE --training_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --eval_data=${DATADIR}/pfam-regions-d0-s0-train.tfrecords --metadata_path=${DATADIR}/pfam-regions-d0-s0-meta.json --num_classes=16714 --save_summary_steps=200 --log_step_count_steps=200 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --use_crf=False --use_conv_batch_norm=True --use_rnn_batch_norm=True --use_conv_1_bank=True --use_conv_1_bank_odd=True --conv_1_filters=32 --conv_1_bank_size=15 --conv_1_strides=1 --use_conv_1_residual=True --use_conv_1_highway=True --conv_1_highway_depth=3 --conv_1_highway_units=512 --use_conv_1_prenet=False --embedded_dropout=0.2 --conv_1_prenet_dropout=0.2 --conv_1_dropout=0.2 --rnn_dropout=0.2 --use_batch_renorm=False --use_conv_1_attention=False --attention_key_channels=512 --attention_value_channels=512 --attention_hidden_size=512 --attention_num_heads=1 --attention_dropout=0.0 --attention_type=local_unmasked --attention_block_length=512 --attention_filter_width=256 --use_conv_1_attention_batch_norm=False --use_rnn_attention=True --rnn_attention_hidden_size=512 --use_rnn_attention_batch_norm=True --use_rnn_peepholes=False --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --rnn_num_units=[512,512,512,512] --rnn_cell_type=CudnnGRU --warmup_schedule=exp --warmup_steps=35000 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --learning_rate=0.065 --batch_size=1 --model_dir=${MODELDIR} --job=export --export_checkpoint=${MODELDIR}/${i} --export_dir=${MODELDIR}/export_epoch
done
# chmod a+x /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/export/20190913-8086K2-2-epoch1-14/*.sh
# /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/export/20190913-8086K2-2-epoch1-14/8086K1-2-export-8.sh
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_rnndrop_TITANRTX_8086K1-2.2/export/* /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_rnndrop_TITANRTX_8086K1-2.2/export_epoch/
# ls /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_attpost_rnndrop_TITANRTX_8086K1-2.2/export_epoch/
# 1568367500
# 1568367510
# 1568367520
# 1568367531
# 1568367542
# 1568367553
# 1568367564
# 1568367575
# 1568367586
# 1568367598
# 1568367609
# 1568367620
# 1568367631
# 1568367642
# 1568367654
# 1568367665
# 1568367676
# 1567787563
