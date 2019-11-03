# cd /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2
# chmod a+x ./run/train/goa-v2/*.sh
# source /opt/intel/parallel_studio_xe_2019/bin/psxevars.sh
# source ~/venv/tf15/bin/activate
# ./run/train/goa-v2/8086K1-2.sh
CARDID=1
HOSTSCRIPT=$(basename ${BASH_SOURCE%.*})
DATADIR=/data12/goa/goa-20191015
DATASET=goa-20191015-v2
TFSCRIPT=goa-v2
BATCHSIZE=1
TOTAL_INSTALLED_GPUS=2
CARDTYPE=TITANRTX
# _TITANRTX_8086K1-2.2
MODELDIR=/data12/checkpoints/${DATASET}/${TFSCRIPT}
PYTHON=/home/hotdogee/venv/tf15/bin/python
# export CUDA_VISIBLE_DEVICES=${CARDID}; ${PYTHON} ${TFSCRIPT}.py --training_data=${DATADIR}/${DATASET}-train.tfrecords --eval_data=${DATADIR}/${DATASET}-train.tfrecords --metadata_path=${DATADIR}/${DATASET}-meta.json --save_summary_steps=100 --log_step_count_steps=10 --learning_rate_decay_steps=180000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=900 --keep_checkpoint_max=24 --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --warmup_schedule=exp --warmup_steps=1300 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --batch_size=1 --gpu_allow_growth=False --encoder_embed_dim=32 --decoder_embed_dim=512 --encoder_num_units=[512] --decoder_num_units=[256,256,256] --attention_num_units=128 --decoder_num_residual_layers=2 --dynamic_rnn_parallel_iterations=32 --decoder_dropout=0.0 --decoder_type=transformer --transformer_num_layers=1 --transformer_d_model=5120 --transformer_num_heads=8 --transformer_dff=1024 --transformer_pe_target=512 --transformer_dropout_rate=0.0 --learning_rate=0.008 --model_dir=${MODELDIR}/gru512x1-tf5120x1-ff2048-dd0-lr08_goa_TITANRTX_8086K1-2.2
# Lower lr at step 41349
# export CUDA_VISIBLE_DEVICES=${CARDID}; ${PYTHON} ${TFSCRIPT}.py --training_data=${DATADIR}/${DATASET}-train.tfrecords --eval_data=${DATADIR}/${DATASET}-train.tfrecords --metadata_path=${DATADIR}/${DATASET}-meta.json --save_summary_steps=100 --log_step_count_steps=10 --learning_rate_decay_steps=180000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=900 --keep_checkpoint_max=24 --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --warmup_schedule=exp --warmup_steps=1300 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --batch_size=1 --gpu_allow_growth=False --encoder_embed_dim=32 --decoder_embed_dim=512 --encoder_num_units=[512] --decoder_num_units=[256,256,256] --attention_num_units=128 --decoder_num_residual_layers=2 --dynamic_rnn_parallel_iterations=32 --decoder_dropout=0.0 --decoder_type=transformer --transformer_num_layers=1 --transformer_d_model=5120 --transformer_num_heads=8 --transformer_dff=1024 --transformer_pe_target=512 --transformer_dropout_rate=0.0 --learning_rate=0.006 --model_dir=${MODELDIR}/gru512x1-tf5120x1-ff2048-dd0-lr08_goa_TITANRTX_8086K1-2.2
# diverged at step 134.1k
export CUDA_VISIBLE_DEVICES=${CARDID}; ${PYTHON} ${TFSCRIPT}.py --training_data=${DATADIR}/${DATASET}-train.tfrecords --eval_data=${DATADIR}/${DATASET}-train.tfrecords --metadata_path=${DATADIR}/${DATASET}-meta.json --save_summary_steps=100 --log_step_count_steps=10 --learning_rate_decay_steps=180000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=900 --keep_checkpoint_max=24 --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --warmup_schedule=exp --warmup_steps=1300 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --batch_size=1 --gpu_allow_growth=False --embedded_dropout=0.2 --conv_dropout=0.2 --encoder_dropout=0.0 --encoder_embed_dim=32 --decoder_embed_dim=512 --encoder_num_units=[512,512] --decoder_num_units=[256,256,256] --attention_num_units=128 --decoder_num_residual_layers=2 --dynamic_rnn_parallel_iterations=32 --decoder_dropout=0.0 --decoder_type=transformer --transformer_num_layers=2 --transformer_d_model=2048 --transformer_num_heads=8 --transformer_dff=1024 --transformer_pe_target=512 --transformer_dropout_rate=0.2 --learning_rate=0.006 --model_dir=${MODELDIR}/gru512x2-tf2048x2-ff1024-dd20-lr6_goa_TITANRTX_8086K1-2.2
