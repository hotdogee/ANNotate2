# cd /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2
# chmod a+x ./run/train/goa-v2/*.sh
# source /opt/intel/parallel_studio_xe_2019/bin/psxevars.sh
# ./run/train/goa-v2/W2125-2.sh
CARDID=1
HOSTSCRIPT=$(basename ${BASH_SOURCE%.*})
DATADIR=/data12/goa/goa-20191015
DATASET=goa-20191015-v2
TFSCRIPT=goa-v2
BATCHSIZE=1
TOTAL_INSTALLED_GPUS=4
CARDTYPE=2080Ti
# _2080Ti_W2125-2.4
MODELDIR=/data12/checkpoints/${DATASET}/${TFSCRIPT}
PYTHON=/home/hotdogee/venv/tf15/bin/python
export CUDA_VISIBLE_DEVICES=${CARDID}; ${PYTHON} ${TFSCRIPT}.py --training_data=${DATADIR}/${DATASET}-train.tfrecords --eval_data=${DATADIR}/${DATASET}-train.tfrecords --metadata_path=${DATADIR}/${DATASET}-meta.json --save_summary_steps=100 --log_step_count_steps=10 --learning_rate_decay_steps=350000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --warmup_schedule=exp --warmup_steps=1300 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --batch_size=1 --gpu_allow_growth=False --encoder_embed_dim=32 --decoder_embed_dim=512 --encoder_num_units=[256,256,256,256] --decoder_num_units=[256,256,256] --attention_num_units=128 --decoder_num_residual_layers=2 --dynamic_rnn_parallel_iterations=32 --decoder_dropout=0.0 --decoder_type=transformer --transformer_num_layers=4 --transformer_d_model=256 --transformer_num_heads=8 --transformer_dff=1024 --transformer_pe_target=512 --transformer_dropout_rate=0.2 --learning_rate=0.12 --model_dir=${MODELDIR}/gru256x4-tr256x4-lr120_goa_2080Ti_W2125-2.4
