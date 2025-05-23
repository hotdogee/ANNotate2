# cd /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2
# chmod a+x ./run/train/goa-v2/*.sh
# ./run/train/goa-v2/4960X-3.sh
CARDID=2
HOSTSCRIPT=$(basename ${BASH_SOURCE%.*})
DATADIR=/data12/goa/goa-20191015
DATASET=goa-20191015-v2
TFSCRIPT=goa-v2
BATCHSIZE=1
TOTAL_INSTALLED_GPUS=3
CARDTYPE=1080Ti
# _1080Ti_4960X-3.3
MODELDIR=/data12/checkpoints/${DATASET}/${TFSCRIPT}
PYTHON=/home/hotdogee/venv/tf15/bin/python
export CUDA_VISIBLE_DEVICES=${CARDID}; ${PYTHON} ${TFSCRIPT}.py --training_data=${DATADIR}/${DATASET}-train.tfrecords --eval_data=${DATADIR}/${DATASET}-train.tfrecords --metadata_path=${DATADIR}/${DATASET}-meta.json --save_summary_steps=100 --log_step_count_steps=10 --learning_rate_decay_steps=180000 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=3600 --keep_checkpoint_max=24 --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --warmup_schedule=exp --warmup_steps=1300 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --batch_size=1 --gpu_allow_growth=False --encoder_embed_dim=32 --decoder_embed_dim=512 --encoder_num_units=[256,256] --decoder_num_units=[512,512] --attention_num_units=160 --decoder_num_residual_layers=2 --dynamic_rnn_parallel_iterations=32 --decoder_dropout=0.0 --learning_rate=0.06 --model_dir=${MODELDIR}/gru256x2-ba160-gru512x2-rl2-p32-lr60_goa_1080Ti_4960X-3.3
