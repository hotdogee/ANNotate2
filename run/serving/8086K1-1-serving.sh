# Starting a GPU enabled container on specific GPUs
$ docker run --gpus '"device=0"' nvidia/cuda:9.0-base nvidia-smi
$ docker run --gpus '"device=0"' -p 8501:8501 --mount type=bind,source=/home/hotdogee/models,target=/models -t tensorflow/serving:latest-gpu --model_config_file=/models/models_config1.proto --file_system_poll_wait_seconds=60

