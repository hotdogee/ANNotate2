# sudo firewall-cmd --list-all
# sudo firewall-cmd --permanent --add-port=6006/tcp
# sudo firewall-cmd --reload
# /home/hotdogee/build/tensorboard/bazel-bin/tensorboard/tensorboard --logdir /data12/checkpoints/pfam-regions-d0-s20/v4-BiRnn --port 6006
# /home/hotdogee/build/tensorboard/bazel-bin/tensorboard/tensorboard --logdir /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn --port 6007 --samples_per_plugin=scalars=100000
# /home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --logdir /data12/checkpoints/pfam-regions-d0-s20/v4-BiRnn --port 6006 --samples_per_plugin=scalars=10000
/home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --logdir /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn --port 6007 --samples_per_plugin=scalars=1000
/home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --logdir /data12/checkpoints/pfam32-regions-d0-s0/v5-BiRnn --port 6008 --samples_per_plugin=scalars=1000
/home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --logdir /data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn --port 6009 --samples_per_plugin=scalars=1000
/home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --logdir /data12/checkpoints/goa-20191015/goa-v1/ --port 6010 --samples_per_plugin=scalars=1000

sudo pm2 start /home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --name tensorboard-6008 --max-memory-restart 10G --interpreter /home/hotdogee/venv/tf15/bin/python -- --logdir /data12/checkpoints/pfam32-regions-d0-s0/v5-BiRnn --port 6008 --samples_per_plugin=scalars=1000

sudo pm2 start /home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --interpreter /home/hotdogee/venv/tf15/bin/python --name tensorboard-6010 --max-memory-restart 10G -- --logdir /data12/checkpoints/goa-20191015/goa-v1/ --port 6010 --samples_per_plugin=scalars=1000
