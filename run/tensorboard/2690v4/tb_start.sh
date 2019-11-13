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
/home/hotdogee/venv/tf15/bin/python /home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --logdir /data12/checkpoints/goa-20191015-v2/goa-v2/ --port 6011 --samples_per_plugin=scalars=1000

# can’t start with root
source tf15.sh
pm2 start /home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --interpreter=/home/hotdogee/venv/tf37/bin/python --max-memory-restart 10G --name tb-6007-pfamfull -- --logdir /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn --port 6007 --samples_per_plugin=scalars=1000

pm2 start /home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --interpreter /home/hotdogee/venv/tf15/bin/python --max-memory-restart 10G --name tb-6008-pfam32 -- --logdir /data12/checkpoints/pfam32-regions-d0-s0/v5-BiRnn --port 6008 --samples_per_plugin=scalars=1000

pm2 start /home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --interpreter /home/hotdogee/venv/tf15/bin/python --max-memory-restart 10G --name tb-6009-pfam31 -- --logdir /data12/checkpoints/pfam-regions-d0-s0/v5-BiRnn --port 6009 --samples_per_plugin=scalars=1000

pm2 start /home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --interpreter /home/hotdogee/venv/tf15/bin/python --max-memory-restart 10G --name tb-6010-go -- --logdir /data12/checkpoints/goa-20191015/goa-v1/ --port 6010 --samples_per_plugin=scalars=1000

pm2 start /home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --interpreter /home/hotdogee/venv/tf15/bin/python --max-memory-restart 20G --name tb-6011-go2 -- --logdir /data12/checkpoints/goa-20191015-v2/goa-v2/ --port 6011 --samples_per_plugin=scalars=1000

pm2 start /home/hotdogee/build/tb-hotdogee/bazel-bin/tensorboard/tensorboard --interpreter /home/hotdogee/venv/tf15/bin/python --max-memory-restart 10G --name tb-6012-hpo -- --logdir /data12/checkpoints/hpo-20191011-seq2seq/hpo-v1/ --port 6012 --samples_per_plugin=scalars=1000
