#!/bin/bash
DATADIR=/data12/datasets3
# FullGru512x4_hw512_TITANV_W2125-2.4
MODELDIR=/data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_TITANV_W2125-2.4
VERSIONS=(
1568346278
1568346287
1568346297
1568346306
1568346315
1568346325
1568346336
1568346347
1568346356
1568346366
1568346376
1568346383
1568346391
1568346397
1568346403
1567719465
)
for VERSION in "${VERSIONS[@]}"
do
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_serving_v2.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_batched_25000 --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_batched_25000/ann31_${VERSION}_results_raw --servers http://localhost:8501/v1/models/pfam/versions/${VERSION}:predict,http://localhost:8601/v1/models/pfam/versions/${VERSION}:predict,http://localhost:8701/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.33:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.33:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.34:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.34:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.74:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.74:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8701/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8801/v1/models/pfam/versions/${VERSION}:predict --readers 12 --writers 12
done
# /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/predict/20190913-p32-n1000/run_p32_n1000.sh
