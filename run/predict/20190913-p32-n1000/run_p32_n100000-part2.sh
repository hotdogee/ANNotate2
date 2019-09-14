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
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/annotate_pfam_batch_msgpack_to_tsv_v2.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n100000_fa_batched_25000/ann31_${VERSION}_results_raw --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n100000.ann31_${VERSION}_results.tsv --workers 24
done
# /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/predict/20190913-p32-n1000/run_p32_n1000.sh
