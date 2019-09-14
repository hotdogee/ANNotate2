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
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/compute_annotate_pfam_classification_performance.py  --pred /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000.ann31_${VERSION}_results.tsv --ans /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n10000.tsv --fasta /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n10000.fa --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000.ann31_${VERSION}_results.all_regions_perf.json --outtsv /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000.ann31_results.all_regions_perf.tsv
done
# /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/predict/20190913-p32-n1000/run_p32_n1000.sh
