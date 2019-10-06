#!/bin/bash
VERSIONS=(
# 1568346278
# 1568346287
# 1568346297
# 1568346306
1568346315
# 1568346325
# 1568346336
# 1568346347
# 1568346356
# 1568346366
# 1568346376
# 1568346383
# 1568346391
# 1568346397
# 1568346403
# 1567719465
)

for VERSION in "${VERSIONS[@]}"
do
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/annotate_pfam_batch_msgpack_to_per_seq_classification_performance.py --indir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_25000/ann31_${VERSION}_results_raw --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --ans /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.fa --output /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.ann31_${VERSION}.per_seq_perf.tsv --readers 24
done
