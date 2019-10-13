#!/bin/bash
# run on 4960X /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/predict/20191013-512x8/run_p31_m10e1.sh
VERSIONS=( # total 10
# m10: Gru512x8_Pfam31_TITANRTX_8086K1-2.2
1570954345
1570954352
1570954359
1570954367
1570954374
1570954382
1570954390
1570954397
1570954405
1570954413
)
for VERSION in "${VERSIONS[@]}"
do
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_serving_v2.py --indir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_25000 --outdir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_25000/ann31_${VERSION}_results_raw --servers http://localhost:8501/v1/models/pfam/versions/${VERSION}:predict,http://localhost:8601/v1/models/pfam/versions/${VERSION}:predict,http://localhost:8701/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8701/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.74:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.74:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.33:8601/v1/models/pfam/versions/${VERSION}:predict --readers 12 --writers 12
done

for VERSION in "${VERSIONS[@]}"
do
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/annotate_pfam_batch_msgpack_to_tsv_v3.py --indir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_25000/ann31_${VERSION}_results_raw --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --output /home/hotdogee/pfam/ann_post_v3/p31_seqs_with_p32_regions_of_p31_domains.ann31_${VERSION}_results_v3.tsv --workers 24
done

for VERSION in "${VERSIONS[@]}"
do
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/compute_annotate_pfam_classification_performance.py --pred /home/hotdogee/pfam/ann_post_v3/p31_seqs_with_p32_regions_of_p31_domains.ann31_${VERSION}_results_v3.tsv --ans /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.fa --output /home/hotdogee/pfam/ann_post_v3/p31_seqs_with_p32_regions_of_p31_domains.ann31_${VERSION}_results_v3.all_regions_perf.json --outtsv /home/hotdogee/pfam/ann_post_v3/p31_seqs_with_p32_regions_of_p31_domains.ann31_results_v3.all_regions_perf.tsv
done

for VERSION in "${VERSIONS[@]}"
do
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/compute_annotate_pfam_classification_performance.py --pred /home/hotdogee/pfam/ann_post_v3/p31_seqs_with_p32_regions_of_p31_domains.ann31_${VERSION}_results_v3.tsv --ans /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --fasta /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.fa --output /home/hotdogee/pfam/ann_post_v3/p31_seqs_with_p32_regions_of_p31_domains.ann31_${VERSION}_results_v3.p32_regions_perf.json --outtsv /home/hotdogee/pfam/ann_post_v3/p31_seqs_with_p32_regions_of_p31_domains.ann31_results_v3.p32_regions_perf.tsv
done

# http://localhost:8501/v1/models/pfam/versions/${VERSION}:predict,
# http://localhost:8601/v1/models/pfam/versions/${VERSION}:predict,
# http://localhost:8701/v1/models/pfam/versions/${VERSION}:predict,
# http://192.168.1.33:8501/v1/models/pfam/versions/${VERSION}:predict,
# http://192.168.1.33:8601/v1/models/pfam/versions/${VERSION}:predict,
# http://192.168.1.34:8501/v1/models/pfam/versions/${VERSION}:predict,
# http://192.168.1.34:8601/v1/models/pfam/versions/${VERSION}:predict,
# http://192.168.1.74:8501/v1/models/pfam/versions/${VERSION}:predict,
# http://192.168.1.74:8601/v1/models/pfam/versions/${VERSION}:predict,
# http://192.168.1.35:8501/v1/models/pfam/versions/${VERSION}:predict,
# http://192.168.1.35:8601/v1/models/pfam/versions/${VERSION}:predict,
# http://192.168.1.35:8701/v1/models/pfam/versions/${VERSION}:predict,
# http://192.168.1.35:8801/v1/models/pfam/versions/${VERSION}:predict
