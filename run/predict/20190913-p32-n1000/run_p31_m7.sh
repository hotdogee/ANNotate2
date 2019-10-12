#!/bin/bash
DATADIR=/data12/datasets3
VERSIONS=(
# m7: FullGru512x4_hw512_rnndrop_TITANRTX_8086K1-1.2
1568363738
1568363743
1568363750
1568363756
1568363762
1568363768
1568363774
1568363780
1568363787
1568363793
1568363800
1568363807
1568363813
1568363819
1567787530
)
for VERSION in "${VERSIONS[@]}"
do
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_serving_v2.py --indir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_25000 --outdir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_25000/ann31_${VERSION}_results_raw --servers http://localhost:8501/v1/models/pfam/versions/${VERSION}:predict,http://localhost:8601/v1/models/pfam/versions/${VERSION}:predict,http://localhost:8701/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.33:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.33:8601/v1/models/pfam/versions/${VERSION}:predict --readers 12 --writers 12
done

for VERSION in "${VERSIONS[@]}"
do
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/annotate_pfam_batch_msgpack_to_tsv_v2.py --indir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_25000/ann31_${VERSION}_results_raw --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --output /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.ann31_${VERSION}_results.tsv --workers 24
done

for VERSION in "${VERSIONS[@]}"
do
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/compute_annotate_pfam_classification_performance.py  --pred /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.ann31_${VERSION}_results.tsv --ans /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.fa --output /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.ann31_${VERSION}_results.all_regions_perf.json --outtsv /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.ann31_results.all_regions_perf.tsv
done

for VERSION in "${VERSIONS[@]}"
do
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/compute_annotate_pfam_classification_performance.py  --pred /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.ann31_${VERSION}_results.tsv --ans /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --fasta /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.fa --output /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.ann31_${VERSION}_results.p32_regions_perf.json --outtsv /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.ann31_results.p32_regions_perf.tsv
done
