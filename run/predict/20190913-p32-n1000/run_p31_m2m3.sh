#!/bin/bash
# chmod a+x /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/predict/20190913-p32-n1000/*.sh
DATADIR=/data12/datasets3
VERSIONS=(
# FullGru512x4_hw512_attpost_2080Ti_W2125-4.4
1568424545
1568424552
1568424557
1568424563
1568424570
1568424576
# FullGru512x4_hw512_attpost_highdrop_2080Ti_W2125-3.4
1568424583
1568424589
1568424596
1568424602
1568424609
1568424615
1568424622
1568424629
1568424635
1568424642
1568424648
1568424655
1568424662
1567786205
# # FullGru512x4_hw512_apo_hd_lr0.055_ws33k_1080Ti_8086K2-1.2
# 1568424669
# 1568424675
# 1568424682
# 1568424689
# 1568424695
# 1568424702
# 1568424709
# 1568424715
# 1568424722
# 1568424729
# 1568424735
# 1568424742
# 1568424749
# 1567786591
# # FullGru512x4_hw512_apo_hd_lr0.075_ws18k_2080Ti_W2125-2.4
# 1568424755
# 1568424762
# 1568424768
# 1568424775
# 1568424782
# 1568424789
# 1568424795
# 1568424802
# 1567786835
# 1568424808
# # FullGru512x4_hw512_apo_hd_lr0.035_ws27k_2080Ti_W2125-1.4
# 1568424815
# 1568424822
# 1568424828
# 1567787010
# # FullGru512x4_hw512_apo_hd_lr0.045_ws30k_1080Ti_4960X-2.2
# 1568424835
)
for VERSION in "${VERSIONS[@]}"
do
    /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_serving_v2.py --indir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_25000 --outdir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_25000/ann31_${VERSION}_results_raw --servers http://localhost:8501/v1/models/pfam/versions/${VERSION}:predict,http://localhost:8601/v1/models/pfam/versions/${VERSION}:predict,http://localhost:8701/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.33:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.33:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.34:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.34:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.74:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.74:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8701/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8801/v1/models/pfam/versions/${VERSION}:predict --readers 12 --writers 12
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
