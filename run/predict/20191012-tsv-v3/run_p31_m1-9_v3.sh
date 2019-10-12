#!/bin/bash
# run on 4960X ./run/predict/20191012-tsv-v3/run_p31_m1-9_v3.sh
DATADIR=/data12/datasets3
MODELDIR=/data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_TITANV_W2125-2.4
VERSIONS=( # total 98
# m1: FullGru512x4_hw512_TITANV_W2125-2.4
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
# m2: FullGru512x4_hw512_attpost_2080Ti_W2125-4.4
1568424545
1568424552
1568424557
1568424563
1568424570
1568424576
# m3: FullGru512x4_hw512_attpost_highdrop_2080Ti_W2125-3.4
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
# m4: FullGru512x4_hw512_apo_hd_lr0.055_ws33k_1080Ti_8086K2-1.2
1568424669
1568424675
1568424682
1568424689
1568424695
1568424702
1568424709
1568424715
1568424722
1568424729
1568424735
1568424742
1568424749
1567786591
# m5: FullGru512x4_hw512_apo_hd_lr0.075_ws18k_2080Ti_W2125-2.4
1568424755
1568424762
1568424768
1568424775
1568424782
1568424789
1568424795
1568424802
1567786835
1568424808
# m6: FullGru512x4_hw512_apo_hd_lr0.035_ws27k_2080Ti_W2125-1.4
1568424815
1568424822
1568424828
1567787010
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
# m8: FullGru512x4_hw512_attpost_rnndrop_TITANRTX_8086K1-2.2
1568367500
1568367510
1568367520
1568367531
1568367542
1568367553
1568367564
1568367575
1568367586
1568367598
1568367609
1568367620
1568367631
1568367642
1568367654
1568367665
1568367676
1567787563
# m9: FullGru512x4_hw512_apo_hd_lr0.045_ws30k_1080Ti_4960X-2.2
1568424835
)
# for VERSION in "${VERSIONS[@]}"
# do
#     /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_serving_v2.py --indir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_25000 --outdir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_25000/ann31_${VERSION}_results_raw --servers http://localhost:8501/v1/models/pfam/versions/${VERSION}:predict,http://localhost:8601/v1/models/pfam/versions/${VERSION}:predict,http://localhost:8701/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.33:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.33:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.34:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.34:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.74:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.74:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8701/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8801/v1/models/pfam/versions/${VERSION}:predict --readers 12 --writers 12
# done

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
