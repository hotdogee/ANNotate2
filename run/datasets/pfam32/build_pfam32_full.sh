# Make full dataset
/home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/datasets/pfam_regions_build.py --pfam_version 32.0 --test_split 0 --meta_file /home/hotdogee/datasets3/pfam-regions-d0-s0-meta.json --dataset_dir /home/hotdogee/datasets3/Pfam32.0
# W2125
# Preparing Dataset: pfam-regions-d0-s0
# Loading Protein Sequence Data from SOURCE: /home/hotdogee/datasets3/Pfam32.0/uniprot.gz
# 49586165246/49586165246==============================] - 1014s 0us/step
# Loaded 115,306,282 Sequences, Amino Acids Min 2, Median 272, Max 36,991, Total 38,869,784,058
# Loading Domain Region Data from SOURCE: /home/hotdogee/datasets3/Pfam32.0/Pfam-A.regions.uniprot.tsv.gz
# Parsing Pfam-A.regions.uniprot.tsv.gz
# 10643100349/10643100349==============================] - 551s 0us/step
# Loaded 17,929 Domains, Regions Min 1, Median 1,210, Max 1,820,703, Total 138,165,283
# Collecting Sequences Containing the Top 17929 Domains
# 17929/17929==============================] - 358s 20ms/step
# Collected 88,966,235 Sequences with 17,929 Domains
# Distributing Sequences to Training and Testing Sets with 0% Split
# 17929/17929==============================] - 79s 4ms/step
# Collected 88,966,232 Training Sequences and 0 Testing Sequences with 0 Domains
# Generating Domain Sequence Representation for TRAIN Dataset
