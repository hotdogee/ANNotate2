# Make full dataset
python datasets/pfam_regions_build.py -s 0 -d D:/datasets3
# Preparing Dataset: pfam-regions-d0-s20
# Loading Protein Sequence Data from CACHE: D:\datasets3\uniprot.msgpack - 112s
# Loaded 71,201,428 Sequences, Amino Acids Min 2, Median 267, Max 36,805, Total 23,867,549,122
# Loading Domain Region Data from CACHE: D:\datasets3\Pfam-A.regions.uniprot.tsv.msgpack - 107s
# Loaded 16,712 Domains, Regions Min 2, Median 863, Max 1,078,482, Total 88,761,542
# Collecting Sequences Containing the Top 16712 Domains
# 16712/16712 [==============================] - 138s 8ms/step
# Collected 54,223,493 Sequences with 16,712 Domains
# Distributing Sequences to Training and Testing Sets with 20% Split
#  WARNING: 1357=PF02980, seq_count=55, test_count=-1, train_count=1, already in test_set= 12, already in train_set= 43
#  WARNING: 2515=PF16633, seq_count=110, test_count=-1, train_count=5, already in test_set= 23, already in train_set= 83
#  WARNING: 2550=PF05937, seq_count=113, test_count=-1, train_count=7, already in test_set= 24, already in train_set= 83
#  WARNING: 5865=PF16665, seq_count=420, test_count=-3, train_count=22, already in test_set= 87, already in train_set= 314
#  WARNING: 6396=PF08832, seq_count=489, test_count=6, train_count=-2, already in test_set= 92, already in train_set= 393
#  WARNING: 8366=PF16749, seq_count=822, test_count=-1, train_count=2, already in test_set= 166, already in train_set= 655
#  WARNING: 10199=PF17192, seq_count=1258, test_count=4, train_count=-2, already in test_set= 248, already in train_set= 1008
#  WARNING: 10748=PF05412, seq_count=1451, test_count=-8, train_count=14, already in test_set= 299, already in train_set= 1146
#  WARNING: 11346=PF08717, seq_count=1738, test_count=-3, train_count=5, already in test_set= 351, already in train_set= 1385
#  WARNING: 13745=PF14527, seq_count=4807, test_count=-3, train_count=15, already in test_set= 965, already in train_set= 3830
# 16712/16712 [==============================] - 94s 6ms/step
# Collected 43,373,260 Training Sequences and 10,850,231 Testing Sequences with 16,711 Domains
# Generating Domain Sequence Representation for TRAIN Dataset
# 43373260/43373260 [==============================] - 3549s 82us/step
# Generating Domain Sequence Representation for TEST Dataset
# 10850231/10850231 [==============================] - 1306s 120us/step
# Writing Metadata: D:\datasets3\pfam-regions-d0-s20-meta.json ... DONE
# Writing D:\datasets3\pfam-regions-d0-s20-train.tfrecords
# 43373260/43373260 [==============================] - 321323s 7ms/step
# Writing D:\datasets3\pfam-regions-d0-s20-test.tfrecords
# 10850231/10850231 [==============================] - 80990s 7ms/step
