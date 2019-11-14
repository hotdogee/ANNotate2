import os
import re
import csv
import time
import gzip
import json
import errno
import struct
import argparse
from tqdm import tqdm
from glob import glob
from pathlib import Path
from pprint import pprint
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict
from ..verify_paths import verify_input_path, verify_output_path, verify_indir_path, verify_outdir_path

# Download: https://www.disprot.org/download
# acc	name	organism	ncbi_taxon_id	disprot_id	region_id	start	end	term_namespace	term	ec	reference	region_sequence	confidence	obsolete
# P03265	DNA-binding protein	Human adenovirus C serotype 5	28285	DP00003	DP00003r002	294	334	Structural state	DO:00076	DO:00130	pmid:8632448	EHVIEMDVTSENGQRALKEQSSKAKIVKNRWGRNVVQISNT
# P03265	DNA-binding protein	Human adenovirus C serotype 5	28285	DP00003	DP00003r003	294	334	Disorder function	DO:00002	DO:00130	pmid:8632448	EHVIEMDVTSENGQRALKEQSSKAKIVKNRWGRNVVQISNT
# P03265	DNA-binding protein	Human adenovirus C serotype 5	28285	DP00003	DP00003r004	454	464	Structural state	DO:00076	DO:00130	pmid:8632448	VYRNSRAQGGG


def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def parse_do_tsv(path, target=None):
    acc_do = defaultdict(set)
    # if gzipped
    if path.suffix == '.gz':
        f = gzip.open(path, mode='rt', encoding='utf-8')
    else:
        f = path.open(mode='r', encoding='utf-8')
    with f, tqdm(
        total=target,
        dynamic_ncols=True,
        ascii=True,
        desc=path.name,
        unit='lines'
    ) as t:
        rows = csv.reader(f, delimiter='\t')
        for row in rows:
            t.update()
            acc = row[0].strip().upper()
            do = row[9].strip().upper()
            acc_do[acc].add(do)
    return acc_do


def print_dict_set_stats(dict_set, dict_name, key_name, value_name):
    dict_set_list = sorted(dict_set, key=lambda k: (len(dict_set[k]), k))
    reversed_dict_set = defaultdict(set)
    for k in dict_set:
        for v in dict_set[k]:
            reversed_dict_set[v].add(k)
    reversed_dict_set_list = sorted(
        reversed_dict_set, key=lambda k: (len(reversed_dict_set[k]), k)
    )
    print(
        f'''=== {dict_name} ===
 - {key_name}s: {len(dict_set)}
 - {value_name}s: {len(reversed_dict_set)}
 - {value_name}s per {key_name}: (Min, Median, Max): {len(dict_set[dict_set_list[0]])}, {len(dict_set[dict_set_list[len(dict_set_list)//2]])}, {len(dict_set[dict_set_list[-1]])}
 - {key_name}s per {value_name}: (Min, Median, Max): {len(reversed_dict_set[reversed_dict_set_list[0]])}, {len(reversed_dict_set[reversed_dict_set_list[len(reversed_dict_set_list)//2]])}, {len(reversed_dict_set[reversed_dict_set_list[-1]])}
'''
    )
    del dict_set_list
    del reversed_dict_set
    del reversed_dict_set_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Print accession disorder mapping stats.'
    )
    parser.add_argument(
        '-d',
        '--do_tsv',
        type=str,
        required=True,
        help="Path to do-20191101.tsv file, required."
    )
    # parser.add_argument(
    #     '-o',
    #     '--out',
    #     type=str,
    #     required=True,
    #     help="Path to gene_acc_phenotype.json file, required."
    # )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    do_tsv_path = verify_input_path(args.do_tsv)
    print(f'Processing: {do_tsv_path.name}')

    acc_do = parse_do_tsv(do_tsv_path)
    print_dict_set_stats(acc_do, 'acc_do', 'Accession', 'DO')

    # expand number of sequences
    # do -> acc -> gene -> acc

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# 1920X
# python -m util.do.make_acc_disorder --do_tsv E:\do\do-20191101\do-20191101.tsv

# do-20191101.tsv: 6220lines [00:00, 163335.32lines/s]
# === acc_do ===
#  - Accessions: 1391
#  - DOs: 57
#  - DOs per Accession: (Min, Median, Max): 1, 2, 8
#  - Accessions per DO: (Min, Median, Max): 1, 8, 1377

# Run time: 0.07 s
#
