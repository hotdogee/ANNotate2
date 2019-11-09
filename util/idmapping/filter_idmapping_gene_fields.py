import os
import re
import csv
import time
import gzip
import errno
import struct
import argparse
from tqdm import tqdm
from glob import glob
from pathlib import Path
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict
from ..verify_paths import verify_input_path, verify_output_path, verify_indir_path, verify_outdir_path

# Download: ftp://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/idmapping/idmapping.dat.gz
# 1. UniProtKB-AC
# 2. ID_type
# 3. ID
# Q6GZX4	UniProtKB-ID	001R_FRG3G
# Q6GZX4	Gene_ORFName	FV3-001R
# Q6GZX4	GI	81941549
# Q6GZX4	GI	49237298
# Q6GZX4	UniRef100	UniRef100_Q6GZX4
# Q6GZX4	UniRef90	UniRef90_Q6GZX4
# Q6GZX4	UniRef50	UniRef50_Q6GZX4
# Q6GZX4	UniParc	UPI00003B0FD4
# Q6GZX4	EMBL	AY548484
# Q6GZX4	EMBL-CDS	AAT09660.1
# Q6GZX4	NCBI_TaxID	654924
# Q6GZX4	RefSeq	YP_031579.1ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
# Q6GZX4	RefSeq_NT	NC_005946.1
# Q6GZX4	GeneID	2947773
# Q6GZX4	KEGG	vg:2947773
# Q6GZX4	CRC64	B4840739BF7D4121


def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def filter_idmapping_gene_fields(path, output_path):
    # Gene_Name
    # Gene_ORFName
    # Gene_OrderedLocusName
    # Gene_Synonym
    gene_fields = {
        'Gene_Name', 'Gene_ORFName', 'Gene_OrderedLocusName', 'Gene_Synonym'
    }
    lines_in = 0
    lines_out = 0
    # if gzipped
    target = os.path.getsize(path)
    if path.suffix == '.gz':
        f = gzip.open(path, mode='rt', encoding='utf-8')
        target = _gzip_size(path)
        while target < os.path.getsize(path):
            # the uncompressed size can't be smaller than the compressed size, so add 4GB
            target += 2**32
    else:
        f = path.open(mode='r', encoding='utf-8')
    if output_path.suffix == '.gz':
        fo = gzip.open(output_path, mode='wt', encoding='utf-8')
    else:
        fo = output_path.open(mode='w', encoding='utf-8')
    with f, fo, tqdm(
        total=target,
        dynamic_ncols=True,
        ascii=True,
        desc=path.name,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as t:
        for line in f:
            lines_in += 1
            t.update(len(line))
            if t.n > t.total:
                t.total += 2**32
            line_s = line.strip()
            if not line_s:
                continue
            tokens = line_s.split('\t')
            id_type = tokens[1].strip()
            if id_type in gene_fields:
                lines_out += 1
                fo.write(line)
    return lines_in, lines_out


if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description='Filter idmapping Gene_* fields.'
    )
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=True,
        help="Path to idmapping.dat[.gz] file, required."
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help="Path to idmapping_filtered.dat[.gz] file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    input_path = verify_input_path(args.input)
    output_path = verify_output_path(args.output)
    print(f'Processing: {input_path.name}')

    lines_in, lines_out = filter_idmapping_gene_fields(input_path, output_path)
    print(f'Lines in: {lines_in}')
    print(f'Lines out: {lines_out}')

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# windows
# python -m util.idmapping.filter_idmapping_gene_fields --input E:\idmapping\idmapping-20191105\idmapping.dat --output E:\idmapping\idmapping-20191105\idmapping_filtered.dat

# idmapping.dat: 2,442,456,179 lines [59:43, 681539 lines/s]
# Lines in: 2442456179
# Lines out: 232575355
# Run time: 3503.51 s
# Run time: 3583.82 s
