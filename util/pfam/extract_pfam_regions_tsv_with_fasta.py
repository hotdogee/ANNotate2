import os
import re
import sys
import time
import gzip
import errno
import random
import struct
import logging
import argparse
from tqdm import tqdm
from glob import glob
from pathlib import Path
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def _fa_gz_to_id_set(fa_path):
    """Parse a .fa or .fa.gz file into a set of seq_id
    """
    path = Path(fa_path)
    # if gzipped
    target = os.path.getsize(path)
    if path.suffix == '.gz':
        f = gzip.open(path, mode='rt', encoding='utf-8')
        target = _gzip_size(path)
        while target < os.path.getsize(path):
            # the uncompressed size can't be smaller than the compressed size, so add 4GB
            target += 2**32
    else:
        f = path.open(mode='rt', encoding='utf-8')
    # initialize
    id_set = set()
    seq_id = ''
    # current = 0
    with f as fa_f, tqdm(
        total=target, unit='bytes', dynamic_ncols=True, ascii=True
    ) as t:
        for line in fa_f:
            t.update(len(line))
            line_s = line.strip()
            if len(line_s) > 0 and line_s[0] == '>':
                if seq_id:
                    id_set.add(seq_id)
                    seq_id = ''
                # parse header
                seq_id = line_s.split()[0][1:]
        if seq_id:  # handle last seq
            id_set.add(seq_id)
    return id_set


def verify_input_path(p):
    # get absolute path to dataset directory
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # doesn't exist
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    # is dir
    if path.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
    return path


def verify_output_path(p):
    # get absolute path to dataset directory
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # existing file
    if path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.EEXIST), path)
    # is dir
    if path.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
    # assert dirs
    path.parent.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
    return path


# windows

# python .\util\pfam\extract_pfam_regions_tsv_with_fasta.py  --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.fa.gz --input D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.n100.tsv.gz

# python .\util\pfam\extract_pfam_regions_tsv_with_fasta.py  --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.fa.gz --input D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.n100.tsv.gz

# python .\util\pfam\extract_pfam_regions_tsv_with_fasta.py  --fasta D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.n100.fa.gz --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --output D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n100.tsv.gz

# fix last seq bug
# python .\util\pfam\extract_pfam_regions_tsv_with_fasta.py  --fasta D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.n100.fa.gz --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --output D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.all_regions.n100.tsv.gz

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract pfam regions for sequences in a FASTA file.'
    )
    parser.add_argument(
        '-fa',
        '--fasta',
        type=str,
        required=True,
        help="Path to the input FASTA file, required."
    )
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=True,
        help="Path to the input TSV file, required."
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help="Path to the output TSV file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    # verify paths
    fasta_path = verify_input_path(args.fasta)
    input_path = verify_input_path(args.input)
    output_path = verify_output_path(args.output)
    logging.info(
        f"Extracting pfam regions from {'/'.join(input_path.parts[-2:])} with fasta file {'/'.join(fasta_path.parts[-2:])} into {'/'.join(output_path.parts[-2:])}"
    )

    # read fasta
    id_set = _fa_gz_to_id_set(fasta_path)
    seq_count = len(id_set)
    logging.info(
        f"Parsed {seq_count} seqs from {'/'.join(fasta_path.parts[-2:])}"
    )

    # filter tsv with id_set
    # output_path
    if output_path.suffix == '.gz':
        out_f = gzip.open(output_path, mode='wt', encoding='utf-8')
    else:
        out_f = output_path.open(mode='wt', encoding='utf-8')

    # input_path
    target = os.path.getsize(input_path)
    if input_path.suffix == '.gz':
        in_f = gzip.open(input_path, mode='rt', encoding='utf-8')
        target = _gzip_size(input_path)
        while target < os.path.getsize(input_path):
            # the uncompressed size can't be smaller than the compressed size, so add 4GB
            target += 2**32
    else:
        in_f = input_path.open(mode='rt', encoding='utf-8')

    # initialize
    line_num = 0
    region_count = 0

    # process input by line
    with in_f, out_f, tqdm(
        total=target, unit='bytes', dynamic_ncols=True, ascii=True
    ) as t:
        for line in in_f:
            t.update(len(line))
            line_num += 1
            if line_num == 1:  # header
                out_f.write(line)
            # parse
            tokens = line.strip().split()
            ua_sv = f'{tokens[0]}.{tokens[1]}'
            if ua_sv in id_set:
                region_count += 1
                out_f.write(line)

    print(f'Regions: {region_count}')
    print(f'Runtime: {time.time() - start_time:.2f} s')
    sys.exit(0)

# 2019-09-04 00:38:58 INFO     Extracting pfam regions from Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv with fasta file Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.fa.gz into Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.n100.tsv.gz
# 100%|#################################################################7| 65725/65925 [00:00<00:00, 32681758.20bytes/s]
# 2019-09-04 00:38:58 INFO     Parsed 100 seqs from Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.fa.gz
#  99%|###########################################################2| 52761399/53447360 [00:01<00:00, 34954476.01bytes/s]
# Regions: 346
# Runtime: 1.54 s

# 2019-09-04 00:39:07 INFO     Extracting pfam regions from Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv with fasta file Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.fa.gz into Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.n100.tsv.gz
# 100%|#################################################################7| 65725/65925 [00:00<00:00, 64757019.12bytes/s]
# 2019-09-04 00:39:07 INFO     Parsed 100 seqs from Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.fa.gz
#  99%|###########################################################2| 18411827/18651856 [00:00<00:00, 34158309.92bytes/s]
# Regions: 123
# Runtime: 0.56 s

# 2019-09-04 04:04:09 INFO     Extracting pfam regions from Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv with fasta file Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.n100.fa.gz into Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.all_regions.n100.tsv.gz
# 100%|#################################################################7| 48619/48819 [00:00<00:00, 48634120.24bytes/s]
# 2019-09-04 04:04:09 INFO     Parsed 100 seqs from Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.n100.fa.gz
#  99%|#######################################################2| 4629155666/4688421289 [02:08<00:01, 36165204.81bytes/s]
# Regions: 170
# Runtime: 128.04 s
