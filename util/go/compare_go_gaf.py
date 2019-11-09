import os
import re
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

# GAF2.1 files have the suffix .gaf and contain the following columns:
# 	Column  Contents
# 	1       DB
# 	2       DB_Object_ID
# 	3       DB_Object_Symbol
# 	4       Qualifier
# 	5       GO_ID
# 	6       DB:Reference
# 	7       Evidence Code
# 	8       With (or) From
# 	9       Aspect
# 	10      DB_Object_Name
# 	11      DB_Object_Synonym
# 	12      DB_Object_Type
# 	13      Taxon and Interacting taxon
# 	14      Date
# 	15      Assigned_By
# 	16      Annotation_Extension
# 	17      Gene_Product_Form_ID


def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def stats_go_gaf(path):
    path = Path(path)
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
    # initialize
    vocab = defaultdict(set)
    line_num = 0
    with f, tqdm(
        total=target,
        unit='bytes',
        dynamic_ncols=True,
        ascii=True,
        desc=path.name
    ) as t:
        for line in f:
            t.update(len(line))
            line_num += 1
            # empty line
            line = line.strip()
            if line == '':
                continue
            # comment
            if line[0] == '!':
                continue
            # line code
            tokens = line.split('\t')
            vocab['db'].add(tokens[0].strip())
            vocab['seqid'].add(tokens[1].strip())
            vocab['symbol'].add(tokens[2].strip())
            vocab['qual'].add(tokens[3].strip())
            vocab['go'].add(tokens[4].strip())
            vocab['evidence'].add(tokens[6].strip())
            vocab['type'].add(tokens[11].strip())
            vocab['tax'].add(tokens[12].strip())
    return vocab, line_num


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


def print_set_stats(n1, s1, n2, s2, unit=''):
    print(
        f'''
{n1}: {len(s1)} {unit}
{n2}: {len(s2)} {unit}
{n1} & {n2}: {len(s1 & s2)} {unit}
{n1} | {n2}: {len(s1 | s2)} {unit}
{n1} - {n2}: {len(s1 - s2)} {unit}
{n2} - {n1}: {len(s2 - s1)} {unit}
'''
    )


if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description='Print the differences of two GO gaf files.'
    )
    parser.add_argument(
        '-1',
        '--gaf1',
        type=str,
        required=True,
        help="Path to the first GO gaf file, required."
    )
    parser.add_argument(
        '-2',
        '--gaf2',
        type=str,
        required=True,
        help="Path to the second GO gaf file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    gaf1_path = verify_input_path(args.gaf1)
    gaf2_path = verify_input_path(args.gaf2)
    print(f'Comparing: {gaf1_path.name} and {gaf2_path.name}')

    vocab1, line_num1 = stats_go_gaf(gaf1_path)
    print(f'Processed {line_num1} lines in {gaf1_path.name}')
    vocab2, line_num2 = stats_go_gaf(gaf2_path)
    print(f'Processed {line_num2} lines in {gaf2_path.name}')

    for sid in vocab1.keys():
        print(f'== {sid} set stats ==')
        print_set_stats(
            gaf1_path.name, vocab1[sid], gaf2_path.name, vocab2[sid]
        )

    print(f'Runtime: {time.time() - start_time:.2f} s\n')

# Conclusion
# goa_uniprot_gcrp.gaf is a subset of goa_uniprot_all.gaf

# windows
# python .\util\go\compare_go_gaf.py --gaf1 D:/go/goa/UNIPROT/goa_uniprot_all.gaf --gaf2 D:/go/goa/UNIPROT/goa_uniprot_gcrp.gaf
# W2125
# python util/go/compare_go_gaf.py --gaf1 /data12/goa/goa_uniprot_all.gaf --gaf2 /data12/goa/goa_uniprot_gcrp.gaf

# Processed 608472856 lines in goa_uniprot_all.gaf
# Processed 137894383 lines in goa_uniprot_gcrp.gaf
# == db set stats ==

# goa_uniprot_all.gaf: 3
# goa_uniprot_gcrp.gaf: 1
# goa_uniprot_all.gaf & goa_uniprot_gcrp.gaf: 1
# goa_uniprot_all.gaf | goa_uniprot_gcrp.gaf: 3
# goa_uniprot_all.gaf - goa_uniprot_gcrp.gaf: 2
# goa_uniprot_gcrp.gaf - goa_uniprot_all.gaf: 0

# == seqid set stats ==

# goa_uniprot_all.gaf: 102989625
# goa_uniprot_gcrp.gaf: 23582588
# goa_uniprot_all.gaf & goa_uniprot_gcrp.gaf: 23582588
# goa_uniprot_all.gaf | goa_uniprot_gcrp.gaf: 102989625
# goa_uniprot_all.gaf - goa_uniprot_gcrp.gaf: 79407037
# goa_uniprot_gcrp.gaf - goa_uniprot_all.gaf: 0

# == symbol set stats ==

# goa_uniprot_all.gaf: 75301818
# goa_uniprot_gcrp.gaf: 18832370
# goa_uniprot_all.gaf & goa_uniprot_gcrp.gaf: 18832370
# goa_uniprot_all.gaf | goa_uniprot_gcrp.gaf: 75301818
# goa_uniprot_all.gaf - goa_uniprot_gcrp.gaf: 56469448
# goa_uniprot_gcrp.gaf - goa_uniprot_all.gaf: 0

# == qual set stats ==

# goa_uniprot_all.gaf: 6
# goa_uniprot_gcrp.gaf: 6
# goa_uniprot_all.gaf & goa_uniprot_gcrp.gaf: 6
# goa_uniprot_all.gaf | goa_uniprot_gcrp.gaf: 6
# goa_uniprot_all.gaf - goa_uniprot_gcrp.gaf: 0
# goa_uniprot_gcrp.gaf - goa_uniprot_all.gaf: 0

# == go set stats ==

# goa_uniprot_all.gaf: 29538
# goa_uniprot_gcrp.gaf: 28694
# goa_uniprot_all.gaf & goa_uniprot_gcrp.gaf: 28694
# goa_uniprot_all.gaf | goa_uniprot_gcrp.gaf: 29538
# goa_uniprot_all.gaf - goa_uniprot_gcrp.gaf: 844
# goa_uniprot_gcrp.gaf - goa_uniprot_all.gaf: 0

# == evidence set stats ==

# goa_uniprot_all.gaf: 23
# goa_uniprot_gcrp.gaf: 23
# goa_uniprot_all.gaf & goa_uniprot_gcrp.gaf: 23
# goa_uniprot_all.gaf | goa_uniprot_gcrp.gaf: 23
# goa_uniprot_all.gaf - goa_uniprot_gcrp.gaf: 0
# goa_uniprot_gcrp.gaf - goa_uniprot_all.gaf: 0

# == type set stats ==

# goa_uniprot_all.gaf: 26
# goa_uniprot_gcrp.gaf: 1
# goa_uniprot_all.gaf & goa_uniprot_gcrp.gaf: 1
# goa_uniprot_all.gaf | goa_uniprot_gcrp.gaf: 26
# goa_uniprot_all.gaf - goa_uniprot_gcrp.gaf: 25
# goa_uniprot_gcrp.gaf - goa_uniprot_all.gaf: 0

# == tax set stats ==

# goa_uniprot_all.gaf: 1334633
# goa_uniprot_gcrp.gaf: 12822
# goa_uniprot_all.gaf & goa_uniprot_gcrp.gaf: 12822
# goa_uniprot_all.gaf | goa_uniprot_gcrp.gaf: 1334633
# goa_uniprot_all.gaf - goa_uniprot_gcrp.gaf: 1321811
# goa_uniprot_gcrp.gaf - goa_uniprot_all.gaf: 0

# Runtime: 1533.14 s
