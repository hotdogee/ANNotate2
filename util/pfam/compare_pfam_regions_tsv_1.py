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

# windows
# python .\util\pfam\compare_pfam_regions_tsv_1.py --tsv1 D:/pfam/Pfam31.0/Pfam-A.regions.uniprot.tsv.gz --tsv2 D:/pfam/Pfam32.0/Pfam-A.regions.uniprot.tsv.gz

# uniprot_acc seq_version crc64            md5                              pfamA_acc seq_start seq_end
# A0A103YDS2  1           3C510D74FCA8479C 31e3d17a0f6766e2a545214e475d72a8 PF04863   52        107
# A0A087Y7L3  1           E79AF82467BBFA7D bb8465c4f3cd34f56f6c60817d43f551 PF09014   261       348
# A0A103YDS2  1           3C510D74FCA8479C 31e3d17a0f6766e2a545214e475d72a8 PF04863   433       488
# A0A103YDS2  1           3C510D74FCA8479C 31e3d17a0f6766e2a545214e475d72a8 PF04863   870       925
# A0A199V6R7  1           F37C9E162DD24DD7 0656490991e5645a3e2b402a861e450e PF04863   66        120
# A0A199V6R7  1           F37C9E162DD24DD7 0656490991e5645a3e2b402a861e450e PF04863   468       522
# A0A0D9V4Z5  1           25C8FDA89273F894 1ddc7bbbfec83e195473a9e51b763b16 PF04863   56        111
# W5N889      1           509B93CB8368603C 0e92ab8939197d40e00e0238de6b8a69 PF09014   264       351
# A0A0D9V4Z5  1           25C8FDA89273F894 1ddc7bbbfec83e195473a9e51b763b16 PF04863   552       607

# set stats:
# * UA: uniprot_acc
# * UA_SV: uniprot_acc + seq_version
# * PA: pfamA_acc
# * UA_SV_PA: uniprot_acc + seq_version + pfamA_acc
# * UA_SV_PA_SE: uniprot_acc + seq_version + pfamA_acc + seq_start + seq_end

# /Pfam31.0 88M
# $ wc -l Pfam-A.regions.uniprot.tsv
# 88761543 Pfam-A.regions.uniprot.tsv
# /Pfam32.0 138M
# $ wc -l Pfam-A.regions.uniprot.tsv
# 138165284 Pfam-A.regions.uniprot.tsv

def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]

def stats_pfam_regions_tsv(path):
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
    with f, tqdm(total=target, unit='bytes', dynamic_ncols=True, ascii=True) as t:
        for line in f:
            t.update(len(line))
            line_num += 1
            if line_num == 1: continue # skip header
            # line code
            tokens = line.strip().split()
            vocab['UA'].add(f'{tokens[0]}')
            vocab['UA_SV'].add(f'{tokens[0]}.{tokens[1]}')
            vocab['PA'].add(f'{tokens[4]}')
            vocab['UA_SV_PA'].add(f'{tokens[0]}.{tokens[1]}.{tokens[4]}')
            vocab['UA_SV_PA_SE'].add(f'{tokens[0]}.{tokens[1]}.{tokens[4]}.{tokens[5]}.{tokens[6]}')
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
    print(f'''
{n1}: {len(s1)} {unit}
{n2}: {len(s2)} {unit}
{n1} & {n2}: {len(s1 & s2)} {unit}
{n1} | {n2}: {len(s1 | s2)} {unit}
{n1} - {n2}: {len(s1 - s2)} {unit}
{n2} - {n1}: {len(s2 - s1)} {unit}
''')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Print the differences of two Pfam-A.regions.uniprot.tsv.gz files.')
    parser.add_argument('-1', '--tsv1', type=str, required=True,
        help="Path to the first Pfam-A.regions.uniprot.tsv.gz file, required.")
    parser.add_argument('-2', '--tsv2', type=str, required=True,
        help="Path to the second Pfam-A.regions.uniprot.tsv.gz file, required.")
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    tsv1_path = verify_input_path(args.tsv1)
    tsv2_path = verify_input_path(args.tsv2)
    print(f'Comparing: {tsv1_path.name} and {tsv2_path.name}')

    tsv1_vocab, tsv1_line_num = stats_pfam_regions_tsv(tsv1_path)
    print(f'Processed {tsv1_line_num} lines in {tsv1_path.name}')
    tsv2_vocab, tsv2_line_num = stats_pfam_regions_tsv(tsv2_path)
    print(f'Processed {tsv2_line_num} lines in {tsv2_path.name}')

    for sid in tsv1_vocab.keys():
        print(f'== {sid} set stats ==')
        print_set_stats('Pfam31', tsv1_vocab[sid], 'Pfam32', tsv2_vocab[sid])

    print(f'Runtime: {time.time() - start_time:.2f} s\n')

# Processed 138165284 lines in Pfam-A.regions.uniprot.tsv.gz
# ==UA set stats==

# Pfam31: 54223493
# Pfam32: 88966235
# Pfam31 & Pfam32: 50546302
# Pfam31 | Pfam32: 92643426
# Pfam31 - Pfam32: 3677191
# Pfam32 - Pfam31: 38419933

# ==UA_SV set stats==

# Pfam31: 54223493
# Pfam32: 88966235
# Pfam31 & Pfam32: 50430906
# Pfam31 | Pfam32: 92758822
# Pfam31 - Pfam32: 3792587
# Pfam32 - Pfam31: 38535329

# ==PA set stats==

# Pfam31: 16712
# Pfam32: 17929
# Pfam31 & Pfam32: 16700
# Pfam31 | Pfam32: 17941
# Pfam31 - Pfam32: 12
# Pfam32 - Pfam31: 1229

# ==UA_SV_PA set stats==

# Pfam31: 80971626
# Pfam32: 125800935
# Pfam31 & Pfam32: 69832952
# Pfam31 | Pfam32: 136939609
# Pfam31 - Pfam32: 11138674
# Pfam32 - Pfam31: 55967983

# ==UA_SV_PA_SE set stats==

# Pfam31: 88761542
# Pfam32: 138165283
# Pfam31 & Pfam32: 73271292
# Pfam31 | Pfam32: 153655533
# Pfam31 - Pfam32: 15490250
# Pfam32 - Pfam31: 64893991

# Runtime: 1724.41 s (1950X)
