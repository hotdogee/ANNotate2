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


def _fa_gz_to_list(fa_path):
    """Parse a .fa or .fa.gz file into a list of Seq(len, entry)
    """
    Seq = namedtuple('Seq', ['len', 'entry'])
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
    seq_list = []
    seq_len, seq_entry = 0, ''
    # current = 0
    with f as fa_f, tqdm(
        total=target, unit='bytes', dynamic_ncols=True, ascii=True
    ) as t:
        for line in fa_f:
            t.update(len(line))
            line_s = line.strip()
            if len(line_s) > 0 and line_s[0] == '>':
                if seq_entry:
                    seq_list.append(Seq(seq_len, seq_entry))
                    seq_len, seq_entry = 0, ''
            else:
                seq_len += len(line_s)
            seq_entry += line
        if seq_entry:  # handle last seq
            seq_list.append(Seq(seq_len, seq_entry))
    return seq_list


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


# windows
# python .\util\pfam\count_fasta.py --input D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz
# python .\util\pfam\count_fasta.py --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.fa.gz
# python .\util\pfam\count_fasta.py --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa.gz

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Print the number of sequences in a FASTA file.'
    )
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=True,
        help="Path to the input FASTA file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    # verify paths
    input_path = verify_input_path(args.input)

    # read fasta
    seqs = _fa_gz_to_list(input_path)
    seq_count = len(seqs)
    logging.info(f"Sequence count: {seq_count}")

    print(f'Runtime: {time.time() - start_time:.2f} s\n')
    sys.exit(0)

# (tf37) PS C:\Users\Hotdogee\Dropbox\Work\Btools\ANNotate\ANNotate2> python .\util\pfam\count_fasta.py --input D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz
# 100%|######################################################8| 147519711/147958183 [00:02<00:00, 62376494.45bytes/s]
# 2019-09-05 01:29:19 INFO     Sequence count: 219236
# Runtime: 2.39 s

# (tf37) PS C:\Users\Hotdogee\Dropbox\Work\Btools\ANNotate\ANNotate2> python .\util\pfam\count_fasta.py --input D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa
# 100%|#####################################################8| 147519711/147958183 [00:01<00:00, 122421551.11bytes/s]
# 2019-09-05 01:29:25 INFO     Sequence count: 219236
# Runtime: 1.23 s

# (tf37) PS C:\Users\Hotdogee\Dropbox\Work\Btools\ANNotate\ANNotate2> python .\util\pfam\count_fasta.py --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.fa.gz
# 17806348108bytes [05:52, 50456415.96bytes/s]
# 2019-09-05 01:36:16 INFO     Sequence count: 38168102
# Runtime: 352.93 s

# (tf37) PS C:\Users\Hotdogee\Dropbox\Work\Btools\ANNotate\ANNotate2> python .\util\pfam\count_fasta.py --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.fa
# 100%|##################################################7| 17806348108/17882684312 [03:23<00:00, 87495726.40bytes/s]
# 2019-09-05 01:34:56 INFO     Sequence count: 38168102
# Runtime: 203.53 s

# (tf37) PS C:\Users\Hotdogee\Dropbox\Work\Btools\ANNotate\ANNotate2> python .\util\pfam\count_fasta.py --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa.gz
# 17806348298bytes [05:53, 50425585.96bytes/s]
# 2019-09-05 01:43:18 INFO     Sequence count: 38168103
# Runtime: 353.14 s

# (tf37) PS C:\Users\Hotdogee\Dropbox\Work\Btools\ANNotate\ANNotate2> python .\util\pfam\count_fasta.py --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa
# 100%|##################################################7| 17806348298/17882684504 [03:27<00:00, 85904900.12bytes/s]
# 2019-09-05 01:38:59 INFO     Sequence count: 38168103
# Runtime: 207.30 s
