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
# python .\util\pfam\test_gz_tqdm.py -1 .\util\pfam\a.tsv
# python .\util\pfam\test_gz_tqdm.py -1 .\util\pfam\a.tsv.gz

def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]

def test_gz_tqdm(path):
    path = Path(path)
    # if gzipped
    target = os.path.getsize(path)
    print(f'os.path.getsize(path): {target}\n')
    if path.suffix == '.gz':
        f = gzip.open(path, mode='rt', encoding='utf-8')
        target = _gzip_size(path)
        while target - os.path.getsize(path) < -30:
            # the uncompressed size can't be smaller than the compressed size, so add 4GB
            target += 2**32
        print(f'gzip_size(path): {target}\n')
    else:
        f = path.open(mode='r', encoding='utf-8')
    # initialize
    total = 0
    with f, tqdm(total=target) as t:
        for line in f:
            time.sleep(0.8)
            t.update(len(line))
            # line code
            total += len(line)
            print(f'len(line): {len(line)}/{total}\n')
    return total

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

if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description='Test.')
    parser.add_argument('-1', '--tsv1', type=str, required=True,
        help="Path to the first .tsv.gz file, required.")
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    tsv1_path = verify_input_path(args.tsv1)
    print(f'Reading: {tsv1_path.name}')

    test_gz_tqdm(tsv1_path)

    print(f'Run time: {time.time() - start_time:.2f} s\n')
