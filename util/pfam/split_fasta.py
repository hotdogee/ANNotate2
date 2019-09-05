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
    level=logging.DEBUG,
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
    # get absolute path
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # doesn't exist
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    # is dir
    if path.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
    return path


def verify_outdir_path(p):
    # get absolute path
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # existing file or directory
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(
                errno.ENOTDIR, os.strerror(errno.ENOTDIR), path
            )
        else:  # got existing directory
            assert len([x for x in path.iterdir()]) == 0, 'Directory not empty'
    # directory does not exist, create it
    path.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
    return path


def verify_output_path(p):
    # get absolute path
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # existing file
    if path.exists():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path)
    # assert dirs
    path.parent.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
    return path


# windows
# python .\util\pfam\split_fasta.py --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa --outdir D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split a FASTA file into N files.'
    )
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=True,
        help="Path to the input FASTA file, required."
    )
    parser.add_argument(
        '-o',
        '--outdir',
        type=str,
        required=True,
        help="Path to the output directory, required."
    )
    parser.add_argument(
        '-n',
        '--num',
        type=int,
        default=200,
        help="Number of files to split into. (default: %(default)s)"
    )
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        default='distributed',
        choices=['distributed', 'sorted'],
        help=". (default: %(default)s)"
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    # verify paths
    input_path = verify_input_path(args.input)
    outdir_path = verify_outdir_path(args.outdir)

    # generate output paths
    stem = input_path.stem
    suffix = input_path.suffix
    if suffix == '.gz':
        stem = Path(stem).stem
        suffix = Path(stem).suffix
    count_fmt = "{0:0" + str(len(str(args.num))) + "d}"
    output_paths = [
        outdir_path / f'{stem}.{count_fmt.format(i+1)}of{args.num}{suffix}'
        for i in range(args.num)
    ]
    # logging.debug(output_paths)

    # read fasta
    seqs = _fa_gz_to_list(input_path)
    seq_count = len(seqs)
    logging.info(
        f"Parsed {seq_count} seqs from {'/'.join(input_path.parts[-2:])}"
    )

    # sort by length
    seqs.sort(key=lambda seq: seq.len)
    logging.info('Sorting done.')

    # write fasta
    out_fs = [path.open(mode='wt', encoding='utf-8') for path in output_paths]

    if args.mode == 'distributed':
        for i in range(seq_count):
            out_fs[i % 200].write(seqs[i].entry)
    else:
        seq_per_file = seq_count / args.num
        for i in range(seq_count):
            out_fs[int(i / seq_per_file)].write(seqs[i].entry)

    out_fs = [f.close() for f in out_fs]
    print(f'Runtime: {time.time() - start_time:.2f} s\n')
    sys.exit(0)

# python .\util\pfam\split_fasta.py --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa --outdir D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed
# 100%|##################################################7| 17806348298/17882684504 [03:26<00:00, 86361656.90bytes/s]
# 2019-09-05 16:20:16 INFO     Parsed 38168103 seqs from Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa
# 2019-09-05 16:20:29 INFO     Sorting done.
# Runtime: 322.64 s

#  python .\util\pfam\split_fasta.py --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa --outdir D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_sorted --mode sorted
# 100%|#############################8| 17806348298/17882684504 [03:25<00:00, 86583108.47bytes/s]
# 2019-09-05 23:34:48 INFO     Parsed 38168103 seqs from Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa
# 2019-09-05 23:35:01 INFO     Sorting done.
# Runtime: 310.01 s
