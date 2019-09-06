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

# python .\util\pfam\batch_fasta.py --input D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa --outdir D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains_fa_split_batched
# Runtime: 24.82 s
# Batches: 3586

# python .\util\pfam\batch_fasta.py --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa --outdir D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_batched

# python ./util/pfam/batch_fasta.py --input /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.fa --outdir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_fa_split_batched
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split a FASTA file into N approximately equal size batches.'
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
        '-s',
        '--size',
        type=int,
        default=35000,
        help=
        "Number amino acids per batch, padding included. (default: %(default)s)"
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
    # out_fs = [path.open(mode='wt', encoding='utf-8') for path in output_paths]

    batch_num = 1
    batch_seqs = []
    max_len = 0
    for seq in seqs:
        if seq.len * (len(batch_seqs) + 1) > args.size and len(batch_seqs) > 0:
            logging.info(f'Batch: {batch_num}')
            out_path = outdir_path / f'{stem}.{batch_num:05d}{suffix}'
            f = out_path.open(mode='wt', encoding='utf-8')
            with f:
                for s in batch_seqs:
                    f.write(s.entry)
            batch_num += 1
            batch_seqs = []
            max_len = 0
        if seq.len > max_len:
            max_len = seq.len
        batch_seqs.append(seq)
    if len(batch_seqs) > 0:  # last batch
        logging.info(f'Batch: {batch_num}')
        out_path = outdir_path / f'{stem}.{batch_num:05d}{suffix}'
        f = out_path.open(mode='wt', encoding='utf-8')
        with f:
            for s in batch_seqs:
                f.write(s.entry)

    logging.info(f'Runtime: {time.time() - start_time:.2f} s')
    sys.exit(0)
