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
import platform
import subprocess
import concurrent.futures
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


def verify_output_path(p):
    # get absolute path
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # existing file
    if path.exists():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path)
    # assert dirs
    path.parent.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
    return path


def verify_indir_path(p):
    # get absolute path
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # doesn't exist
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    # existing file or directory
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(
                errno.ENOTDIR, os.strerror(errno.ENOTDIR), path
            )
        else:  # got existing directory
            assert len([x for x in path.iterdir()]) != 0, 'Directory is empty'
    return path


def verify_outdir_path(p, required_empty=True):
    # get absolute path
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # existing file or directory
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(
                errno.ENOTDIR, os.strerror(errno.ENOTDIR), path
            )
        elif required_empty:  # got existing directory
            assert len([x for x in path.iterdir()]) == 0, 'Directory not empty'
    # directory does not exist, create it
    path.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
    return path


# windows
# python .\util\pfam\run_batch_pfam.py --indir D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed --outdir D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed/p31_results

# ubuntu 18.04
# export PERL5LIB=/opt/PfamScan:$PERL5LIB
# source /home/hotdogee/venv/tf37/bin/activate
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_pfam.py --indir /home/hotdogee/pfam/test --outdir /home/hotdogee/pfam/test/p31_results -- workers 12
# W2125
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_pfam.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed/p31_results --workers 4
# 4960X
# python ./run_batch_pfam.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed/p31_results --workers 12 --start 100
# 8086K1
# python ./run_batch_pfam.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed/p31_results --workers 8 --start 33
# 8086K2
# python ./run_batch_pfam.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed/p31_results --workers 8 --start 66
# 2650v1
# python ./run_batch_pfam.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed/p31_results --workers 16 --start 150

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run pfam_scan.pl on each FASTA file in the input directory.'
    )
    parser.add_argument(
        '-i',
        '--indir',
        type=str,
        required=True,
        help="Path to the input directory, required."
    )
    parser.add_argument(
        '-o',
        '--outdir',
        type=str,
        required=True,
        help="Path to the output directory, required."
    )
    parser.add_argument(
        '-w',
        '--workers',
        type=int,
        default=12,
        help=
        "Number of concurrent pfam_scan processes to run. (default: %(default)s)"
    )
    parser.add_argument(
        '-s',
        '--start',
        type=int,
        default=0,
        help=
        "Index of input file list to start processing. (default: %(default)s)"
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    # verify paths
    indir_path = verify_indir_path(args.indir)
    outdir_path = verify_outdir_path(args.outdir, required_empty=False)

    # read existing output paths
    existing_stems = set([Path(p.stem).stem for p in outdir_path.glob('*.tsv')])

    # read input fasta paths excluding those with existing output
    in_paths = sorted(
        [p for p in indir_path.glob('*.fa') if p.stem not in existing_stems]
    )
    in_paths = in_paths[args.start:] + in_paths[:args.start]

    # node name
    node_name = platform.node()

    # ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as e:
        future_path = {
            e.submit(
                subprocess.run, [
                    '/usr/bin/time', '-v', '-o',
                    str(
                        outdir_path /
                        f'{p.stem}.p31_results.tsv.{node_name}.time'
                    ), '/opt/PfamScan/pfam_scan.pl', '-fasta',
                    str(p), '-dir', './', '-outfile',
                    str(outdir_path / f'{p.stem}.p31_results.tsv')
                ]
            ): p
            for p in in_paths
        }
        for future in concurrent.futures.as_completed(future_path):
            path = future_path[future]
            try:
                completed_process = future.result()
            except Exception as exc:
                logging.info(
                    f"{'/'.join(path.parts[-2:])} generated an exception: {exc:s}"
                )
            else:
                logging.info(
                    f"Completed processing {'/'.join(path.parts[-2:])}, returncode={completed_process.returncode}"
                )

    logging.info(f'Runtime: {time.time() - start_time:.2f} s')
    sys.exit(0)
