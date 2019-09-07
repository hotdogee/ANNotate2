import os
import re
import sys
import time
import gzip
import json
import errno
import random
import struct
import msgpack
import logging
import requests
import argparse
import platform
import subprocess
import concurrent.futures
import tensorflow as tf
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


# URL = 'https://ann.hanl.in/v101/models/pfam:predict'
# URL = 'http://localhost:8601/v1/models/pfam31:predict'
# URL = 'http://localhost:8601/v1/models/pfam31/versions/1567765316:predict'


def serving_predict_fasta(
    fa_path, out_path, url='http://localhost:8501/v1/models/pfam:predict'
):
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
    entry = {'id': '', 'seq': ''}
    # current = 0
    with f as fa_f:
        for line in fa_f:
            line_s = line.strip()
            if len(line_s) > 0 and line_s[0] == '>':
                if entry['seq']:
                    seq_list.append(entry)
                    entry = {'id': '', 'seq': ''}
                entry['id'] = line.split()[0][1:]
            else:
                entry['seq'] += line_s
        if entry['seq']:  # handle last seq
            seq_list.append(entry)

    payload = {'instances': [e['seq'] for e in seq_list]}
    r = requests.post(url, data=json.dumps(payload))
    try:
        predictions = r.json()['predictions']
    except Exception as exc:
        print(r.json())
        raise exc
    for i in range(len(seq_list)):
        seq_len = len(seq_list[i]['seq'])
        del seq_list[i]['seq']
        seq_list[i]['classes'] = predictions[i]['classes'][:seq_len]
        seq_list[i]['top_probs'] = predictions[i]['top_probs'][:seq_len]
        seq_list[i]['top_classes'] = predictions[i]['top_classes'][:seq_len]

    # print(f"{seq_list[0]['id']}:", seq_list[0]['classes'])
    # write output file
    f = out_path.open(mode='wb')
    with f:
        msgpack.dump(seq_list, f)

    return fa_path


# windows
# python .\util\pfam\run_batch_serving.py --indir D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains_fa_split_batched --outdir D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains_fa_split_batched/pfam31_results_raw

# ubuntu 18.04
# export PERL5LIB=/opt/PfamScan:$PERL5LIB
# source /home/hotdogee/venv/tf37/bin/activate

# python ./util/pfam/run_batch_serving.py --indir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_fa_split_batched_25000 --outdir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_fa_split_batched_25000/pfam31_1567787563_results_raw --workers 2 --server http://localhost:8501/v1/models/pfam:predict
# Runtime: 3369.25 s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Run ANNotate prediction locally on each FASTA file in the input directory.'
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
        default=2,
        help="Number of concurrent processes to run. (default: %(default)s)"
    )
    parser.add_argument(
        '-s',
        '--server',
        type=str,
        default='http://localhost:8501/v1/models/pfam:predict',
        help="URL of tensorflow serving server. (default: %(default)s)"
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    # verify paths
    indir_path = verify_indir_path(args.indir)
    outdir_path = verify_outdir_path(args.outdir, required_empty=False)

    # read existing output paths
    existing_stems = set(
        [Path(p.stem).stem for p in outdir_path.glob('*.msgpack')]
    )

    # read input fasta paths excluding those with existing output
    in_paths = sorted(
        [p for p in indir_path.glob('*.fa') if p.stem not in existing_stems]
    )

    # ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as e:
        future_path = {
            e.submit(
                serving_predict_fasta, p,
                outdir_path / f'{p.stem}.pfam31_results_raw.msgpack',
                args.server
            ): p
            for p in in_paths
        }
        for future in concurrent.futures.as_completed(future_path):
            path = future_path[future]
            try:
                result = future.result()
            except Exception as exc:
                logging.info(
                    f"{'/'.join(path.parts[-2:])} generated an exception: {str(exc)}"
                )
            else:
                logging.info(
                    f"Completed processing {'/'.join(path.parts[-2:])}"
                )

    logging.info(f'Runtime: {time.time() - start_time:.2f} s')
    sys.exit(0)
