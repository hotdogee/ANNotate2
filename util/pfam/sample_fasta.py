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


# ubuntu 18.04, 0.26s/seq
# source ~/venv/tf37/bin/activate
# /usr/bin/time -v -o p31_seqs_with_p32_regions_of_p31_domains.n100.p31_results.tsv.2950X.time /opt/PfamScan/pfam_scan.pl -fasta p31_seqs_with_p32_regions_of_p31_domains.n100.fa -dir ./ -outfile ./p31_seqs_with_p32_regions_of_p31_domains.n100.p31_results.tsv
# Elapsed (wall clock) time (h:mm:ss or m:ss): 0:26.68
# /usr/bin/time -v -o p32_seqs_with_p32_regions_of_p31_domains.n100.p31_results.tsv.2950X.time /opt/PfamScan/pfam_scan.pl -fasta p32_seqs_with_p32_regions_of_p31_domains.n100.fa -dir ./ -outfile ./p32_seqs_with_p32_regions_of_p31_domains.n100.p31_results.tsv
# Elapsed (wall clock) time (h:mm:ss or m:ss): 0:25.08

# windows
# python .\util\pfam\sample_fasta.py --input D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.fa.gz --num 100 --seed 42
# python .\util\pfam\sample_fasta.py --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.n100.fa.gz --num 100 --seed 42

# fix last seq bug
# python .\util\pfam\sample_fasta.py --input D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa.gz --output D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.n100.fa.gz --num 100 --seed 42

# /usr/bin/time -v -o p32_seqs_with_p32_regions_of_p31_domains_2.n100.p31_results.tsv.2950X.time /opt/PfamScan/pfam_scan.pl -fasta p32_seqs_with_p32_regions_of_p31_domains_2.n100.fa -dir ./ -outfile ./p32_seqs_with_p32_regions_of_p31_domains_2.n100.p31_results.tsv
# Elapsed (wall clock) time (h:mm:ss or m:ss): 0:24.63

# make n1000, n10000, n100000 datasets
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/sample_fasta.py --input /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.fa --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n1000.fa --num 1000 --seed 42
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/extract_pfam_regions_tsv_with_fasta.py  --fasta /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n1000.fa --input /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n1000.tsv
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/split_fasta.py --input /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n1000.fa --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n1000_fa_distributed --num 12 --mode distributed
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_pfam.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n1000_fa_distributed --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n1000_fa_distributed/p31_results --workers 12
# Runtime: 34.09 s
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/compute_pfam_scan_classification_performance.py --pred /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n1000_fa_distributed/p31_results --ans /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n1000.tsv --fasta /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n1000.fa --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n1000.p31_results.all_regions_perf.json

# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/sample_fasta.py --input /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.fa --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n10000.fa --num 10000 --seed 42
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/extract_pfam_regions_tsv_with_fasta.py  --fasta /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n10000.fa --input /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n10000.tsv
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/split_fasta.py --input /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n10000.fa --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_distributed --num 12 --mode distributed
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_pfam.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_distributed --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_distributed/p31_results --workers 12
# Runtime: 318.28 s
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/compute_pfam_scan_classification_performance.py --pred /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_distributed/p31_results --ans /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n10000.tsv --fasta /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n10000.fa --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000.p31_results.all_regions_perf.json
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/batch_fasta.py --input /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n10000.fa --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_batched_34000 --size 34000
# Batch: 112 OOM
# /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/batch_fasta.py --input /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n10000.fa --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_batched_25000 --size 25000
# Batch: 152
# /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_serving_v2.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_batched_25000 --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_batched_25000/ann31_${VERSION}_results_raw --servers http://localhost:8501/v1/models/pfam/versions/${VERSION}:predict,http://localhost:8601/v1/models/pfam/versions/${VERSION}:predict,http://localhost:8701/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.33:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.33:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.34:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.34:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.74:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.74:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8501/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8601/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8701/v1/models/pfam/versions/${VERSION}:predict,http://192.168.1.35:8801/v1/models/pfam/versions/${VERSION}:predict --readers 12 --writers 12
# /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/predict/20190913-p32-n1000/run_p32_n10000.sh
# /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/annotate_pfam_batch_msgpack_to_tsv_v2.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_batched_25000/ann31_${VERSION}_results_raw --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000.ann31_${VERSION}_results.tsv --workers 24
# /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/predict/20190913-p32-n1000/run_p32_n10000-part2.sh
# /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/compute_annotate_pfam_classification_performance.py  --pred /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000.ann31_${VERSION}_results.tsv --ans /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n10000.tsv --fasta /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n10000.fa --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000.ann31_${VERSION}_results.all_regions_perf.json --outtsv /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000.ann31_results.all_regions_perf.tsv
# /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/predict/20190913-p32-n1000/run_p32_n10000-part3.sh

# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/sample_fasta.py --input /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.fa --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n100000.fa --num 100000 --seed 42
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/extract_pfam_regions_tsv_with_fasta.py  --fasta /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n100000.fa --input /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n100000.tsv
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/split_fasta.py --input /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n100000.fa --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n100000_fa_distributed --num 12 --mode distributed
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_pfam.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n100000_fa_distributed --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n100000_fa_distributed/p31_results --workers 12
# Runtime: 3170.15 s
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/compute_pfam_scan_classification_performance.py --pred /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n100000_fa_distributed/p31_results --ans /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n100000.tsv --fasta /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n100000.fa --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n100000.p31_results.all_regions_perf.json
# Runtime: 3170.15 s
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/batch_fasta.py --input /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n100000.fa --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n100000_fa_batched_25000 --size 25000
# Batch: 1500
# /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/predict/20190913-p32-n1000/run_p32_n100000.sh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sample N sequences from a FASTA file.'
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
        '--output',
        type=str,
        required=True,
        help="Path to the output FASTA file, required."
    )
    parser.add_argument(
        '-n',
        '--num',
        type=int,
        default=100,
        help="Number of sequences to sample. (default: %(default)s)"
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=42,
        help="Random seed. (default: %(default)s)"
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    # verify paths
    input_path = verify_input_path(args.input)
    output_path = verify_output_path(args.output)
    logging.info(
        f"Sampling {args.num} seqs from {'/'.join(input_path.parts[-2:])} with random seed {args.seed} to {'/'.join(output_path.parts[-2:])}"
    )

    # read fasta
    seqs = _fa_gz_to_list(input_path)
    seq_count = len(seqs)
    logging.info(
        f"Parsed {seq_count} seqs from {'/'.join(input_path.parts[-2:])}"
    )

    # sort by length
    seqs.sort(key=lambda seq: seq.len)
    logging.info('Sorting done.')

    # pick N
    random.seed(args.seed)

    # write fasta
    if output_path.suffix == '.gz':
        f = gzip.open(output_path, mode='wt', encoding='utf-8')
    else:
        f = output_path.open(mode='wt', encoding='utf-8')
    with f:
        bucket_size = seq_count / args.num
        for i in range(args.num):
            seq = seqs[random.randrange(
                round(bucket_size * i), round(bucket_size * (i + 1))
            )]
            f.write(seq.entry)

    print(f'Runtime: {time.time() - start_time:.2f} s\n')
    sys.exit(0)

# 2019-09-03 16:24:13 INFO     Sampling 100 seqs from Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz with random seed 42 to Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.fa.gz
# 100%|#########################################################8| 147519711/147958183 [00:02<00:00, 57397600.18bytes/s]
# 2019-09-03 16:24:15 INFO     Parsed 219235 seqs from Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz
# 2019-09-03 16:24:15 INFO     Sorting done.
# Runtime: 2.66 s

# 2019-09-03 16:26:28 INFO     Sampling 100 seqs from Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.fa.gz with random seed 42 to Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.n100.fa.gz
# 17806348108bytes [06:32, 45362137.90bytes/s]
# 2019-09-03 16:33:01 INFO     Parsed 38168101 seqs from Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.fa.gz
# 2019-09-03 16:33:15 INFO     Sorting done.
# Runtime: 406.27 s
