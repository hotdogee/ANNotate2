import os
import re
import sys
import time
import gzip
import json
import errno
import random
import struct
import logging
import argparse
from tqdm import tqdm
from glob import glob
from array import array
from pathlib import Path
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger().setLevel(logging.INFO)


def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def _open_input_path(input_path):
    """Return a file handle and file size from path, supports gzip
    """
    target = os.path.getsize(input_path)
    if input_path.suffix == '.gz':
        in_f = gzip.open(input_path, mode='rt', encoding='utf-8')
        target = _gzip_size(input_path)
        while target < os.path.getsize(input_path):
            # the uncompressed size can't be smaller than the compressed size, so add 4GB
            target += 2**32
    else:
        in_f = input_path.open(mode='rt', encoding='utf-8')
    return in_f, target


def _fa_gz_to_id_len_dict(fa_path):
    """Parse a .fa or .fa.gz file into a dict of {seq_id: seq_len}
    """
    Seq = namedtuple('Seq', ['len', 'index'])
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
    id_len_dict = {}
    seq_len, seq_id = 0, ''
    i = 0
    # current = 0
    with f as fa_f, tqdm(
        total=target, unit='bytes', dynamic_ncols=True, ascii=True
    ) as t:
        for line in fa_f:
            t.update(len(line))
            line_s = line.strip()
            if len(line_s) > 0 and line_s[0] == '>':
                if seq_id:
                    id_len_dict[seq_id] = Seq(seq_len, i)
                    seq_len, seq_id = 0, ''
                    i += 1
                # parse header
                seq_id = line_s.split()[0][1:]
            else:
                seq_len += len(line_s)
        if seq_id:  # handle last seq
            id_len_dict[seq_id] = Seq(seq_len, i)
    return id_len_dict


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
        raise FileExistsError(errno.ENOENT, os.strerror(errno.EEXIST), path)
    # is dir
    if path.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
    # assert dirs
    path.parent.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
    return path


# windows

# ann31_1567719465_results
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567719465_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567719465_results.all_regions_perf.json
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567719465_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567719465_results.p32_regions_perf.json

# ann31_1567765316_results
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567765316_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567765316_results.all_regions_perf.json
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567765316_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567765316_results.p32_regions_perf.json

# ann31_1567786205_results
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786205_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786205_results.all_regions_perf.json
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786205_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786205_results.p32_regions_perf.json

# ann31_1567786591_results
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786591_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786591_results.all_regions_perf.json
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786591_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786591_results.p32_regions_perf.json

# ann31_1567786835_results
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786835_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786835_results.all_regions_perf.json
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786835_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786835_results.p32_regions_perf.json

# ann31_1567787010_results
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787010_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787010_results.all_regions_perf.json
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787010_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787010_results.p32_regions_perf.json

# ann31_1567787530_results
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787530_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787530_results.all_regions_perf.json
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787530_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787530_results.p32_regions_perf.json

# ann31_1567787563_results
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787563_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787563_results.all_regions_perf.json
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787563_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787563_results.p32_regions_perf.json

# ann31_1567787595_results
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787595_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787595_results.all_regions_perf.json
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787595_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787595_results.p32_regions_perf.json

# p32_seqs_with_p32_regions_of_p31_domains_2
# ann31_1567889661_results
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/compute_annotate_pfam_classification_performance.py  --pred /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.ann31_1567889661_results.tsv --ans /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.fa --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.ann31_1567889661_results.all_regions_perf.json
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/compute_annotate_pfam_classification_performance.py  --pred /data12/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.ann31_1567889661_results.tsv --ans /data12/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta /data12/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.fa --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.ann31_1567889661_results.all_regions_perf.json
# python .\util\pfam\compute_annotate_pfam_classification_performance.py  --pred D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.ann31_1567889661_results.tsv --ans D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa --output D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.ann31_1567889661_results.all_regions_perf.json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute ANNotate Pfam performance metrics.'
    )
    parser.add_argument(
        '-p',
        '--pred',
        type=str,
        required=True,
        help="Path to the ANNotate predicted TSV file, required."
    )
    parser.add_argument(
        '-a',
        '--ans',
        type=str,
        required=True,
        help="Path to the regions TSV file, required."
    )
    parser.add_argument(
        '-f',
        '--fasta',
        type=str,
        required=True,
        help="Path to the FASTA file, required."
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help="Path to the output JSON file, required."
    )
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        default='all',
        choices=['all', 'p32'],
        help=
        "Whether the regions TSV file contains all regions or only newly found regions in pfam32. (default: %(default)s)"
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    # Outputs
    # Accuracy per amino acid
    #
    # PredictedSet = set((seq,domain))
    # AnswerSet = set((seq,domain))
    # PredictedPositive = len(PredictedSet)
    # AnswerPositive = len(AnswerSet)
    # TruePositive = len(PredictedSet & AnswerSet)
    # FalsePositive = len(PredictedSet - AnswerSet)
    # FalseNegative = len(AnswerSet - PredictedSet)
    #
    # precision = TruePositive / PredictedPositive
    # recall = TruePositive / AnswerPositive
    # f1_score = 2*precision*recall/(precision+recall)
    # false_discovery_rate = FalsePositive / PredictedPositive
    # false_omission_rate = FalseNegative / AnswerPositive

    # verify paths
    pred_path = verify_input_path(args.pred)
    ans_path = verify_input_path(args.ans)
    fasta_path = verify_input_path(args.fasta)
    output_path = verify_output_path(args.output)
    logging.info(
        f"""Computing pfam classification performance with:
  Prediction results: {'/'.join(pred_path.parts[-2:])}
  Regions answers: {'/'.join(ans_path.parts[-2:])}
  FASTA sequences: {'/'.join(fasta_path.parts[-2:])}
  Mode: {args.mode}
  Output: {'/'.join(output_path.parts[-2:])}"""
    )

    # parse fasta into {seq_id: seq_len}
    id_len_dict = _fa_gz_to_id_len_dict(fasta_path)
    seq_count = len(id_len_dict)
    logging.info(
        f"Parsed {seq_count} seqs from {'/'.join(fasta_path.parts[-2:])}"
    )

    # parse pred
    # use envelope coordinates, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2808889/
    # coordinates are 1-inclusive
    pred_f, target = _open_input_path(pred_path)

    # initialize data structures
    domain_index = {'NO_DOMAIN': 0}
    seq_index = {}
    pred_set = set()
    pred_sequence = {}
    line_num = 0

    # process input by line
    with pred_f, tqdm(
        total=target, unit='bytes', dynamic_ncols=True, ascii=True
    ) as t:
        for line in pred_f:
            t.update(len(line))
            line = line.strip()
            line_num += 1
            if line_num == 1:
                continue  # skip header
            # parse
            tokens = line.split()
            # assert length == 15
            assert (
                len(tokens) == 4
            ), f'pred token length: {len(tokens)}, expected: 4'
            # get seq_len
            seq_id = tokens[0]
            seq_len = id_len_dict[seq_id].len
            si = id_len_dict[seq_id].index
            # get domain index
            domain = tokens[3]
            if domain not in domain_index:
                domain_index[domain] = len(domain_index)
            di = domain_index[domain]
            # add to set
            pred_set.add((si, di))
            # add to sequence
            if si not in pred_sequence:
                pred_sequence[si] = array('i', [0] * seq_len)
            seq_start, seq_end = int(tokens[1]), int(tokens[2])
            pred_sequence[si] = pred_sequence[si][:seq_start - 1] + array(
                'i', [di] * (seq_end - seq_start + 1)
            ) + pred_sequence[si][seq_end:]

    # parse ans
    # coordinates are 1-inclusive
    ans_f, target = _open_input_path(ans_path)

    # initialize data structures
    line_num = 0
    ans_set = set()
    ans_sequence = {}

    # process input by line
    with ans_f, tqdm(
        total=target, unit='bytes', dynamic_ncols=True, ascii=True
    ) as t:
        for line in ans_f:
            t.update(len(line))
            line_num += 1
            if line_num == 1:
                continue  # skip header
            line = line.strip()
            # parse
            tokens = line.split()
            # assert length == 15
            assert (
                len(tokens) == 7
            ), f'ans token length: {len(tokens)}, expected: 7'
            # get seq_len
            seq_id = f'{tokens[0]}.{tokens[1]}'
            seq_len = id_len_dict[seq_id].len
            si = id_len_dict[seq_id].index
            # get domain index
            domain = tokens[4]
            if domain not in domain_index:
                domain_index[domain] = len(domain_index)
            di = domain_index[domain]
            # add to set
            ans_set.add((si, di))
            # add to sequence
            if si not in ans_sequence:
                ans_sequence[si] = array('i', [0] * seq_len)
            seq_start, seq_end = int(tokens[5]), int(tokens[6])
            ans_sequence[si] = ans_sequence[si][:seq_start - 1] + array(
                'i', [di] * (seq_end - seq_start + 1)
            ) + ans_sequence[si][seq_end:]

    # Accuracy per amino acid
    aa_positive = 0
    aa_total = 0
    with tqdm(
        total=len(ans_sequence), unit='seq', dynamic_ncols=True, ascii=True
    ) as t:
        for si, ans in ans_sequence.items():
            t.update(1)
            aa_total += len(ans)
            if si not in pred_sequence:
                continue
            pred = pred_sequence[si]
            for i in range(len(ans)):
                if ans[i] == pred[i]:
                    aa_positive += 1
    # set stats
    print(f'Computing set stats...')
    aa_accuracy = aa_positive / aa_total
    predicted_positive = len(pred_set)
    answer_positive = len(ans_set)
    true_positive = len(ans_set & pred_set)
    false_positive = len(pred_set - ans_set)
    false_negative = len(ans_set - pred_set)
    precision = true_positive / predicted_positive
    recall = true_positive / answer_positive
    f1_score = 2 * precision * recall / (precision + recall)
    false_discovery_rate = false_positive / predicted_positive
    false_omission_rate = false_negative / answer_positive

    print(
        f'''Results:
         aa_positive: {aa_positive}
            aa_total: {aa_total}
         aa_accuracy: {aa_accuracy:.3%}
  predicted_positive: {predicted_positive}
     answer_positive: {answer_positive}
       true_positive: {true_positive}
      false_positive: {false_positive}
      false_negative: {false_negative}
           precision: {precision:.3%}
              recall: {recall:.3%}
            f1_score: {f1_score:.3%}
false_discovery_rate: {false_discovery_rate:.3%}
 false_omission_rate: {false_omission_rate:.3%}'''
    )

    results = {
        'aa_positive': aa_positive,
        'aa_total': aa_total,
        'aa_accuracy': aa_accuracy,
        'predicted_positive': predicted_positive,
        'answer_positive': answer_positive,
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'false_discovery_rate': false_discovery_rate,
        'false_omission_rate': false_omission_rate
    }
    # save json
    # output_path
    if output_path.suffix == '.gz':
        out_f = gzip.open(output_path, mode='wt', encoding='utf-8')
    else:
        out_f = output_path.open(mode='wt', encoding='utf-8')
    with out_f:
        json.dump(results, out_f, indent=2)

    print(f'Runtime: {time.time() - start_time:.2f} s')
    sys.exit(0)

# Computing pfam classification performance with:
#   Prediction results: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p31_results.tsv
#   Regions answers: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv
#   FASTA sequences: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz
#   Mode: all
#   Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.perf.json
# Results:
#          aa_positive: 102491766,
#             aa_total: 127570640,
#          aa_accuracy: 80.341%
#   predicted_positive: 407499
#      answer_positive: 548956
#        true_positive: 322113
#       false_positive: 85386
#       false_negative: 226843
#            precision: 79.046%
#               recall: 58.677%
#             f1_score: 67.356%
# false_discovery_rate: 20.954%
#  false_omission_rate: 41.323%
# Runtime: 62.12 s

# Computing pfam classification performance with:
#   Prediction results: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567719465_results.tsv
#   Regions answers: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv
#   FASTA sequences: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz
#   Mode: all
#   Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567719465_results.all_regions_perf.json
# Results:
#          aa_positive: 102061621
#             aa_total: 127570640
#          aa_accuracy: 80.004%
#   predicted_positive: 462762
#      answer_positive: 548956
#        true_positive: 367977
#       false_positive: 94785
#       false_negative: 180979
#            precision: 79.518%
#               recall: 67.032%
#             f1_score: 72.743%
# false_discovery_rate: 20.482%
#  false_omission_rate: 32.968%
# Runtime: 63.90 s

# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567765316_results.all_regions_perf.json
# Results:
#          aa_positive: 101053242
#             aa_total: 127570640
#          aa_accuracy: 79.214%
#   predicted_positive: 412228
#      answer_positive: 548956
#        true_positive: 336991
#       false_positive: 75237
#       false_negative: 211965
#            precision: 81.749%
#               recall: 61.388%
#             f1_score: 70.120%
# false_discovery_rate: 18.251%
#  false_omission_rate: 38.612%

# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786205_results.all_regions_perf.json
# Results:
#          aa_positive: 100333880
#             aa_total: 127570640
#          aa_accuracy: 78.650%
#   predicted_positive: 403588
#      answer_positive: 548956
#        true_positive: 330100
#       false_positive: 73488
#       false_negative: 218856
#            precision: 81.791%
#               recall: 60.132%
#             f1_score: 69.309%
# false_discovery_rate: 18.209%
#  false_omission_rate: 39.868%

# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786591_results.all_regions_perf.json
# Results:
#          aa_positive: 101131742
#             aa_total: 127570640
#          aa_accuracy: 79.275%
#   predicted_positive: 422229
#      answer_positive: 548956
#        true_positive: 342135
#       false_positive: 80094
#       false_negative: 206821
#            precision: 81.031%
#               recall: 62.325%
#             f1_score: 70.457%
# false_discovery_rate: 18.969%
#  false_omission_rate: 37.675%

# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786835_results.all_regions_perf.json
# Results:
#          aa_positive: 96949072
#             aa_total: 127570640
#          aa_accuracy: 75.996%
#   predicted_positive: 375359
#      answer_positive: 548956
#        true_positive: 310353
#       false_positive: 65006
#       false_negative: 238603
#            precision: 82.682%
#               recall: 56.535%
#             f1_score: 67.153%
# false_discovery_rate: 17.318%
#  false_omission_rate: 43.465%

# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787010_results.all_regions_perf.json
# Results:
#          aa_positive: 101351206
#             aa_total: 127570640
#          aa_accuracy: 79.447%
#   predicted_positive: 436777
#      answer_positive: 548956
#        true_positive: 352035
#       false_positive: 84742
#       false_negative: 196921
#            precision: 80.598%
#               recall: 64.128%
#             f1_score: 71.426%
# false_discovery_rate: 19.402%
#  false_omission_rate: 35.872%

# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787530_results.all_regions_perf.json
# Results:
#          aa_positive: 101581410
#             aa_total: 127570640
#          aa_accuracy: 79.628%
#   predicted_positive: 465652
#      answer_positive: 548956
#        true_positive: 370552
#       false_positive: 95100
#       false_negative: 178404
#            precision: 79.577%
#               recall: 67.501%
#             f1_score: 73.043%
# false_discovery_rate: 20.423%
#  false_omission_rate: 32.499%

# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787563_results.all_regions_perf.json
# Results:
#          aa_positive: 99506302
#             aa_total: 127570640
#          aa_accuracy: 78.001%
#   predicted_positive: 400792
#      answer_positive: 548956
#        true_positive: 329927
#       false_positive: 70865
#       false_negative: 219029
#            precision: 82.319%
#               recall: 60.101%
#             f1_score: 69.477%
# false_discovery_rate: 17.681%
#  false_omission_rate: 39.899%

# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787595_results.all_regions_perf.json
# Results:
#          aa_positive: 99750620
#             aa_total: 127570640
#          aa_accuracy: 78.192%
#   predicted_positive: 417783
#      answer_positive: 548956
#        true_positive: 339719
#       false_positive: 78064
#       false_negative: 209237
#            precision: 81.315%
#               recall: 61.885%
#             f1_score: 70.281%
# false_discovery_rate: 18.685%
#  false_omission_rate: 38.115%

# Computing pfam classification performance with:
#   Prediction results: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p31_results.tsv
#   Regions answers: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv
#   FASTA sequences: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz
#   Mode: all
#   Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p31_results.p32_regions_perf.json
# Results:
#      answer_positive: 222373
#        true_positive: 17240
#               recall: 7.753%
# Runtime: 48.40 s

# Computing pfam classification performance with:
#   Prediction results: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567719465_results.tsv
#   Regions answers: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv
#   FASTA sequences: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz
#   Mode: all
#   Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567719465_results.p32_regions_perf.json
# Results:
#      answer_positive: 222373
#        true_positive: 74178
#               recall: 33.357%
# Runtime: 45.99 s
# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567765316_results.p32_regions_perf.json
# Results:
#      answer_positive: 222373
#        true_positive: 50486
#               recall: 22.703%
# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786205_results.p32_regions_perf.json
# Results:
#      answer_positive: 222373
#        true_positive: 46044
#               recall: 20.706%
# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786591_results.p32_regions_perf.json
# Results:
#      answer_positive: 222373
#        true_positive: 54728
#               recall: 24.611%
# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567786835_results.p32_regions_perf.json
# Results:
#      answer_positive: 222373
#        true_positive: 35910
#               recall: 16.149%
# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787010_results.p32_regions_perf.json
# Results:
#      answer_positive: 222373
#        true_positive: 63933
#               recall: 28.750%
#  Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787530_results.p32_regions_perf.json
# Results:
#      answer_positive: 222373
#        true_positive: 80572
#               recall: 36.233%
# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787563_results.p32_regions_perf.json
# Results:
#      answer_positive: 222373
#        true_positive: 50891
#               recall: 22.885%
# Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567787595_results.p32_regions_perf.json
# Results:
#      answer_positive: 222373
#        true_positive: 60705
#               recall: 27.299%
