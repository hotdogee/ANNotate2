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


def verify_input_or_dir_path(p):
    # get absolute path
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # doesn't exist
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    # existing file or directory
    if path.is_dir():  # got existing directory
        assert len([x for x in path.iterdir()]) != 0, 'Directory is empty'
    return path


if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description='Compute pfam_scan.pl performance metrics.'
    )
    parser.add_argument(
        '-p',
        '--pred',
        type=str,
        required=True,
        help=
        "Path to the predicted TSV file or directory of predicted TSV files, required."
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
        help="Path to output performance metrics to in JSON format, required."
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
    pred_path = verify_input_or_dir_path(args.pred)
    ans_path = verify_input_path(args.ans)
    fasta_path = verify_input_path(args.fasta)
    output_path = verify_output_path(args.output)
    logging.info(
        f"""Computing pfam_scan classification performance with:
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

    # initialize data structures
    domain_index = {'NO_DOMAIN': 0}
    pred_set = set()
    pred_sequence = {}

    if pred_path.is_dir():
        pred_paths = pred_path.glob('*.tsv')
    else:
        pred_paths = [pred_path]
    for pred_p in pred_paths:
        print(f"Parsing {'/'.join(pred_p.parts[-2:])}")
        pred_f, target = _open_input_path(pred_p)
        # process input by line
        with pred_f:
            for line in pred_f:
                line = line.strip()
                if len(line) == 0 or line[0] == '#':
                    continue
                # parse
                tokens = line.split()
                # assert length == 15
                assert (
                    len(tokens) == 15
                ), f'pred token length: {len(tokens)}, expected: 15'
                # get seq_len
                seq_id = tokens[0]
                seq_len = id_len_dict[seq_id].len
                si = id_len_dict[seq_id].index
                # get domain index
                domain = tokens[5].split('.')[0]
                if domain not in domain_index:
                    domain_index[domain] = len(domain_index)
                di = domain_index[domain]
                # add to set
                pred_set.add((si, di))
                # add to sequence
                if si not in pred_sequence:
                    pred_sequence[si] = array('i', [0] * seq_len)
                seq_start, seq_end = int(tokens[3]), int(tokens[4])
                pred_sequence[si] = pred_sequence[si][:seq_start - 1] + array(
                    'i', [di] * (seq_end - seq_start + 1)
                ) + pred_sequence[si][seq_end:]
    logging.info(f"Pred has {len(pred_sequence)} seqs")

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
    logging.info(f"Ans has {len(ans_sequence)} seqs")

    # Accuracy per amino acid
    aa_positive = 0
    aa_total = 0
    with tqdm(
        total=len(ans_sequence), unit='seq', dynamic_ncols=True, ascii=True
    ) as t:
        for seq_id, ans in ans_sequence.items():
            t.update(1)
            aa_total += len(ans)
            if seq_id not in pred_sequence:
                continue
            pred = pred_sequence[seq_id]
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

# windows

# python .\util\pfam\compute_pfam_scan_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p31_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p31_results.all_regions_perf.json

# python .\util\pfam\compute_pfam_scan_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p31_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p31_results.p32_regions_perf.json

# python .\util\pfam\compute_pfam_scan_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.p31_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.n100.tsv.gz --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.n100.perf.json

# python .\util\pfam\compute_pfam_scan_classification_performance.py  --pred D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.p31_results.tsv --ans D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.n100.tsv.gz --fasta D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.fa.gz --output D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.n100.perf.json --mode p32

# python .\util\pfam\compute_pfam_scan_classification_performance.py  --pred D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.n100.p31_results.tsv --ans D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.all_regions.n100.tsv.gz --fasta D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.n100.fa.gz --output D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.all_regions.n100.perf.json

# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/compute_pfam_scan_classification_performance.py  --pred /data12/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed/p31_results --ans /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv  --fasta /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.fa --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.p31_results.all_regions_perf.json

# python .\util\pfam\compute_pfam_scan_classification_performance.py  --pred D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed/p31_results --ans D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv  --fasta D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa --output D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.p31_results.all_regions_perf.json

# Computing pfam_scan classification performance with:
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

# p31_seqs_with_p32_regions_of_p31_domains.p31_results.p32_regions_perf.json
#   "answer_positive": 222373,
#   "true_positive": 17240,
#   "recall": 0.07752739766068723,

# Computing pfam_scan classification performance with:
#   Prediction results: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.p31_results.tsv
#   Regions answers: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.n100.tsv.gz
#   FASTA sequences: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.fa.gz
#   Mode: all
#   Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.n100.perf.json
# Results:
#          aa_positive: 44050,
#             aa_total: 56506,
#          aa_accuracy: 77.956%
#   predicted_positive: 191
#      answer_positive: 260
#        true_positive: 155
#       false_positive: 36
#       false_negative: 105
#            precision: 81.152%
#               recall: 59.615%
#             f1_score: 68.736%
# false_discovery_rate: 18.848%
#  false_omission_rate: 40.385%

# Computing pfam_scan classification performance with:
#   Prediction results: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.p31_results.tsv
#   Regions answers: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.n100.tsv.gz
#   FASTA sequences: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.n100.fa.gz
#   Mode: p32
#   Output: Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.n100.perf.json
# Results:
#      answer_positive: 105
#        true_positive: 6
#               recall: 5.714%

# Computing pfam_scan classification performance with:
#   Prediction results: Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.n100.p31_results.tsv
#   Regions answers: Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.all_regions.n100.tsv.gz
#   FASTA sequences: Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.n100.fa.gz
#   Mode: all
#   Output: Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.all_regions.n100.perf.json
# Results:
#          aa_positive: 37631,
#             aa_total: 38631,
#          aa_accuracy: 97.411%
#   predicted_positive: 160
#      answer_positive: 163
#        true_positive: 160
#       false_positive: 0
#       false_negative: 3
#            precision: 100.000%
#               recall: 98.160%
#             f1_score: 99.071%
# false_discovery_rate: 0.000%
#  false_omission_rate: 1.840%

# merge prep
# cp -r /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed/p31_results /data12/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed/
# cp -r /data12/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed/p31_results /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_distributed/
# Pred has 38,073,253 seqs
#  Ans has 38,168,102 seqs
# Results:
#          aa_positive: 13856742256
#             aa_total: 14102519533
#          aa_accuracy: 98.257%
#   predicted_positive: 53323162
#      answer_positive: 54177531
#        true_positive: 52985285
#       false_positive: 337877
#       false_negative: 1192246
#            precision: 99.366%
#               recall: 97.799%
#             f1_score: 98.577%
# false_discovery_rate: 0.634%
#  false_omission_rate: 2.201%
# Runtime: 3206.23 s

################# n1000,n10000,n100000 #################

# Computing pfam_scan classification performance with:
#   Prediction results: p32_seqs_with_p32_regions_of_p31_domains_2_n1000_fa_distributed/p31_results
#   Regions answers: pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n1000.tsv
#   FASTA sequences: pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n1000.fa
#   Mode: all
#   Output: pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n1000.p31_results.all_regions_perf.json
# Pred has 997 seqs
# Ans has 1000 seqs
# Results:
#          aa_positive: 359924
#             aa_total: 368804
#          aa_accuracy: 97.592%
#   predicted_positive: 1395
#      answer_positive: 1431
#        true_positive: 1390
#       false_positive: 5
#       false_negative: 41
#            precision: 99.642%
#               recall: 97.135%
#             f1_score: 98.372%
# false_discovery_rate: 0.358%
#  false_omission_rate: 2.865%
# Runtime: 0.10 s

# Computing pfam_scan classification performance with:
#   Prediction results: p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_distributed/p31_results
#   Regions answers: pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n10000.tsv
#   FASTA sequences: pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n10000.fa
#   Mode: all
#   Output: pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000.p31_results.all_regions_perf.json
# Pred has 9976 seqs
# Ans has 10000 seqs
# Results:
#          aa_positive: 3629603
#             aa_total: 3691975
#          aa_accuracy: 98.311%
#   predicted_positive: 13920
#      answer_positive: 14142
#        true_positive: 13830
#       false_positive: 90
#       false_negative: 312
#            precision: 99.353%
#               recall: 97.794%
#             f1_score: 98.567%
# false_discovery_rate: 0.647%
#  false_omission_rate: 2.206%
# Runtime: 0.97 s

# Computing pfam_scan classification performance with:
#   Prediction results: p32_seqs_with_p32_regions_of_p31_domains_2_n100000_fa_distributed/p31_results
#   Regions answers: pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n100000.tsv
#   FASTA sequences: pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n100000.fa
#   Mode: all
#   Output: pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n100000.p31_results.all_regions_perf.json
# Pred has 99772 seqs
# Ans has 100000 seqs
# Results:
#          aa_positive: 36303124
#             aa_total: 36950452
#          aa_accuracy: 98.248%
#   predicted_positive: 140009
#      answer_positive: 142200
#        true_positive: 139094
#       false_positive: 915
#       false_negative: 3106
#            precision: 99.346%
#               recall: 97.816%
#             f1_score: 98.575%
# false_discovery_rate: 0.654%
#  false_omission_rate: 2.184%
# Runtime: 10.24 s
