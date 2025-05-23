import os
import re
import csv
import sys
import time
import gzip
import json
import errno
import random
import struct
import msgpack
import logging
import argparse
from tqdm import tqdm
from glob import glob
from pathlib import Path
from statistics import mean
from collections import namedtuple
from collections import defaultdict
from collections import OrderedDict
from multiprocessing import Process, Queue, Value, Manager

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
    # current = 0
    with f as fa_f, tqdm(
        total=target, unit='bytes', dynamic_ncols=True, ascii=True
    ) as t:
        for line in fa_f:
            t.update(len(line))
            line_s = line.strip()
            if len(line_s) > 0 and line_s[0] == '>':
                if seq_id:
                    id_len_dict[seq_id] = seq_len
                    seq_len, seq_id = 0, ''
                # parse header
                seq_id = line_s.split()[0][1:]
            else:
                seq_len += len(line_s)
        if seq_id:  # handle last seq
            id_len_dict[seq_id] = seq_len
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


def verify_indir_path(p):
    # get absolute path
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # doesn't exist
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    # existing file or directory
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


def msgpack_to_tsv(input_queue, output_queue, input_stats, min_region_length=4):
    print('process id:', os.getpid())
    path_count = 0
    input_stats[os.getpid()] = path_count
    for path, in iter(input_queue.get, 'STOP'):
        try:
            path_count += 1
            with path.open(mode='rb') as f:
                seq_list = msgpack.load(f, raw=False)  # return utf-8 stings
            # >>> seq_list[0]
            # {
            #     'id': 'Q9PSC1.1',
            #     'classes': [1, 105, 105, 105, 105],
            #     'top_probs': [
            #         [0.739147, 0.17484, 0.049279],
            #         [0.433389, 0.399197, 0.112347],
            #         [0.622605, 0.217352, 0.120705],
            #         [0.726235, 0.146998, 0.102315],
            #         [0.445134, 0.309731, 0.200903]
            #     ],
            #     'top_classes': [
            #         [1, 105, 12],
            #         [105, 1, 12],
            #         [105, 1, 12],
            #         [105, 12, 1],
            #         [105, 1, 12]
            #     ]
            # }
            # <seqId> <start> <end> <pfamAcc>
            # consider <score> <evalue>
            final_regions = []
            for seq in seq_list:
                class_probs = defaultdict(list)
                seq_len = len(seq['top_probs'])
                top_n = len(seq['top_probs'][0])
                for i in range(seq_len):
                    for j in range(top_n):
                        class_probs[seq['top_classes'][i][j]].append(
                            seq['top_probs'][i][j]
                        )

                class_scores = {}
                for c, probs in class_probs.items():
                    if c == 1:
                        class_scores[c] = 2
                    else:
                        class_scores[c] = mean(probs)

                PROB_MINIMUM = 0.008
                SCORE_THRESHOLD = 0.5
                class_thresholds = {}
                for c, score in class_scores.items():
                    if score > 1 or score < PROB_MINIMUM:
                        class_thresholds[c] = 999
                    elif score > 0.5:
                        class_thresholds[c] = 0.5
                    else:
                        class_thresholds[c] = score * SCORE_THRESHOLD

                LINK_THRESHOLD = 4
                best_classes = []
                for i in range(seq_len):
                    if seq['top_classes'][i][0] == 1 and seq['top_probs'][i][
                        1] > class_thresholds[seq['top_classes'][i][1]]:
                        best_classes.append(seq['top_classes'][i][1])
                    else:
                        best_classes.append(seq['top_classes'][i][0])

                regions = []
                for i in range(seq_len):
                    if len(regions) == 0 and best_classes[i] != 1:
                        regions.append(
                            {
                                'seqId': seq['id'],
                                'start': i + 1,
                                'end': i + 1,
                                'pfamAcc': best_classes[i],
                                'score': class_scores[best_classes[i]]
                            }
                        )
                    elif len(regions) > 0:
                        if best_classes[i] == regions[-1]['pfamAcc']:
                            if i - regions[-1]['end'] < LINK_THRESHOLD:
                                regions[-1]['end'] = i + 1
                            else:
                                regions.append(
                                    {
                                        'seqId': seq['id'],
                                        'start': i + 1,
                                        'end': i + 1,
                                        'pfamAcc': best_classes[i],
                                        'score': class_scores[best_classes[i]]
                                    }
                                )
                        elif best_classes[i] != 1 and best_classes[
                            i] != regions[-1]['pfamAcc']:
                            regions.append(
                                {
                                    'seqId': seq['id'],
                                    'start': i + 1,
                                    'end': i + 1,
                                    'pfamAcc': best_classes[i],
                                    'score': class_scores[best_classes[i]]
                                }
                            )
                final_regions.extend(regions)

            # filter for length >= 5
            output_queue.put(
                (
                    [
                        r for r in final_regions
                        if (r['end'] - r['start']) > (min_region_length - 2)
                    ], len(seq_list)
                )
            )
            # logging.info(f"Input: {'/'.join(path.parts[-2:])}")
        except Exception as exc:
            print(exc)
            logging.info(
                f"ERROR: Input {os.getpid()}: {'/'.join(path.parts[-2:])}"
            )
            # put input back into predict_queue
            time.sleep(5)
            input_queue.put((path, ))
    # # last worker to finish
    # this doesn't work
    # with done_counter.get_lock():
    #     done_counter.value += 1
    #     logging.info(f"Input {done_counter.value} Done")
    #     if done_counter.value == total_workers:
    #         logging.info(f"Input STOP")
    #         output_queue.put('STOP')
    input_stats[os.getpid()] = path_count
    # logging.info(f"path_count {os.getpid()}: {path_count}")


def tsv_writer(output_queue, output_path, meta_path):
    print('process id:', os.getpid())
    # read meta.json
    with meta_path.open(mode='rt') as f:
        meta = json.load(f)

    fieldnames = ['seqId', 'start', 'end', 'pfamAcc', 'score']
    with output_path.open(mode='wt', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter='\t', lineterminator='\n'
        )
        writer.writeheader()
        i = 0
        total = 0
        for regions, seq_num in iter(output_queue.get, 'STOP'):
            i += 1
            total += seq_num
            for region in regions:
                region['pfamAcc'] = meta['domain_list'][region['pfamAcc']]
                writer.writerow(region)
            logging.info(f"Output {i}")
        logging.info(f"Total seq {total}, batches {i}")
    logging.info(f"Output Exit")


# v1: implementation with concurrent.futures.ThreadPoolExecutor
# v2: speed improvement, implementation with multiprocessing.Process
# v3: new scoring algorithm
if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description=
        'Convert ANNotate Pfam msgpack output directory to Pfam regions TSV file v3 with scores.'
    )
    parser.add_argument(
        '-i',
        '--indir',
        type=str,
        required=True,
        help="Path to the ANNotate Pfam msgpack directory, required."
    )
    parser.add_argument(
        '-m',
        '--meta',
        type=str,
        required=True,
        help=
        "Path to the meta.json file generated by datasets/pfam_regions_build.py for mapping index to pfamAcc, required."
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help="Path to the output regions TSV file, required."
    )
    parser.add_argument(
        '-n',
        '--workers',
        type=int,
        default=4,
        help='Number of processing workers. (default: %(default)s)'
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    # verify paths
    indir_path = verify_indir_path(args.indir)
    meta_path = verify_input_path(args.meta)
    output_path = verify_output_path(args.output)

    # read input fasta paths
    # in_paths = sorted([p for p in indir_path.glob('*.msgpack')])
    with Manager() as manager:
        # proxy objects
        input_stats = manager.dict()

        # create queues
        input_queue = Queue(1000)
        output_queue = Queue(1000)

        # input process
        input_processes = []
        for i in range(args.workers):
            ip = Process(
                target=msgpack_to_tsv,
                args=(input_queue, output_queue, input_stats)
            )
            ip.start()
            input_processes.append(ip)

        # output process
        op = Process(
            target=tsv_writer, args=(output_queue, output_path, meta_path)
        )
        op.start()

        # submit input tasks
        input_queue_i = 0
        for fa_path in indir_path.glob('*.msgpack'):
            input_queue_i += 1
            input_queue.put((fa_path, ))
        for i in range(args.workers):
            input_queue.put('STOP')

        # join
        try:
            for ip in input_processes:
                ip.join()
            output_queue.put('STOP')
            op.join()
        except Exception as exc:
            logging.info(f"exception: {str(exc)}")
        finally:
            # for pid, count in input_stats.items():
            #     logging.info(f"input_stats {pid}: {count}")
            # logging.info(f"input_queue_i: {input_queue_i}")
            print(f'Runtime: {time.time() - start_time:.2f} s')
            sys.exit(0)

# python ./util/pfam/annotate_pfam_batch_msgpack_to_tsv.py  --indir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_fa_split_batched/pfam31_1567719465_results_raw --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --output /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.ann31_1567719465_results.tsv --workers 4
# Runtime: 462.16 s

# python ./util/pfam/annotate_pfam_batch_msgpack_to_tsv.py  --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_batched_34000/pfam31_1567889661_results_raw --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.ann31_1567889661_results.tsv --workers 4

# v1 Runtime: 470.25 s
# PYTHON=/home/hotdogee/venv/tf37/bin/python
# SCRIPTPATH=/home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/annotate_pfam_batch_msgpack_to_tsv_v2.py
# WORKERS=12
# VERSION=1567719465
# VERSION=1567787530
# VERSION=1567787595
# ${PYTHON} ${SCRIPTPATH} --indir /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_fa_split_batched/pfam31_${VERSION}_results_raw --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --output /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains.ann31_${VERSION}_results_v2_test.tsv --workers ${WORKERS}
# v2 Runtime: 309.70 s (1 worker @8086K)
# v2 Runtime: 152.60 s (2 worker @8086K)
# v2 Runtime: 81.84 s (4 worker @8086K)
# v2 Runtime: 54.39 s (6 worker @8086K)
# v2 Runtime: 36.11 s (12 worker @8086K)
# v2 Runtime: 19.35 s (24 worker @2920X)
# v2 Runtime: 34.80 s (12 worker @2920X)
# v2 Runtime: 59.91 s (6 worker @2920X)

# PYTHON=/home/hotdogee/venv/tf37/bin/python
# SCRIPTPATH=/home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/annotate_pfam_batch_msgpack_to_tsv_v2.py
# ${PYTHON} ${SCRIPTPATH} --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_batched_34000/pfam31_1567889661_results_raw --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.ann31_1567889661_results.tsv --workers 24
# Runtime: 3689.20 s (24 worker @2920X)
# cp /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.ann31_1567889661_results.tsv /data12/pfam/

# PYTHON=/home/hotdogee/venv/tf37/bin/python
# SCRIPTPATH=/home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/annotate_pfam_batch_msgpack_to_tsv_v2.py
# ${PYTHON} ${SCRIPTPATH} --indir /data12/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_batched_34000/pfam31_1567889661_results_raw --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.ann31_1567889661_results.tsv --workers 12

# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/annotate_pfam_batch_msgpack_to_tsv_v2.py  --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_batched_25000/pfam31_1567787530_results_raw --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.ann31_1567787530_results.tsv --workers 24
# Total Seq 38168024

# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/annotate_pfam_batch_msgpack_to_tsv_v2.py  --indir /home/hotdogee/pfam/test_msgpack --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --output /home/hotdogee/pfam/test_msgpack.tsv --workers 24
