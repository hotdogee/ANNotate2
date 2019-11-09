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


def verify_output_path(p, can_exist=False):
    # get absolute path to dataset directory
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # existing file
    if not can_exist and path.exists():
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


def msgpack_to_domains(
    input_queue, output_queue, id_len_dict, input_stats, top=1
):
    print('process id:', os.getpid())
    # print(input_queue, output_queue, input_stats, top)
    path_count = 0
    # input_stats[os.getpid()] = path_count
    for path, in iter(input_queue.get, 'STOP'):
        # print(f'path_count: {path_count}, {path}')
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
            pred_set = set()
            for seq in seq_list:
                si = id_len_dict[seq['id']].index
                pred_set |= set(
                    [(si, c) for tc in seq['top_classes'] for c in tc[:top]]
                )

            # filter for length >= 5
            output_queue.put((pred_set, len(seq_list)))
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
    # input_stats[os.getpid()] = path_count
    # logging.info(f"path_count {os.getpid()}: {path_count}")


def collect_seq_domains(output_queue, return_proxy):
    print('process id:', os.getpid())
    i = 0
    total = 0
    pred_set = set()
    for pred_set_part, seq_num in iter(output_queue.get, 'STOP'):
        i += 1
        total += seq_num
        pred_set |= pred_set_part
        # logging.info(f"Output {i}")
    logging.info(f"Total seq {total}, batches {i}, pred_set = {len(pred_set)}")
    return_proxy['pred_set'] = pred_set


if __name__ =='__main__':
    # goal: examine the effect on recall after adding 2ed and 3rd classes
    parser = argparse.ArgumentParser(
        description=
        'Convert ANNotate Pfam msgpack output directory to Pfam regions TSV file.'
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
        default='',
        help="Path to the JSON file for saving results."
    )
    parser.add_argument(
        '-t',
        '--outtsv',
        type=str,
        default='',
        help="Path to the TSV file for appending results."
    )
    parser.add_argument(
        '-n',
        '--top',
        type=int,
        default=1,
        help="Include the top N predicted domains. (default: %(default)s)"
    )
    parser.add_argument(
        '-r',
        '--readers',
        type=int,
        default=4,
        help='Number of input processing workers. (default: %(default)s)'
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    # verify paths
    indir_path = verify_indir_path(args.indir)
    meta_path = verify_input_path(args.meta)
    ans_path = verify_input_path(args.ans)
    fasta_path = verify_input_path(args.fasta)
    output_path = None
    if args.output:
        output_path = verify_output_path(args.output, can_exist=True)
    outtsv_path = None
    if args.outtsv:
        outtsv_path = verify_output_path(args.outtsv, can_exist=True)

    # read meta.json
    with meta_path.open(mode='rt') as f:
        meta = json.load(f)

    # parse fasta into {seq_id: seq_len}
    id_len_dict = _fa_gz_to_id_len_dict(fasta_path)
    seq_count = len(id_len_dict)
    logging.info(
        f"Parsed {seq_count} seqs from {'/'.join(fasta_path.parts[-2:])}"
    )

    # parse ans
    # coordinates are 1-inclusive
    ans_f, target = _open_input_path(ans_path)

    # initialize data structures
    line_num = 0
    ans_set = set()

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
            if domain not in meta['domain_index']:
                meta['domain_index'][domain] = len(meta['domain_index'])
            di = meta['domain_index'][domain]
            # add to set
            ans_set.add((si, di))

    # read input fasta paths
    # in_paths = sorted([p for p in indir_path.glob('*.msgpack')])
    with Manager() as manager:
        # proxy objects
        input_stats = manager.dict()
        return_proxy = manager.dict()

        # create queues
        input_queue = Queue(1000)
        output_queue = Queue(1000)

        # input process
        input_processes = []
        for i in range(args.readers):
            ip = Process(
                target=msgpack_to_domains,
                args=(
                    input_queue, output_queue, id_len_dict, input_stats,
                    args.top
                )
            )
            ip.start()
            input_processes.append(ip)

        # output process
        op = Process(
            target=collect_seq_domains, args=(output_queue, return_proxy)
        )
        op.start()

        # submit input tasks
        input_queue_i = 0
        for fa_path in indir_path.glob('*.msgpack'):
            input_queue_i += 1
            input_queue.put((fa_path, ))
        for i in range(args.readers):
            input_queue.put('STOP')

        # join
        try:
            for ip in input_processes:
                ip.join()
            output_queue.put('STOP')
            op.join()
            pred_set = return_proxy['pred_set']

            # set stats
            print(len(ans_set))
            print(len(pred_set))
            print(f'Computing set stats...')
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

            export_version = re.findall(r'\w+31_(\d+)_results', indir_path.name)
            if export_version:
                export_version = export_version[0]
            else:
                export_version = 'N/A'

            print(
                f'''Results:
     prediction_name: {indir_path.name}
      export_version: {export_version}
       top_N_domains: {args.top}
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

            results = OrderedDict(
                {
                    'prediction_name': indir_path.name,
                    'export_version': export_version,
                    'top_N_domains': args.top,
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
            )

            # save json
            if output_path:
                if output_path.exists():
                    with output_path.open(mode='rt', encoding='utf-8') as f:
                        res = json.load(f)
                        if type(res) is list:
                            res.append(results)
                        else:
                            res = [res, results]
                else:
                    res = [results]
                with output_path.open(mode='wt', encoding='utf-8') as f:
                    json.dump(res, f, indent=2)

            # save tsv
            if outtsv_path:
                fieldnames = list(results.keys())
                output_exists = outtsv_path.exists()
                with outtsv_path.open(
                    mode='at', newline='', encoding='utf-8'
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=fieldnames,
                        delimiter='\t',
                        lineterminator='\n'
                    )
                    if not output_exists:
                        writer.writeheader()
                    writer.writerow(results)
        except Exception as exc:
            logging.info(f"exception: {str(exc)}")
        finally:
            # for pid, count in input_stats.items():
            #     logging.info(f"input_stats {pid}: {count}")
            # logging.info(f"input_queue_i: {input_queue_i}")
            print(f'Runtime: {time.time() - start_time:.2f} s')
            sys.exit(0)

# run_p32_n10000-part4.sh
# #!/bin/bash
# DATADIR=/data12/datasets3
# MODELDIR=/data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_TITANV_W2125-2.4
# VERSIONS=(
# 1568346278
# 1568346287
# 1568346297
# 1568346306
# 1568346315
# 1568346325
# 1568346336
# 1568346347
# 1568346356
# 1568346366
# 1568346376
# 1568346383
# 1568346391
# 1568346397
# 1568346403
# 1567719465
# )
# for VERSION in "${VERSIONS[@]}"
# do
#     /home/hotdogee/venv/tf37/bin/python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/annotate_pfam_batch_msgpack_to_n_classification_performance.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_batched_25000/ann31_${VERSION}_results_raw --meta /home/hotdogee/pfam/pfam-regions-d0-s0-meta.json --ans /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains.all_regions.n10000.tsv --fasta /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2.n10000.fa --output /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000.ann31_top2.all_regions_perf.json --outtsv /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000.ann31_top2.all_regions_perf.tsv --top 2 --readers 24
# done
# # run on 4960X
# # /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/run/predict/20190913-p32-n1000/run_p32_n10000-part4.sh
