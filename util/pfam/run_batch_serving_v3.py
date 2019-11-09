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
from tqdm import tqdm
from glob import glob
from pathlib import Path
from collections import namedtuple
from collections import defaultdict
from collections import OrderedDict
from multiprocessing import Process, Queue

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


def input_worker(input_queue, predict_queue):
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    for fa_path, out_path in iter(input_queue.get, 'STOP'):
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

        input_data = json.dumps({'instances': [e['seq'] for e in seq_list]})
        predict_queue.put((input_data, seq_list, out_path))
        # logging.info(f"Input: {'/'.join(fa_path.parts[-2:])}")


def predict_worker(predict_queue, output_queue, server):
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    for input_data, seq_list, out_path in iter(predict_queue.get, 'STOP'):
        try:
            r = requests.post(server, data=input_data)
            output_data = r.json()['predictions']
            r.close()
            output_queue.put((output_data, seq_list, out_path))
            # logging.info(f"Predict-{device}: {'/'.join(out_path.parts[-2:])}")
            # free mem
            del input_data
        except Exception as exc:
            print(exc)
            logging.info(
                f"ERROR: Predict {server}: {'/'.join(out_path.parts[-2:])}"
            )
            # put input back into predict_queue
            time.sleep(5)
            predict_queue.put((input_data, seq_list, out_path))


def output_worker(output_queue):
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    for output_data, seq_list, out_path in iter(output_queue.get, 'STOP'):
        for i in range(len(seq_list)):
            seq_len = len(seq_list[i]['seq'])
            del seq_list[i]['seq']
            seq_list[i]['classes'] = output_data[i]['classes'][:seq_len]
            seq_list[i]['top_probs'] = output_data[i]['top_probs'][:seq_len]
            seq_list[i]['top_classes'] = output_data[i]['top_classes'][:seq_len]

        # print(f"{seq_list[0]['id']}:", seq_list[0]['classes'])
        # write output file
        with out_path.open(mode='wb') as f:
            msgpack.dump(seq_list, f)

        # free mem
        for i in range(len(seq_list)):
            del seq_list[i]['classes']
            del seq_list[i]['top_probs']
            del seq_list[i]['top_classes']
            del output_data[i]['classes']
            del output_data[i]['top_probs']
            del output_data[i]['top_classes']
        del seq_list
        del output_data

        # log
        logging.info(f"Output: {'/'.join(out_path.parts[-2:])}")


# ubuntu 18.04
# source /home/hotdogee/venv/tf37/bin/activate
# cp -r /data12/checkpoints/pfam-regions-d0-s0/v4-BiRnn/FullGru512x4_hw512_rnndrop_TITANRTX_8086K1-1.2/export/1567787530 /home/hotdogee/models/pfam1/

# 2920X
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_serving_v2.py --indir /home/hotdogee/pfam/test_p32_25000 --outdir /home/hotdogee/pfam/test_p32_25000/pfam31_1567787530_results_raw --servers http://localhost:8501/v1/models/pfam:predict --readers 1 --writers 1

# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_serving_v2.py --indir /home/hotdogee/pfam/test_p32_25000 --outdir /home/hotdogee/pfam/test_p32_25000/pfam31_1567787530_results_raw --servers http://localhost:8501/v1/models/pfam:predict,http://localhost:8601/v1/models/pfam:predict,http://localhost:8701/v1/models/pfam:predict,http://192.168.1.33:8501/v1/models/pfam:predict,http://192.168.1.33:8601/v1/models/pfam:predict,http://192.168.1.74:8501/v1/models/pfam:predict,http://192.168.1.74:8601/v1/models/pfam:predict --readers 8 --writers 8

# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_serving_v2.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_batched_25000 --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_batched_25000/pfam31_1567787530_results_raw --servers http://localhost:8501/v1/models/pfam:predict,http://localhost:8601/v1/models/pfam:predict,http://localhost:8701/v1/models/pfam:predict,http://192.168.1.33:8501/v1/models/pfam:predict,http://192.168.1.33:8601/v1/models/pfam:predict,http://192.168.1.74:8501/v1/models/pfam:predict,http://192.168.1.74:8601/v1/models/pfam:predict --readers 12 --writers 12
# 38168103 seqs, 571180 batches
# 272s (2323 batches) (8.54 batches/sec)
# 880s (8674 batches) (9.85 batches/sec) after switching to generator
# 24430s (304610 batches) (12.46 batches/sec) (832 seqs/sec)
# fix corrupt p32_seqs_with_p32_regions_of_p31_domains_2.157070.pfam31_results_raw.msgpack
# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_serving_v2.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_batched_25000 --outdir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_fa_split_batched_25000/pfam31_1567787530_results_raw --servers http://localhost:8501/v1/models/pfam:predict --readers 1 --writers 1

# python /home/hotdogee/Dropbox/Work/Btools/ANNotate/ANNotate2/util/pfam/run_batch_serving_v3.py --indir /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_batched_34000 --outroot /home/hotdogee/pfam/p32_seqs_with_p32_regions_of_p31_domains_2_n10000_fa_batched_34000 --outname ann_{version}_results_raw --servers http://localhost:8501/v1/models/pfam:predict --readers 1 --writers 1

if __name__ =='__main__':
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
        '-s',
        '--servers',
        type=str,
        required=True,
        help=
        'A comma separated list of tensorflow serving server REST URLs, ex: "http://localhost:8501/v1/models/pfam:predict,http://localhost:8601/v1/models/pfam:predict".'
    )
    parser.add_argument(
        '-r',
        '--readers',
        type=int,
        default=4,
        help='Number of input workers. (default: %(default)s)'
    )
    parser.add_argument(
        '-w',
        '--writers',
        type=int,
        default=4,
        help='Number of output workers. (default: %(default)s)'
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

    # create queues
    input_queue = Queue(1000)
    predict_queue = Queue(1000)
    output_queue = Queue(1000)

    # servers
    servers = args.servers.split(',')

    # input process
    input_processes = []
    for i in range(args.readers):
        ip = Process(target=input_worker, args=(input_queue, predict_queue))
        ip.start()
        input_processes.append(ip)

    # predict process
    predict_processes = []
    for server in servers:
        pp = Process(
            target=predict_worker, args=(predict_queue, output_queue, server)
        )
        pp.start()
        predict_processes.append(pp)

    # output process
    output_processes = []
    for i in range(args.writers):
        op = Process(target=output_worker, args=(output_queue, ))
        op.start()
        output_processes.append(op)

    # submit input tasks

    # read input fasta paths excluding those with existing output
    # in_paths = sorted(
    #     [p for p in indir_path.glob('*.fa') if p.stem not in existing_stems]
    # )
    for fa_path in indir_path.glob('*.fa'):
        if fa_path.stem in existing_stems:
            continue
        out_path = outdir_path / f'{fa_path.stem}.pfam31_results_raw.msgpack'
        input_queue.put((fa_path, out_path))
    for i in range(args.readers):
        input_queue.put('STOP')

    # join
    try:
        for ip in input_processes:
            # logging.info(f"ip.join()")
            ip.join()
        # push predict stop tokens after all input processes have exited
        for server in servers:
            predict_queue.put('STOP')

        for pp in predict_processes:
            # logging.info(f"pp.join()")
            pp.join()
        # push output stop tokens after all predict processes have exited
        for i in range(args.writers):
            output_queue.put('STOP')

        for op in output_processes:
            # logging.info(f"op.join()")
            op.join()
    except Exception as exc:
        logging.info(f"exception: {str(exc)}")
        sys.exit(0)

    logging.info(f'Runtime: {time.time() - start_time:.2f} s')
    sys.exit(0)

    # Load model from export directory, and make a predict function.
    # predict = tf.contrib.predictor.from_saved_model(str(modeldir_path))
    # # >>> predict.feed_tensors
    # # {'protein_sequences': <tf.Tensor 'input_protein_string_tensor:0' shape=(?,) dtype=string>}
    # input_dict = {'protein_sequences': ['FIV', 'LMP']}
    # input_dict = {'protein_sequences': ['FLIM', 'VP', 'AWGST']}
    # predict(input_dict)
    # {
    #     'top_classes': array([
    #         [
    #             [1, 9580, 13679],
    #             [1, 9580, 13679],
    #             [1, 366, 28],
    #             [1, 28, 39],
    #             [1, 295, 371]
    #         ],
    #         [
    #             [1, 9211, 615],
    #             [1, 9211, 9612],
    #             [1, 295, 273],
    #             [1, 273, 295],
    #             [1, 273, 295]
    #         ],
    #         [
    #             [1, 295, 1204],
    #             [1, 295, 1204],
    #             [1, 295, 1204],
    #             [1, 295, 273],
    #             [1, 295, 273]
    #         ]
    #     ], dtype = int32),
    #     'top_probs': array([
    #         [
    #             [9.9756598e-01, 1.0554678e-04, 6.7393841e-05],
    #             [9.9559766e-01, 9.8593999e-05, 8.5413361e-05],
    #             [9.9482805e-01, 1.1286281e-04, 1.0071106e-04],
    #             [9.9752134e-01, 1.0686450e-04, 8.4549436e-05],
    #             [9.9998784e-01, 9.4184279e-06, 2.0532018e-06]
    #         ],
    #         [
    #             [9.9867058e-01, 7.3625270e-05, 4.8228339e-05],
    #             [9.9252945e-01, 5.6753954e-04, 3.9153339e-04],
    #             [9.9996555e-01, 1.8193850e-05, 8.1562621e-06],
    #             [9.9994874e-01, 4.9559905e-05, 1.6222458e-06],
    #             [9.9996805e-01, 3.0887117e-05, 1.0678857e-06]
    #         ],
    #         [
    #             [9.9970752e-01, 2.7202181e-05, 9.1542242e-06],
    #             [9.9979395e-01, 3.5809506e-05, 9.2555129e-06],
    #             [9.9981719e-01, 6.7291963e-05, 7.3239585e-06],
    #             [9.9990547e-01, 8.7760120e-05, 1.9479849e-06],
    #             [9.9996436e-01, 3.5095421e-05, 4.0521169e-07]
    #         ]
    #     ], dtype = float32),
    #     'classes': array([
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1]
    #     ], dtype = int32)
    # }
