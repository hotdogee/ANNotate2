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
from multiprocessing import Process, Queue

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


def serving_predict_fasta(fa_path, predict):
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

    input = {'protein_sequences': [e['seq'] for e in seq_list]}
    output = predict(input)

    return seq_list, output


def input_worker(input_queue, predict_queue, devices):
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

        input = {'protein_sequences': [e['seq'] for e in seq_list]}
        predict_queue.put((input, seq_list, out_path))
        logging.info(f"Input: {'/'.join(fa_path.parts[-2:])}")
    for device in devices:
        predict_queue.put('STOP')


def predict_worker(predict_queue, output_queue, modeldir_path, device='0'):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    predict = tf.contrib.predictor.from_saved_model(str(modeldir_path))
    for input, seq_list, out_path in iter(predict_queue.get, 'STOP'):
        output = predict(input)
        output_queue.put((output, seq_list, out_path))
        logging.info(f"Predict-{device}: {'/'.join(out_path.parts[-2:])}")
    if predict_queue.empty():  # last worker to finish
        logging.info(f"Predict-{device}: STOP")
        output_queue.put('STOP')
    else:
        logging.info(f"Predict-{device}: FINISH")


def output_worker(output_queue):
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    for output, seq_list, out_path in iter(output_queue.get, 'STOP'):
        for i in range(len(seq_list)):
            seq_len = len(seq_list[i]['seq'])
            del seq_list[i]['seq']
            seq_list[i]['classes'] = output['classes'][i][:seq_len].tolist()
            seq_list[i]['top_probs'] = output['top_probs'][i][:seq_len].tolist()
            seq_list[i]['top_classes'] = output['top_classes'][i][:seq_len
                                                                 ].tolist()

        # print(f"{seq_list[0]['id']}:", seq_list[0]['classes'])
        # write output file
        f = out_path.open(mode='wb')
        with f:
            msgpack.dump(seq_list, f)
        logging.info(f"Output: {'/'.join(out_path.parts[-2:])}")


# ubuntu 18.04
# source /home/hotdogee/venv/tf37/bin/activate

# 8086K1
# python ./util/pfam/run_batch_predictor.py --indir /home/hotdogee/pfam/test3 --outdir /home/hotdogee/pfam/test3/pfam31_1567884207_results_raw --workers 1 --modeldir /home/hotdogee/models/pfam2/1567884207
# Runtime: 23.41 s

# 4960X
# python ./util/pfam/run_batch_predictor.py --indir /home/hotdogee/pfam/test3 --outdir /home/hotdogee/pfam/test3/pfam31_1567889661_results_raw --workers 1 --modeldir /home/hotdogee/models/pfam2/1567889661
# Runtime: 22.88 s (predict + input processing)
# Runtime: 22.69 s (predict only)

# Implement multi GPU
# python ./util/pfam/run_batch_predictor.py --indir /home/hotdogee/pfam/test3 --outdir /home/hotdogee/pfam/test3/pfam31_1567889661_results_raw --modeldir /home/hotdogee/models/pfam2/1567889661 --devices 0,1,2
# Runtime: 30.28 s (1x 1080Ti)
# Runtime: 19.12 s (2x 1080Ti)
# Runtime: 16.14 s (3x 1080Ti)

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
        '-m',
        '--modeldir',
        type=str,
        default='/home/hotdogee/models/pfam1/1567719465',
        help="Path to the saved model directory. (default: %(default)s)"
    )
    parser.add_argument(
        '-d',
        '--devices',
        type=str,
        default='0',
        help=
        'A comma separated list of CUDA devices to use, ex: "0,1,2". (default: %(default)s)'
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    # verify paths
    indir_path = verify_indir_path(args.indir)
    modeldir_path = verify_indir_path(args.modeldir)
    outdir_path = verify_outdir_path(args.outdir, required_empty=False)

    # read existing output paths
    existing_stems = set(
        [Path(p.stem).stem for p in outdir_path.glob('*.msgpack')]
    )

    # read input fasta paths excluding those with existing output
    in_paths = sorted(
        [p for p in indir_path.glob('*.fa') if p.stem not in existing_stems]
    )

    # create queues
    input_queue = Queue()
    predict_queue = Queue()
    output_queue = Queue()

    # devices
    devices = args.devices.split(',')

    # input process
    ip = Process(
        target=input_worker, args=(input_queue, predict_queue, devices)
    )
    ip.start()

    # predict process
    for device in args.devices.split(','):
        pp = Process(
            target=predict_worker,
            args=(predict_queue, output_queue, modeldir_path, device)
        )
        pp.start()

    # output process
    op = Process(target=output_worker, args=(output_queue, ))
    op.start()

    # submit input tasks
    for fa_path in in_paths:
        out_path = outdir_path / f'{fa_path.stem}.pfam31_results_raw.msgpack'
        input_queue.put((fa_path, out_path))
    input_queue.put('STOP')

    # join
    try:
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
