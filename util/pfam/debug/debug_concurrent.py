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


def serving_predict_fasta(i):
    time.sleep(0.2)
    logging.info(f'inside worker {i}')
    return i


# windows
# python .\util\pfam\test_concurrent.py --workers 1

if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description=
        'Run ANNotate prediction locally on each FASTA file in the input directory.'
    )
    parser.add_argument(
        '-w',
        '--workers',
        type=int,
        default=1,
        help="Number of concurrent processes to run. (default: %(default)s)"
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    # ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as e:
        future_path = {e.submit(serving_predict_fasta, i): i for i in range(10)}
        for future in concurrent.futures.as_completed(future_path):
            time.sleep(0.2)
            i = future_path[future]
            try:
                result = future.result()
            except Exception as exc:
                logging.info(f"{i} generated an exception: {str(exc)}")
            else:
                logging.info(f"Completed processing {i}")

    logging.info(f'Runtime: {time.time() - start_time:.2f} s')
    sys.exit(0)
