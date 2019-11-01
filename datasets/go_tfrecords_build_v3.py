import os
import re
import time
import gzip
import json
import errno
import struct
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from pathlib import Path
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict

# GAF2.1 files have the suffix .gaf and contain the following columns:
# 	Column  Contents
# 	1       DB
# 	2       DB_Object_ID = UniqueIdentifier
# 	3       DB_Object_Symbol = GN=GeneName
# 	4       Qualifier
# 	5       GO_ID
# 	6       DB:Reference
# 	7       Evidence Code
# 	8       With (or) From
# 	9       Aspect
# 	10      DB_Object_Name
# 	11      DB_Object_Synonym
# 	12      DB_Object_Type
# 	13      Taxon and Interacting taxon
# 	14      Date
# 	15      Assigned_By
# 	16      Annotation_Extension
# 	17      Gene_Product_Form_ID


def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def index_uniprot_fa(path, id_seq, whitelist):
    # >db|UniqueIdentifier|EntryName ProteinName OS=OrganismName OX=OrganismIdentifier [GN=GeneName ]PE=ProteinExistence SV=SequenceVersion
    # db is 'sp' for UniProtKB/Swiss-Prot and 'tr' for UniProtKB/TrEMBL.
    # UniqueIdentifier is the primary accession number of the UniProtKB entry.
    # EntryName is the entry name of the UniProtKB entry.
    # ProteinName is the recommended name of the UniProtKB entry as annotated in the RecName field. For UniProtKB/TrEMBL entries without a RecName field, the SubName field is used. In case of multiple SubNames, the first one is used. The 'precursor' attribute is excluded, 'Fragment' is included with the name if applicable.
    # OrganismName is the scientific name of the organism of the UniProtKB entry.
    # OrganismIdentifier is the unique identifier of the source organism, assigned by the NCBI.
    # GeneName is the first gene name of the UniProtKB entry. If there is no gene name, OrderedLocusName or ORFname, the GN field is not listed.
    # ProteinExistence is the numerical value describing the evidence for the existence of the protein.
    # SequenceVersion is the version number of the sequence.
    path = Path(path)
    # if gzipped
    target = os.path.getsize(path)
    if path.suffix == '.gz':
        f = gzip.open(path, mode='rt', encoding='utf-8')
        target = _gzip_size(path)
        while target < os.path.getsize(path):
            # the uncompressed size can't be smaller than the compressed size, so add 4GB
            target += 2**32
    else:
        f = path.open(mode='r', encoding='utf-8')
    # initialize
    sequence = ''
    with f, tqdm(
        total=target,
        unit='bytes',
        dynamic_ncols=True,
        ascii=True,
        desc=path.name
    ) as t:
        for line in f:
            t.update(len(line))
            # empty line
            line = line.strip()
            if len(line) > 0 and line[0] == '>':
                if sequence:
                    if uid in whitelist:
                        id_seq[uid] = sequence
                    sequence = ''
                # parse header
                seq_id = line.split()[0][1:]
                tokens = seq_id.split('|')
                # DB_Object_ID = UniqueIdentifier
                uid = tokens[1].strip()
            else:
                sequence += line
        if sequence:
            if uid in whitelist:
                id_seq[uid] = sequence
            sequence = ''
    return


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build GO training dataset.')
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.register('type', 'list', lambda v: ast.literal_eval(v))
    parser.add_argument(
        '-1',
        '--gaf1',
        type=str,
        required=True,
        help="Path to the GO gaf file, required."
    )
    parser.add_argument(
        '-2',
        '--fa2',
        type=str,
        required=True,
        help="Path to the uniprot trembl fasta file, required."
    )
    parser.add_argument(
        '-3',
        '--fa3',
        type=str,
        required=True,
        help="Path to the uniprot sprot fasta file, required."
    )
    parser.add_argument(
        '-4',
        '--fa4',
        type=str,
        required=True,
        help="Path to the uniprot sprot_varsplic fasta file, required."
    )
    parser.add_argument(
        '-m',
        '--meta',
        type=str,
        required=True,
        help="Path to output meta.json, required."
    )
    parser.add_argument(
        '-t',
        '--train',
        type=str,
        required=True,
        help="Path to output train.tfrecords, required."
    )
    parser.add_argument(
        '--label_type',
        type=str,
        choices=['multilabel', 'seq2seq'],
        default='multilabel',
        help='Type of label to output'
    )
    parser.add_argument(
        '--ignore_iea',
        type='bool',
        default='True',
        help='Do not include annotations with evidence type of IEA (Inferred from Electronic Annotation).'
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    gaf1_path = verify_input_path(args.gaf1)
    fa2_path = verify_input_path(args.fa2)
    fa3_path = verify_input_path(args.fa3)
    fa4_path = verify_input_path(args.fa4)
    meta_path = verify_output_path(args.meta)
    train_path = verify_output_path(args.train)
    print(
        f'Building from {gaf1_path.name}, {fa2_path.name}, {fa3_path.name} and {fa4_path.name}'
    )

    path = gaf1_path
    # if gzipped
    target = os.path.getsize(path)
    if path.suffix == '.gz':
        f = gzip.open(path, mode='rt', encoding='utf-8')
        target = _gzip_size(path)
        while target < os.path.getsize(path):
            # the uncompressed size can't be smaller than the compressed size, so add 4GB
            target += 2**32
    else:
        f = path.open(mode='r', encoding='utf-8')
    # initialize
    line_num = 0
    annotation_count = 0
    seq_gos = defaultdict(set)
    seq_count = defaultdict(int)
    missing_max = 10
    with f, tqdm(
        total=target,
        unit='bytes',
        dynamic_ncols=True,
        ascii=True,
        desc=path.name
    ) as t:
        for line in f:
            t.update(len(line))
            line_num += 1
            # empty line
            line = line.strip()
            if line == '':
                continue
            # comment
            if line[0] == '!':
                continue
            # line code
            tokens = line.split('\t')
            seq_type = tokens[11].strip()
            if seq_type != 'protein':
                continue
            evidence = tokens[6].strip()
            if args.ignore_iea and evidence == 'IEA':
                continue
            qualifier = tokens[3].strip()
            if qualifier[:3] == 'NOT':
                continue
            seq_id = tokens[1].strip().split(':')[0]
            if seq_id[-2:] == '-1':
                seq_id = seq_id[:-2]
            annotation_count += 1
            go_id = tokens[4].strip()
            seq_gos[seq_id].add(go_id)
            seq_count[go_id] += 1
    print(f'Annotations: {annotation_count}')
    print(f'Seqs with GOs: {len(seq_gos)}')
    print(f'GOs: {len(seq_count)}')
    index_from = 1
    aa_list = 'FLIMVPAWGSTYQNCO*UHKRDEBZX-'
    aa_index = dict(zip(aa_list, range(index_from, index_from + len(aa_list))))
    if args.label_type == 'multilabel':
        go_list = sorted(go_count, key=lambda k: (go_count[k], k), reverse=True)
    else:
        go_list = [
            'PAD', 'START', 'END'
        ] + sorted(seq_count, key=lambda k: (seq_count[k], k), reverse=True)
    go_index = dict([(d, i) for i, d in enumerate(go_list)])
    go_total = len(go_list)
    seq_list = sorted(seq_gos, key=lambda k: len(seq_gos[k]), reverse=True)

    # read sequences
    id_seq = {}
    index_uniprot_fa(fa2_path, id_seq, seq_gos)
    index_uniprot_fa(fa3_path, id_seq, seq_gos)
    index_uniprot_fa(fa4_path, id_seq, seq_gos)
    print(f'Sequences: {len(id_seq)}')

    # train list
    seed = 113
    np.random.seed(seed)
    train_list = list(id_seq)
    np.random.shuffle(train_list)
    train_count = len(train_list)

    # output meta
    seq_lens = sorted([len(seq) for seq in id_seq.values()])
    metadata = {
        'train':
            {
                'seq_count':
                    {
                        'total': train_count,
                        'per_class':
                            {
                                'min': seq_count[go_list[-1]],
                                'median': seq_count[go_list[len(go_list) // 2]],
                                'max': seq_count[go_list[0]]
                            }
                    },
                'seq_len':
                    {
                        'total': sum(seq_lens),
                        'min': seq_lens[0],
                        'median': seq_lens[len(seq_lens) // 2],
                        'max': seq_lens[-1]
                    },
                'go_count':
                    {
                        'total': go_total,
                        'per_class':
                            {
                                'min':
                                    len(seq_gos[seq_list[-1]]),
                                'median':
                                    len(seq_gos[seq_list[len(seq_list) // 2]]),
                                'max':
                                    len(seq_gos[seq_list[0]])
                            }
                    }
            },
        'test':
            {
                'seq_count':
                    {
                        'total': 0,
                        'per_class': {
                            'min': 0,
                            'median': 0,
                            'max': 0
                        }
                    },
                'seq_len': {
                    'total': 0,
                    'min': 0,
                    'median': 0,
                    'max': 0
                }
            },
        'aa_list': aa_list,
        'aa_index': aa_index,
        'go_list': go_list,
        'go_index': go_index
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=False))
    del seq_lens

    # output tfrecords
    with tf.python_io.TFRecordWriter(str(train_path)) as writer, tqdm(
        total=train_count,
        unit='seqs',
        dynamic_ncols=True,
        ascii=True,
        desc=train_path.name
    ) as t:
        for seq_id in train_list:
            t.update(1)
            protein = np.array(
                [aa_index[a] for a in id_seq[seq_id]], dtype=np.uint8
            )
            if args.label_type == 'multilabel':
                go = np.zeros(go_total, dtype=np.uint8)
                for go_id in seq_gos[seq_id]:
                    go[go_index[go_id]] = 1
            else:
                go = np.array(
                    [1] + sorted([go_index[go_id]
                                for go_id in seq_gos[seq_id]]) + [2],
                    dtype=np.uint16
                )
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'protein':
                            tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[protein.tobytes()]
                                )
                            ),
                        'go':
                            tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[go.tobytes()]
                                )
                            )
                    }
                )
            )
            writer.write(example.SerializeToString())
            del protein
            del go
            del example

    # end
    print(f'Runtime: {time.time() - start_time:.2f} s\n')

# W2125

# python datasets/go_tfrecords_build_v3.py --gaf1 /data12/goa/goa-20191015/goa_uniprot_all.gaf --fa2 /data12/uniprot/uniprot-20191015/uniprot_trembl.fasta --fa3 /data12/uniprot/uniprot-20191015/uniprot_sprot.fasta --fa4 /data12/uniprot/uniprot-20191015/uniprot_sprot_varsplic.fasta --meta /data12/goa/goa-20191015/goa-20191015-v3-iea-multilabel-meta.json --train /data12/goa/goa-20191015/goa-20191015-v3-iea-multilabel-train.tfrecords --label_type multilabel --ignore_iea False

# python datasets/go_tfrecords_build_v3.py --gaf1 /data12/goa/goa-20191015/goa_uniprot_all.gaf --fa2 /data12/uniprot/uniprot-20191015/uniprot_trembl.fasta --fa3 /data12/uniprot/uniprot-20191015/uniprot_sprot.fasta --fa4 /data12/uniprot/uniprot-20191015/uniprot_sprot_varsplic.fasta --meta /data12/goa/goa-20191015/goa-20191015-v3-iea-seq2seq-meta.json --train /data12/goa/goa-20191015/goa-20191015-v3-iea-seq2seq-train.tfrecords --label_type seq2seq --ignore_iea False

# v1,v2,v3 filters:
# seq_type == 'protein' && evidence != 'IEA' && qualifier[:3] != 'NOT'
# Annotations: 4197534
# Seqs with GOs: 794688
# GOs: 27702
# Sequences: 790366
# Runtime: 1772.25 s
