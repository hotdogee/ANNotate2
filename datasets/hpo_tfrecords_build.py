import os
import re
import ast
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


def index_uniprot_fa(path, acc_seq, whitelist):
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
    # initialize
    sequence = ''
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
    with f, tqdm(
        total=target,
        dynamic_ncols=True,
        ascii=True,
        smoothing=0.01,
        desc=path.name,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as t:
        for line in f:
            t.update(len(line))
            if t.n > t.total:
                t.total += 2**32
            # empty line
            line = line.strip()
            if len(line) > 0 and line[0] == '>':
                if sequence:
                    if acc in whitelist:
                        acc_seq[acc] = sequence
                    sequence = ''
                # parse header
                seq_id = line.split()[0][1:]
                tokens = seq_id.split('|')
                # DB_Object_ID = UniqueIdentifier
                acc = tokens[1].strip()
            else:
                sequence += line
        if sequence:
            if acc in whitelist:
                acc_seq[acc] = sequence
            sequence = ''
    return


def parse_obo(path):
    # initialize
    line_num = 0
    header = True
    current_term = ''
    # return
    term_is_a = defaultdict(set)
    term_info = defaultdict(dict)
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
    with f, tqdm(
        total=target,
        dynamic_ncols=True,
        ascii=True,
        smoothing=0.01,
        desc=path.name,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as t:
        for line in f:
            t.update(len(line))
            if t.n > t.total:
                t.total += 2**32
            line_num += 1
            # empty line
            line = line.strip()
            if not line:
                continue
            # check out of header section
            if line == '[Term]':
                if header:
                    header = False
                continue
            if header:
                continue
            tokens = line.split(': ')
            key = tokens[0].strip()
            value = tokens[1].strip()
            if key == 'id':
                current_term = value
                term_is_a[current_term]
            else:
                term_info[current_term][key] = value
                if key == 'is_a':
                    term_is_a[current_term].add(value.split(' ! ')[0].strip())
                elif key == 'replaced_by':
                    term_is_a[value.split()[0].strip()].add(current_term)

    def walk(p_set, p_list):
        if len(p_list) == 0:
            return
        else:
            for p in p_list:
                if p not in p_set:
                    p_set.add(p)
                    walk(p_set, term_is_a[p])

    term_parents = defaultdict(set)
    for t in term_is_a:
        p_set = term_parents[t]
        walk(p_set, term_is_a[t])
    return term_is_a, term_info, term_parents


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
    parser = argparse.ArgumentParser(description='Build HPO training dataset.')
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.register('type', 'list', lambda v: ast.literal_eval(v))
    parser.add_argument(
        '-g',
        '--gene_acc_phenotype',
        type=str,
        required=True,
        help="Path to gene_acc_phenotype.json file, required."
    )
    parser.add_argument(
        '-svf',
        '--sprot_varsplic_fa',
        type=str,
        required=True,
        help="Path to uniprot_sprot_varsplic.fasta[.gz] file, required."
    )
    parser.add_argument(
        '-sf',
        '--sprot_fa',
        type=str,
        required=True,
        help="Path to uniprot_sprot.fasta[.gz] file, required."
    )
    parser.add_argument(
        '-tf',
        '--trembl_fa',
        type=str,
        required=True,
        help="Path to uniprot_trembl.fasta[.gz] file, required."
    )
    parser.add_argument(
        '-ob',
        '--obo',
        type=str,
        required=True,
        help="Path to hp.obo[.gz] file, required."
    )
    parser.add_argument(
        '-o',
        '--out_prefix',
        type=str,
        required=True,
        help="Path prefix of output files, required."
    )
    parser.add_argument(
        '--label_type',
        type=str,
        choices=['multilabel', 'seq2seq'],
        default='multilabel',
        help='Type of label to output'
    )
    parser.add_argument(
        '--train_split',
        type=int,
        default=5,
        help='Split training dataset into multiple tfrecords.'
    )
    parser.add_argument(
        '--max_protein_length',
        type=int,
        default=2**14, # 16384
        help='Split training dataset into multiple tfrecords.'
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    gene_acc_phenotype_path = verify_input_path(args.gene_acc_phenotype)
    sprot_varsplic_fa_path = verify_input_path(args.sprot_varsplic_fa)
    sprot_fa_path = verify_input_path(args.sprot_fa)
    trembl_fa_path = verify_input_path(args.trembl_fa)
    obo_path = verify_input_path(args.obo)
    print(
        f'Building from {gene_acc_phenotype_path.name}, {sprot_varsplic_fa_path.name}, {sprot_fa_path.name}, {trembl_fa_path.name} and {obo_path.name}'
    )

    print(f'load gene_acc_phenotype')
    with gene_acc_phenotype_path.open(mode='r', encoding='utf-8') as f:
        gene_acc_phenotype = json.load(f)

    # train_gene_acc = gene_acc_phenotype['train_gene_acc']
    train_phenotype_acc = gene_acc_phenotype['train_phenotype_acc']
    # test_gene_acc = gene_acc_phenotype['test_gene_acc']
    test_phenotype_acc = gene_acc_phenotype['test_phenotype_acc']

    # build acc_phenotype
    print(f'build acc_phenotype')
    train_acc_phenotype = defaultdict(set)
    for p in train_phenotype_acc:
        for a in train_phenotype_acc[p]:
            train_acc_phenotype[a].add(p)
    print(f'=== TRAIN ===')
    print(f'Annotations: {sum([len(train_phenotype_acc[p]) for p in train_phenotype_acc])}')
    print(f'Seqs with HPOs: {len(train_acc_phenotype)}')
    print(f'HPOs: {len(train_phenotype_acc)}')

    test_acc_phenotype = defaultdict(set)
    for p in test_phenotype_acc:
        for a in test_phenotype_acc[p]:
            test_acc_phenotype[a].add(p)
    print(f'=== TEST ===')
    print(f'Annotations: {sum([len(test_phenotype_acc[p]) for p in test_phenotype_acc])}')
    print(f'Seqs with HPOs: {len(test_acc_phenotype)}')
    print(f'HPOs: {len(test_phenotype_acc)}')

    # remove parent phenotypes
    print(f'remove parent phenotypes')
    term_is_a, term_info, term_parents = parse_obo(obo_path)
    for a in train_acc_phenotype:
        p_set = set()
        for p in train_acc_phenotype[a]:
            p_set |= term_parents[p]
        train_acc_phenotype[a] -= p_set
        if len(train_acc_phenotype[a]) == 0:
            print(f'===DEBUG: {a} has no phenotypes')
    train_phenotype_acc = defaultdict(set)
    for a in train_acc_phenotype:
        for p in train_acc_phenotype[a]:
            train_phenotype_acc[p].add(a)
    print(f'=== TRAIN COMPACTED ===')
    print(f'Annotations: {sum([len(train_phenotype_acc[p]) for p in train_phenotype_acc])}')
    print(f'Seqs with HPOs: {len(train_acc_phenotype)}')
    print(f'HPOs: {len(train_phenotype_acc)}')

    for a in test_acc_phenotype:
        p_set = set()
        for p in test_acc_phenotype[a]:
            p_set |= term_parents[p]
        test_acc_phenotype[a] -= p_set
        if len(test_acc_phenotype[a]) == 0:
            print(f'===DEBUG: {a} has no phenotypes')
    test_phenotype_acc = defaultdict(set)
    for a in test_acc_phenotype:
        for p in test_acc_phenotype[a]:
            test_phenotype_acc[p].add(a)
    print(f'=== TEST COMPACTED ===')
    print(f'Annotations: {sum([len(test_phenotype_acc[p]) for p in test_phenotype_acc])}')
    print(f'Seqs with HPOs: {len(test_acc_phenotype)}')
    print(f'HPOs: {len(test_phenotype_acc)}')

    # build phenotype_list and acc_list
    print(f'build phenotype_list and acc_list')
    train_phenotype_list = sorted(
        train_phenotype_acc, key=lambda k: (len(train_phenotype_acc[k]), k)
    )
    test_phenotype_list = sorted(
        test_phenotype_acc, key=lambda k: (len(test_phenotype_acc[k]), k)
    )
    train_acc_list = sorted(
        train_acc_phenotype, key=lambda k: (len(train_acc_phenotype[k]), k)
    )
    test_acc_list = sorted(
        test_acc_phenotype, key=lambda k: (len(test_acc_phenotype[k]), k)
    )

    # split training
    print(f'split training')
    acc_lists = [[] for i in range(args.train_split)]
    acc_set = set()
    for p in train_phenotype_list:
        for a in train_phenotype_acc[p]:
            if a not in acc_set:
                acc_lists[len(acc_set)%args.train_split].append(a)
                acc_set.add(a)

    # shuffle training
    print(f'shuffle training')
    seed = 113
    np.random.seed(seed)
    for al in acc_lists:
        np.random.shuffle(al)

    # read sequences
    acc_set = set(train_acc_list) | set(test_acc_list)
    acc_seq = {}
    index_uniprot_fa(sprot_varsplic_fa_path, acc_seq, acc_set)
    index_uniprot_fa(sprot_fa_path, acc_seq, acc_set)
    index_uniprot_fa(trembl_fa_path, acc_seq, acc_set)
    print(f'Sequences: {len(acc_seq)}')

    # truncate long sequences
    print(f'truncate long sequences to {args.max_protein_length}')
    for a in acc_seq:
        if len(acc_seq[a]) > args.max_protein_length:
            acc_seq[a] = acc_seq[a][:args.max_protein_length]

    # build indices
    print(f'build indices')
    index_from = 1
    aa_list = 'FLIMVPAWGSTYQNCO*UHKRDEBZX-'
    aa_index = dict(zip(aa_list, range(index_from, index_from + len(aa_list))))
    hpo_set = set(train_phenotype_acc) | set(test_phenotype_acc)
    hpo_list = sorted(hpo_set, key=lambda k: (len(train_phenotype_acc[k]) + len(test_phenotype_acc.get(k, [])), k), reverse=True)
    if args.label_type == 'seq2seq':
        hpo_list = ['PAD', 'START', 'END'] + hpo_list
    hpo_index = dict([(d, i) for i, d in enumerate(hpo_list)])
    hpo_total = len(hpo_list)
    print(f'HPO total: {hpo_total}')

    # write metadata
    print(f'write metadata')
    train_seq_lens = sorted([len(acc_seq[acc]) for acc in train_acc_list])
    test_seq_lens = sorted([len(acc_seq[acc]) for acc in test_acc_list])
    label_description = {
        'seq2seq': 'sequence of HPO ids wrapped between <START> and <END> labels',
        'multilabel': f'int array of length <hpo_total>, a value of 1 at index <id> indicates that HPO is included'
    }
    metadata = {
        'model': args.label_type,
        'feature_desc': {
            'protein': 'sequence of amino acid ids'
        },
        'label_desc': {
            'hpo': label_description[args.label_type]
        },
        'train':
            {
                'accessions': len(train_acc_phenotype), # Accessions: 1118642
                'accessions_per_phenotype': # Accessions per Phenotype: (Min, Median, Max): 1, 1173, 741799
                    {
                        'min':
                            len(train_phenotype_acc[train_phenotype_list[0]]),
                        'median':
                            len(train_phenotype_acc[train_phenotype_list[len(train_phenotype_list) // 2]]),
                        'max':
                            len(train_phenotype_acc[train_phenotype_list[-1]])
                    },
                'phenotypes': len(train_phenotype_acc), # Phenotypes: 8017, doesn't include ['PAD', 'START', 'END']
                'phenotypes_per_accession': # Phenotypes per Accession: (Min, Median, Max): 1, 29, 457
                    {
                        'min':
                            len(train_acc_phenotype[train_acc_list[0]]),
                        'median':
                            len(train_acc_phenotype[train_acc_list[len(train_acc_list) // 2]]),
                        'max':
                            len(train_acc_phenotype[train_acc_list[-1]])
                    },
                'sequence_lengths':
                    {
                        'total': sum(train_seq_lens),
                        'min': train_seq_lens[0],
                        'median': train_seq_lens[len(train_seq_lens) // 2],
                        'max': train_seq_lens[-1]
                    }
            },
        'test':
            {
                'accessions': len(test_acc_phenotype), # Accessions: 1118642
                'accessions_per_phenotype': # Accessions per Phenotype: (Min, Median, Max): 1, 1173, 741799
                    {
                        'min':
                            len(test_phenotype_acc[test_phenotype_list[0]]),
                        'median':
                            len(test_phenotype_acc[test_phenotype_list[len(test_phenotype_list) // 2]]),
                        'max':
                            len(test_phenotype_acc[test_phenotype_list[-1]])
                    },
                'phenotypes': len(test_phenotype_acc), # Phenotypes: 3617, doesn't include ['PAD', 'START', 'END']
                'phenotypes_per_accession': # Phenotypes per Accession: (Min, Median, Max): 1, 29, 457
                    {
                        'min':
                            len(test_acc_phenotype[test_acc_list[0]]),
                        'median':
                            len(test_acc_phenotype[test_acc_list[len(test_acc_list) // 2]]),
                        'max':
                            len(test_acc_phenotype[test_acc_list[-1]])
                    },
                'sequence_lengths':
                    {
                        'total': sum(test_seq_lens),
                        'min': test_seq_lens[0],
                        'median': test_seq_lens[len(test_seq_lens) // 2],
                        'max': test_seq_lens[-1]
                    }
            },
        'aa_list': aa_list,
        'aa_index': aa_index,
        'hpo_list': hpo_list,
        'hpo_index': hpo_index
    }

    meta_path = verify_output_path(args.out_prefix + '-meta.json')
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=False))

    # write tfrecords
    count_fmt = "{0:0" + str(len(str(args.train_split))) + "d}"
    datasets = [[str(verify_output_path(args.out_prefix + f'-train.{count_fmt.format(i+1)}of{args.train_split}.tfrecords')), acc_lists[i], train_acc_phenotype] for i in range(args.train_split)] + [[str(verify_output_path(args.out_prefix + '-test.tfrecords')), test_acc_list, test_acc_phenotype]]

    for path_str, acc_list, acc_phenotype in datasets:
        with tf.python_io.TFRecordWriter(path_str) as writer, tqdm(
            total=len(acc_list),
            unit='seqs',
            dynamic_ncols=True,
            ascii=True,
            smoothing=0.01,
            desc=Path(path_str).name
        ) as t:
            for acc in acc_list:
                t.update(1)
                protein_feature = np.array(
                    [aa_index[a] for a in acc_seq[acc]], dtype=np.uint8
                )
                if args.label_type == 'multilabel':
                    hpo_feature = np.zeros(hpo_total, dtype=np.uint8)
                    for hpo in acc_phenotype[acc]:
                        hpo_feature[hpo_index[hpo]] = 1
                else:
                    hpo_feature = np.array(
                        [1] + sorted([hpo_index[hpo]
                                    for hpo in acc_phenotype[acc]]) + [2],
                        dtype=np.uint16
                    )
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'protein':
                                tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[protein_feature.tobytes()]
                                    )
                                ),
                            'hpo':
                                tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[hpo_feature.tobytes()]
                                    )
                                )
                        }
                    )
                )
                writer.write(example.SerializeToString())
                del protein_feature
                del hpo_feature
                del example

    # end
    print(f'Runtime: {time.time() - start_time:.2f} s\n')

# W2125
# python datasets/hpo_tfrecords_build.py --gene_acc_phenotype /data12/hpo/hpo-20191011/hpo-20191011-gene_acc_phenotype.json --sprot_varsplic_fa /data12/uniprot/uniprot-20191109/uniprot_sprot_varsplic.fasta --sprot_fa /data12/uniprot/uniprot-20191109/uniprot_sprot.fasta --trembl_fa /data12/uniprot/uniprot-20191109/uniprot_trembl.fasta --obo /data12/hpo/hpo-20191011/ontology/hp.obo --out_prefix /data12/hpo/hpo-20191011/hpo-20191011-seq2seq --label_type seq2seq --train_split 5 --max_protein_length 16384

# python datasets/hpo_tfrecords_build.py --gene_acc_phenotype /data12/hpo/hpo-20191011/hpo-20191011-gene_acc_phenotype.json --sprot_varsplic_fa /data12/uniprot/uniprot-20191109/uniprot_sprot_varsplic.fasta --sprot_fa /data12/uniprot/uniprot-20191109/uniprot_sprot.fasta --trembl_fa /data12/uniprot/uniprot-20191109/uniprot_trembl.fasta --obo /data12/hpo/hpo-20191011/ontology/hp.obo --out_prefix /data12/hpo/hpo-20191011/hpo-20191011-multilabel --label_type multilabel --train_split 5

# before compacting
# Building from hpo-20191011-gene_acc_phenotype.json, uniprot_sprot_varsplic.fasta, uniprot_sprot.fasta and uniprot_trembl.fasta
# HPO total: 8017
# === TRAIN ===
# Annotations: 47196486
# Seqs with HPOs: 1118642
# HPOs: 8017
# === TEST ===
# Annotations: 113879312
# Seqs with HPOs: 2381707
# HPOs: 3617
# uniprot_sprot_varsplic.fasta: 100%|########################################################################################| 27.5M/27.5M [00:00<00:00, 78.6MB/s]
# uniprot_sprot.fasta: 100%|###################################################################################################| 264M/264M [00:03<00:00, 76.8MB/s]
# uniprot_trembl.fasta: 100%|################################################################################################| 78.4G/78.4G [16:24<00:00, 85.5MB/s]
# Sequences: 3500349
# hpo-20191011-multilabel-train.1of5.tfrecords: 100%|#################################################################| 223729/223729 [00:24<00:00, 9270.68seqs/s]
# hpo-20191011-multilabel-train.2of5.tfrecords: 100%|#################################################################| 223729/223729 [00:23<00:00, 9343.01seqs/s]
# hpo-20191011-multilabel-train.3of5.tfrecords: 100%|#################################################################| 223728/223728 [00:23<00:00, 9450.09seqs/s]
# hpo-20191011-multilabel-train.4of5.tfrecords: 100%|#################################################################| 223728/223728 [00:24<00:00, 9240.52seqs/s]
# hpo-20191011-multilabel-train.5of5.tfrecords: 100%|#################################################################| 223728/223728 [00:23<00:00, 9372.66seqs/s]
# hpo-20191011-multilabel-test.tfrecords: 100%|####################################################################| 2381707/2381707 [03:07<00:00, 12670.65seqs/s]
# Runtime: 1516.87 s
