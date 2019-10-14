import os
import re
import time
import gzip
import errno
import struct
import argparse
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


def stats_fa(path):
    # >db|UniqueIdentifier|EntryName ProteinName OS=OrganismName OX=OrganismIdentifier [GN=GeneName ]PE=ProteinExistence SV=SequenceVersion
    # db is ‘sp’ for UniProtKB/Swiss-Prot and ‘tr’ for UniProtKB/TrEMBL.
    # UniqueIdentifier is the primary accession number of the UniProtKB entry.
    # EntryName is the entry name of the UniProtKB entry.
    # ProteinName is the recommended name of the UniProtKB entry as annotated in the RecName field. For UniProtKB/TrEMBL entries without a RecName field, the SubName field is used. In case of multiple SubNames, the first one is used. The ‘precursor’ attribute is excluded, ‘Fragment’ is included with the name if applicable.
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
    vocab = defaultdict(set)
    line_num = 0
    pattern = re.compile(r'^.+?GN=([^= ]+)')
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
            if len(line) > 0 and line[0] == '>':
                # parse header
                seq_id = line.split()[0][1:]
                tokens = seq_id.split('|')
                # DB_Object_ID = UniqueIdentifier
                vocab['seqid'].add(tokens[1].strip())
                # DB_Object_Symbol = GN=GeneName
                match = pattern.match(line)
                if match:
                    vocab['symbol'].add(match.group(1))
    return vocab, line_num


def stats_go_gaf(path):
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
    vocab = defaultdict(set)
    line_num = 0
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
            vocab['seqid'].add(tokens[1].strip())
            vocab['symbol'].add(tokens[2].strip())
    return vocab, line_num


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


def print_set_stats(n1, s1, n2, s2, unit=''):
    print(
        f'''
{n1}: {len(s1)} {unit}
{n2}: {len(s2)} {unit}
{n1} & {n2}: {len(s1 & s2)} {unit}
{n1} | {n2}: {len(s1 | s2)} {unit}
{n1} - {n2}: {len(s1 - s2)} {unit}
{n2} - {n1}: {len(s2 - s1)} {unit}
'''
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Print the differences of two GO gaf and two uniprot fasta files.'
    )
    parser.add_argument(
        '-1',
        '--gaf1',
        type=str,
        required=True,
        help="Path to the first GO gaf file, required."
    )
    parser.add_argument(
        '-2',
        '--gaf2',
        type=str,
        required=True,
        help="Path to the second GO gaf file, required."
    )
    parser.add_argument(
        '-3',
        '--fa3',
        type=str,
        required=True,
        help="Path to the first uniprot fasta file, required."
    )
    parser.add_argument(
        '-4',
        '--fa4',
        type=str,
        required=True,
        help="Path to the second uniprot fasta file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    gaf1_path = verify_input_path(args.gaf1)
    gaf2_path = verify_input_path(args.gaf2)
    fa3_path = verify_input_path(args.fa3)
    fa4_path = verify_input_path(args.fa4)
    print(
        f'Comparing: {gaf1_path.name}, {gaf2_path.name}, {fa3_path.name} and {fa4_path.name}'
    )

    vocab1, line_num1 = stats_go_gaf(gaf1_path)
    print(f'Processed {line_num1} lines in {gaf1_path.name}')
    vocab2, line_num2 = stats_go_gaf(gaf2_path)
    print(f'Processed {line_num2} lines in {gaf2_path.name}')
    vocab3, line_num3 = stats_fa(fa3_path)
    print(f'Processed {line_num3} lines in {fa3_path.name}')
    vocab4, line_num4 = stats_fa(fa4_path)
    print(f'Processed {line_num4} lines in {fa4_path.name}')

    for sid in vocab1.keys():
        print(f'== {sid} set stats ==')
        print_set_stats(
            gaf1_path.name, vocab1[sid], gaf2_path.name, vocab2[sid]
        )
        print(f'== {sid} set stats ==')
        print_set_stats(fa3_path.name, vocab3[sid], fa4_path.name, vocab4[sid])
        print(f'== {sid} set stats ==')
        print_set_stats(gaf1_path.name, vocab1[sid], fa3_path.name, vocab3[sid])
        print(f'== {sid} set stats ==')
        print_set_stats(gaf1_path.name, vocab1[sid], fa4_path.name, vocab4[sid])
        print(f'== {sid} set stats ==')
        print_set_stats(gaf2_path.name, vocab2[sid], fa3_path.name, vocab3[sid])
        print(f'== {sid} set stats ==')
        print_set_stats(gaf2_path.name, vocab2[sid], fa4_path.name, vocab4[sid])

    print(f'Runtime: {time.time() - start_time:.2f} s\n')

# windows
# python .\util\go\compare_go_gaf_uniprot_fa.py --gaf1 D:/go/goa/UNIPROT/goa_uniprot_all.gaf --gaf2 D:/go/goa/UNIPROT/goa_uniprot_gcrp.gaf
# W2125
# python util/go/compare_go_gaf_uniprot_fa.py --gaf1 /data12/goa/goa_uniprot_all.gaf --gaf2 /data12/goa/goa_uniprot_gcrp.gaf --fa3 /data12/uniprot/uniprot-20190211/uniprot_trembl.fasta --fa4 /data12/uniprot/uniprot-20190211/uniprot_sprot.fasta

# goa_uniprot_all.gaf includes goa_uniprot_gcrp.gaf
# uniprot_trembl.fasta does not include uniprot_sprot.fasta

# Processed 608472856 lines in goa_uniprot_all.gaf
# Processed 137894383 lines in goa_uniprot_gcrp.gaf
# Processed 989314744 lines in uniprot_trembl.fasta
# Processed 4185641 lines in uniprot_sprot.fasta

# == seqid set stats ==

# goa_uniprot_all.gaf: 102989625
# goa_uniprot_gcrp.gaf: 23582588
# goa_uniprot_all.gaf & goa_uniprot_gcrp.gaf: 23582588
# goa_uniprot_all.gaf | goa_uniprot_gcrp.gaf: 102989625
# goa_uniprot_all.gaf - goa_uniprot_gcrp.gaf: 79407037
# goa_uniprot_gcrp.gaf - goa_uniprot_all.gaf: 0

# == seqid set stats ==

# uniprot_trembl.fasta: 139694261
# uniprot_sprot.fasta: 559077
# uniprot_trembl.fasta & uniprot_sprot.fasta: 0
# uniprot_trembl.fasta | uniprot_sprot.fasta: 140253338
# uniprot_trembl.fasta - uniprot_sprot.fasta: 139694261
# uniprot_sprot.fasta - uniprot_trembl.fasta: 559077

# == seqid set stats ==

# goa_uniprot_all.gaf: 102989625
# uniprot_trembl.fasta: 139694261
# goa_uniprot_all.gaf & uniprot_trembl.fasta: 95753442
# goa_uniprot_all.gaf | uniprot_trembl.fasta: 146930444
# goa_uniprot_all.gaf - uniprot_trembl.fasta: 7236183
# uniprot_trembl.fasta - goa_uniprot_all.gaf: 43940819

# == seqid set stats ==

# goa_uniprot_all.gaf: 102989625
# uniprot_sprot.fasta: 559077
# goa_uniprot_all.gaf & uniprot_sprot.fasta: 535884
# goa_uniprot_all.gaf | uniprot_sprot.fasta: 103012818
# goa_uniprot_all.gaf - uniprot_sprot.fasta: 102453741
# uniprot_sprot.fasta - goa_uniprot_all.gaf: 23193

# == seqid set stats ==

# goa_uniprot_gcrp.gaf: 23582588
# uniprot_trembl.fasta: 139694261
# goa_uniprot_gcrp.gaf & uniprot_trembl.fasta: 23287728
# goa_uniprot_gcrp.gaf | uniprot_trembl.fasta: 139989121
# goa_uniprot_gcrp.gaf - uniprot_trembl.fasta: 294860
# uniprot_trembl.fasta - goa_uniprot_gcrp.gaf: 116406533

# == seqid set stats ==

# goa_uniprot_gcrp.gaf: 23582588
# uniprot_sprot.fasta: 559077
# goa_uniprot_gcrp.gaf & uniprot_sprot.fasta: 294860
# goa_uniprot_gcrp.gaf | uniprot_sprot.fasta: 23846805
# goa_uniprot_gcrp.gaf - uniprot_sprot.fasta: 23287728
# uniprot_sprot.fasta - goa_uniprot_gcrp.gaf: 264217

# == symbol set stats ==

# goa_uniprot_all.gaf: 75301818
# goa_uniprot_gcrp.gaf: 18832370
# goa_uniprot_all.gaf & goa_uniprot_gcrp.gaf: 18832370
# goa_uniprot_all.gaf | goa_uniprot_gcrp.gaf: 75301818
# goa_uniprot_all.gaf - goa_uniprot_gcrp.gaf: 56469448
# goa_uniprot_gcrp.gaf - goa_uniprot_all.gaf: 0

# == symbol set stats ==

# uniprot_trembl.fasta: 103171005
# uniprot_sprot.fasta: 143232
# uniprot_trembl.fasta & uniprot_sprot.fasta: 89585
# uniprot_trembl.fasta | uniprot_sprot.fasta: 103224652
# uniprot_trembl.fasta - uniprot_sprot.fasta: 103081420
# uniprot_sprot.fasta - uniprot_trembl.fasta: 53647

# == symbol set stats ==

# goa_uniprot_all.gaf: 75301818
# uniprot_trembl.fasta: 103171005
# goa_uniprot_all.gaf & uniprot_trembl.fasta: 64177789
# goa_uniprot_all.gaf | uniprot_trembl.fasta: 114295034
# goa_uniprot_all.gaf - uniprot_trembl.fasta: 11124029
# uniprot_trembl.fasta - goa_uniprot_all.gaf: 38993216

# == symbol set stats ==

# goa_uniprot_all.gaf: 75301818
# uniprot_sprot.fasta: 143232
# goa_uniprot_all.gaf & uniprot_sprot.fasta: 127508
# goa_uniprot_all.gaf | uniprot_sprot.fasta: 75317542
# goa_uniprot_all.gaf - uniprot_sprot.fasta: 75174310
# uniprot_sprot.fasta - goa_uniprot_all.gaf: 15724

# == symbol set stats ==

# goa_uniprot_gcrp.gaf: 18832370
# uniprot_trembl.fasta: 103171005
# goa_uniprot_gcrp.gaf & uniprot_trembl.fasta: 17836046
# goa_uniprot_gcrp.gaf | uniprot_trembl.fasta: 104167329
# goa_uniprot_gcrp.gaf - uniprot_trembl.fasta: 996324
# uniprot_trembl.fasta - goa_uniprot_gcrp.gaf: 85334959

# == symbol set stats ==

# goa_uniprot_gcrp.gaf: 18832370
# uniprot_sprot.fasta: 143232
# goa_uniprot_gcrp.gaf & uniprot_sprot.fasta: 111649
# goa_uniprot_gcrp.gaf | uniprot_sprot.fasta: 18863953
# goa_uniprot_gcrp.gaf - uniprot_sprot.fasta: 18720721
# uniprot_sprot.fasta - goa_uniprot_gcrp.gaf: 31583

# Runtime: 2376.93 s
