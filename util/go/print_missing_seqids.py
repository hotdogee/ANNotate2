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
                uid = tokens[1].strip()
                vocab['seqid'].add(uid)
                # if uid in ["A0A000", "A0A001", "A0A002"]:
                #     print(line)
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
        'Check if we can find all the seqids in the GO gaf file in the uniprot fasta file.'
    )
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
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    gaf1_path = verify_input_path(args.gaf1)
    fa2_path = verify_input_path(args.fa2)
    fa3_path = verify_input_path(args.fa3)
    fa4_path = verify_input_path(args.fa4)
    print(
        f'Checking {gaf1_path.name} in {fa2_path.name}, {fa3_path.name} or {fa4_path.name}'
    )

    vocab2, line_num2 = stats_fa(fa2_path)
    print(f'Processed {line_num2} lines in {fa2_path.name}')
    vocab3, line_num3 = stats_fa(fa3_path)
    print(f'Processed {line_num3} lines in {fa3_path.name}')
    vocab4, line_num4 = stats_fa(fa4_path)
    print(f'Processed {line_num4} lines in {fa4_path.name}')

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
    missing_seqids = set()
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
            seqid = tokens[1].strip().split(':')[0]
            if seqid not in vocab2['seqid'] and seqid not in vocab3[
                'seqid'
            ] and seqid not in vocab4['seqid'] and seqid not in missing_seqids:
                missing_seqids.add(seqid)
                if len(missing_seqids) > missing_max:
                    break
                print(
                    f'line {line_num}: "{seqid}" not found in {fa2_path.name}'
                )
                print(line)

    print(f'Runtime: {time.time() - start_time:.2f} s\n')

# windows
# python .\util\go\compare_go_gaf_uniprot_fa.py --gaf1 D:/go/goa/UNIPROT/goa_uniprot_all.gaf --gaf2 D:/go/goa/UNIPROT/goa_uniprot_gcrp.gaf
# W2125
# python util/go/print_missing_seqids.py --gaf1 /data12/goa/goa_uniprot_all.gaf --fa2 /data12/uniprot/uniprot-20190211/uniprot_trembl.fasta
# python util/go/print_missing_seqids.py --gaf1 /data12/goa/goa_uniprot_all.gaf --fa2 /data12/uniprot/uniprot-20190211/uniprot_trembl.fasta --fa3 /data12/uniprot/uniprot-20190211/uniprot_sprot.fasta
# python util/go/print_missing_seqids.py --gaf1 /data12/goa/goa_uniprot_all.gaf --fa2 /data12/uniprot/uniprot-20190211/uniprot_trembl.fasta --fa3 /data12/uniprot/uniprot-20190211/uniprot_sprot.fasta --fa4 /data12/uniprot/uniprot-20191015/uniprot_sprot_varsplic.fasta
