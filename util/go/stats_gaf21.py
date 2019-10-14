import os
import re
import time
import gzip
import errno
import argparse
from glob import glob
from pathlib import Path
from collections import defaultdict

# windows
# python .\util\go\stats_gaf21.py --gaf D:/go/goa/UNIPROT/goa_uniprot_all.gaf.gz
# python .\util\go\stats_gaf21.py --gaf D:/go/goa/UNIPROT/goa_uniprot_gcrp.gaf.gz

# GAF2.1 files have the suffix .gaf and contain the following columns:
# 	Column  Contents
# 	1       DB
# 	2       DB_Object_ID
# 	3       DB_Object_Symbol
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


def stats_gaf(path):
    path = Path(path)
    # if gzipped
    if path.suffix == '.gz':
        f = gzip.open(path, mode='rt', encoding='utf-8')
    else:
        f = path.open(mode='r', encoding='utf-8')
    vocab = defaultdict(set)
    protein_vocab = defaultdict(set)
    no_iea_vocab = defaultdict(set)
    exp_only_vocab = defaultdict(set)
    count = defaultdict(int)
    protein_count = defaultdict(int)
    no_iea_count = defaultdict(int)
    exp_only_count = defaultdict(int)
    with f:
        for line in f:
            # empty line
            line = line.strip()
            if line == '':
                continue
            # comment
            if line[0] == '!':
                continue
            line = line.split('\t')
            vocab['db'].add(line[0].strip())
            vocab['seqid'].add(line[1].strip())
            vocab['symbol'].add(line[2].strip())
            vocab['qual'].add(line[3].strip())
            vocab['go'].add(line[4].strip())
            vocab['evidence'].add(line[6].strip())
            vocab['type'].add(line[11].strip())
            vocab['tax'].add(line[12].strip())
            if line[3].strip() != '':
                count[line[3].strip()] += 1
            if line[11].strip() == 'protein':
                protein_vocab['db'].add(line[0].strip())
                protein_vocab['seqid'].add(line[1].strip())
                protein_vocab['symbol'].add(line[2].strip())
                protein_vocab['qual'].add(line[3].strip())
                protein_vocab['go'].add(line[4].strip())
                protein_vocab['evidence'].add(line[6].strip())
                protein_vocab['type'].add(line[11].strip())
                protein_vocab['tax'].add(line[12].strip())
                if line[3].strip() != '':
                    protein_count[line[3].strip()] += 1
                if line[6].strip() != 'IEA':
                    no_iea_vocab['db'].add(line[0].strip())
                    no_iea_vocab['seqid'].add(line[1].strip())
                    no_iea_vocab['symbol'].add(line[2].strip())
                    no_iea_vocab['qual'].add(line[3].strip())
                    no_iea_vocab['go'].add(line[4].strip())
                    no_iea_vocab['evidence'].add(line[6].strip())
                    no_iea_vocab['type'].add(line[11].strip())
                    no_iea_vocab['tax'].add(line[12].strip())
                    if line[3].strip() != '':
                        no_iea_count[line[3].strip()] += 1
                    if line[6].strip() in [
                        'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP'
                    ]:
                        exp_only_vocab['db'].add(line[0].strip())
                        exp_only_vocab['seqid'].add(line[1].strip())
                        exp_only_vocab['symbol'].add(line[2].strip())
                        exp_only_vocab['qual'].add(line[3].strip())
                        exp_only_vocab['go'].add(line[4].strip())
                        exp_only_vocab['evidence'].add(line[6].strip())
                        exp_only_vocab['type'].add(line[11].strip())
                        exp_only_vocab['tax'].add(line[12].strip())
                        if line[3].strip() != '':
                            exp_only_count[line[3].strip()] += 1
    return count, vocab, protein_count, protein_vocab, no_iea_count, no_iea_vocab, exp_only_count, exp_only_vocab


def print_vocab(vocab):
    print(
        f'''
db: {vocab['db']}
qual: {vocab['qual']}
evidence: {vocab['evidence']}
type: {vocab['type']}
'''
    )
    print(
        f'''
db: {len(vocab['db'])}
seqid: {len(vocab['seqid'])}
symbol: {len(vocab['symbol'])}
qual: {len(vocab['qual'])}
go: {len(vocab['go'])}
evidence: {len(vocab['evidence'])}
type: {len(vocab['type'])}
tax: {len(vocab['tax'])}
'''
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Print some statistics of the GO .gaf.gz file.'
    )
    parser.add_argument(
        '-g',
        '--gaf',
        type=str,
        required=True,
        help="Path to .gaf.gz file, required."
    )

    args, unparsed = parser.parse_known_args()
    # print(f'GAF: {args.gaf}\n')

    start_time = time.time()

    # get absolute path to dataset directory
    gaf_path = Path(os.path.abspath(os.path.expanduser(args.gaf)))
    # print(f'GAF: {gaf_path}\n')

    # doesn't exist
    if not gaf_path.exists():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), gaf_path
        )
    # is dir
    if gaf_path.is_dir():
        raise IsADirectoryError(
            errno.EISDIR, os.strerror(errno.EISDIR), gaf_path
        )
    print(f'Processing: {gaf_path}')

    count, vocab, protein_count, protein_vocab, no_iea_count, no_iea_vocab, exp_only_count, exp_only_vocab = stats_gaf(
        gaf_path
    )
    print('==ALL==')
    print_vocab(vocab)
    print('==QUAL COUNT==')
    for k, v in count.items():
        print(f'{k}: {v}')
    print('==PROTEIN ONLY==')
    print_vocab(protein_vocab)
    print('==QUAL COUNT==')
    for k, v in protein_count.items():
        print(f'{k}: {v}')
    print('==EXCLUDE IEA==')
    print_vocab(no_iea_vocab)
    print('==QUAL COUNT==')
    for k, v in no_iea_count.items():
        print(f'{k}: {v}')
    print('==EXP CODES ONLY==')
    print_vocab(exp_only_vocab)
    print('==QUAL COUNT==')
    for k, v in exp_only_count.items():
        print(f'{k}: {v}')

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# goa_uniprot_all.[gaf|gpa]
# This set contains all GO annotations for proteins in the UniProt KnowledgeBase (UniProtKB) and for entities other than proteins, e.g. macromolecular complexes (Complex Portal identifiers) or RNAs (RNAcentral identifiers).

# goa_uniprot_gcrp.[gaf|gpa]
# This set contains all GO annotations for canonical accessions from the UniProt reference proteomes for all species, which provide one protein per gene. The reference proteomes comprise the protein sequences annotated in Swiss-Prot or the longest TrEMBL transcript if there is no Swiss-Prot record.

# goa_uniprot_gcrp.gaf

# db: {'UniProtKB'}
# qual: {'', 'contributes_to', 'NOT|colocalizes_with', 'NOT', 'NOT|contributes_to', 'colocalizes_with'}
# evidence: {'IKR', 'HDA', 'RCA', 'IBA', 'IPI', 'HMP', 'IEP', 'HGI', 'IDA', 'HEP', 'ISA', 'IC', 'ISO', 'EXP', 'IGC', 'IEA', 'NAS', 'IMP', 'TAS', 'ISM', 'ND', 'ISS', 'IGI'}
# type: {'protein'}

# db: 1
# seqid: 23582588
# symbol: 18832370
# qual: 6
# go: 28694
# evidence: 23
# type: 1
# tax: 12822

# Run time: 518.13 s

# goa_uniprot_all.gaf
# db: 3
# seqid: 102989625
# symbol: 75301818
# qual: 6
# go: 29538
# evidence: 23
# type: 26
# tax: 1334633

# >>> vocab['db']
# {'RNAcentral', 'ComplexPortal', 'UniProtKB'}
# >>> vocab['qual']
# {'', 'contributes_to', 'colocalizes_with', 'NOT|colocalizes_with', 'NOT', 'NOT|contributes_to'}
# >>> vocab['evidence']
# {'EXP', 'ISA', 'HEP', 'TAS', 'ISS', 'HMP', 'ISM', 'IBA', 'IMP', 'IPI', 'ISO', 'IEA', 'IC', 'HDA', 'IEP', 'IDA', 'RCA', 'HGI', 'IKR', 'IGI', 'ND', 'NAS', 'IGC'}
# >>> vocab['type']
# {'snRNA', 'rRNA', 'Y_RNA', 'ncRNA', 'protein_complex', 'scRNA', 'SRP_RNA', 'tRNA', 'ribozyme', 'piRNA', 'protein', 'hammerhead_ribozyme', 'RNase_P_RNA', 'miRNA', 'tmRNA', 'antisense_RNA', 'siRNA', 'lnc_RNA', 'snoRNA', 'autocatalytically_spliced_intron', 'guide_RNA', 'telomerase_RNA', 'transcript', 'primary_transcript', 'RNase_MRP_RNA', 'vault_RNA'}

# Experimental evidence codes
# 'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP',
# high throughput evidence code
# 'HDA', 'HMP', 'HGI', 'HEP',
# Phylogenetically-inferred annotations
# 'IBA', 'IKR',
# Computational analysis evidence codes
# 'ISS', 'ISO', 'ISA', 'ISM', 'IGC', 'RCA'
# Author statement evidence codes
# 'TAS', 'NAS',
# Curator statement evidence codes
# 'IC', 'ND',
# Electronic annotation evidence code
# 'IEA',

# Processing: D:\go\goa\UNIPROT\goa_uniprot_all.gaf.gz
# ==ALL==

# db: {'RNAcentral', 'UniProtKB', 'ComplexPortal'}
# qual: {'', 'colocalizes_with', 'NOT', 'contributes_to', 'NOT|contributes_to', 'NOT|colocalizes_with'}
# evidence: {'HMP', 'ND', 'TAS', 'ISS', 'ISM', 'NAS', 'IDA', 'RCA', 'IGI', 'HDA', 'HGI', 'IBA', 'IEP', 'EXP', 'HEP', 'IEA', 'IPI', 'IKR', 'IMP', 'IGC', 'ISO', 'IC', 'ISA'}
# type: {'transcript', 'RNase_MRP_RNA', 'tmRNA', 'antisense_RNA', 'autocatalytically_spliced_intron', 'rRNA', 'lnc_RNA', 'scRNA',
# 'vault_RNA', 'telomerase_RNA', 'tRNA', 'hammerhead_ribozyme', 'siRNA', 'ribozyme', 'primary_transcript', 'piRNA', 'Y_RNA', 'snoRNA', 'snRNA', 'protein', 'miRNA', 'ncRNA', 'protein_complex', 'SRP_RNA', 'RNase_P_RNA', 'guide_RNA'}

# db: 3
# seqid: 102989625
# symbol: 75301818
# qual: 6
# go: 29538
# evidence: 23
# type: 26
# tax: 1334633

# ==QUAL COUNT==
# NOT: 9695
# contributes_to: 40288
# colocalizes_with: 15959
# NOT|colocalizes_with: 38
# NOT|contributes_to: 4
# ==PROTEIN ONLY==

# db: {'UniProtKB'}
# qual: {'', 'colocalizes_with', 'NOT', 'contributes_to', 'NOT|contributes_to', 'NOT|colocalizes_with'}
# evidence: {'HMP', 'ND', 'TAS', 'ISS', 'ISM', 'NAS', 'IDA', 'RCA', 'IGI', 'HDA', 'HGI', 'IBA', 'IEP', 'EXP', 'HEP', 'IEA', 'IPI', 'IKR', 'IMP', 'IGC', 'ISO', 'IC', 'ISA'}
# type: {'protein'}

# db: 1
# seqid: 96293088
# symbol: 68605281
# qual: 6
# go: 29348
# evidence: 23
# type: 1
# tax: 1014100

# ==QUAL COUNT==
# NOT: 9668
# contributes_to: 40258
# colocalizes_with: 15928
# NOT|colocalizes_with: 37
# NOT|contributes_to: 4
# ==EXCLUDE IEA==

# db: {'UniProtKB'}
# qual: {'', 'colocalizes_with', 'NOT', 'contributes_to', 'NOT|contributes_to', 'NOT|colocalizes_with'}
# evidence: {'HMP', 'ND', 'TAS', 'ISS', 'ISM', 'NAS', 'IDA', 'RCA', 'IGI', 'HDA', 'HGI', 'IBA', 'IEP', 'EXP', 'HEP', 'IPI', 'IKR', 'IMP', 'IGC', 'ISO', 'IC', 'ISA'}
# type: {'protein'}

# db: 1
# seqid: 755728
# symbol: 533011
# qual: 6
# go: 27651
# evidence: 22
# type: 1
# tax: 4946

# ==QUAL COUNT==
# NOT: 9080
# contributes_to: 40234
# colocalizes_with: 15928
# NOT|colocalizes_with: 37
# NOT|contributes_to: 4
# ==EXP CODES ONLY==

# db: {'UniProtKB'}
# qual: {'', 'colocalizes_with', 'NOT', 'contributes_to', 'NOT|contributes_to', 'NOT|colocalizes_with'}
# evidence: {'IDA', 'IGI', 'IPI', 'IMP', 'IEP', 'EXP'}
# type: {'protein'}

# db: 1
# seqid: 123585
# symbol: 81756
# qual: 6
# go: 26741
# evidence: 6
# type: 1
# tax: 3053

# ==QUAL COUNT==
# NOT: 4294
# colocalizes_with: 2851
# contributes_to: 2968
# NOT|colocalizes_with: 23
# NOT|contributes_to: 4
# Run time: 3605.09 s

# Processing: D:\go\goa\UNIPROT\goa_uniprot_gcrp.gaf.gz
# ==ALL==

# db: {'UniProtKB'}
# qual: {'', 'NOT|colocalizes_with', 'NOT|contributes_to', 'NOT', 'contributes_to', 'colocalizes_with'}
# evidence: {'ND', 'IGI', 'EXP', 'TAS', 'HMP', 'NAS', 'IEA', 'HDA', 'IGC', 'RCA', 'IBA', 'IMP', 'HGI', 'IEP', 'IC', 'ISO', 'ISM',
# 'IPI', 'ISA', 'ISS', 'HEP', 'IKR', 'IDA'}
# type: {'protein'}

# db: 1
# seqid: 23582588
# symbol: 18832370
# qual: 6
# go: 28694
# evidence: 23
# type: 1
# tax: 12822

# ==QUAL COUNT==
# NOT: 8018
# contributes_to: 36179
# colocalizes_with: 13114
# NOT|colocalizes_with: 32
# NOT|contributes_to: 4
# ==PROTEIN ONLY==

# db: {'UniProtKB'}
# qual: {'', 'NOT|colocalizes_with', 'NOT|contributes_to', 'NOT', 'contributes_to', 'colocalizes_with'}
# evidence: {'ND', 'IGI', 'EXP', 'TAS', 'HMP', 'NAS', 'IEA', 'HDA', 'IGC', 'RCA', 'IBA', 'IMP', 'HGI', 'IEP', 'IC', 'ISO', 'ISM',
# 'IPI', 'ISA', 'ISS', 'HEP', 'IKR', 'IDA'}
# type: {'protein'}

# db: 1
# seqid: 23582588
# symbol: 18832370
# qual: 6
# go: 28694
# evidence: 23
# type: 1
# tax: 12822

# ==QUAL COUNT==
# NOT: 8018
# contributes_to: 36179
# colocalizes_with: 13114
# NOT|colocalizes_with: 32
# NOT|contributes_to: 4
# ==EXCLUDE IEA==

# db: {'UniProtKB'}
# qual: {'', 'NOT|colocalizes_with', 'NOT|contributes_to', 'NOT', 'contributes_to', 'colocalizes_with'}
# evidence: {'ND', 'IGI', 'EXP', 'TAS', 'HMP', 'NAS', 'HDA', 'IGC', 'RCA', 'IBA', 'IMP', 'HGI', 'IEP', 'IC', 'ISO', 'ISM', 'IPI',
# 'ISA', 'ISS', 'HEP', 'IKR', 'IDA'}
# type: {'protein'}

# db: 1
# seqid: 652694
# symbol: 501094
# qual: 6
# go: 27021
# evidence: 22
# type: 1
# tax: 1200

# ==QUAL COUNT==
# NOT: 7461
# contributes_to: 36156
# colocalizes_with: 13114
# NOT|colocalizes_with: 32
# NOT|contributes_to: 4
# ==EXP CODES ONLY==

# db: {'UniProtKB'}
# qual: {'', 'NOT|colocalizes_with', 'NOT|contributes_to', 'NOT', 'contributes_to', 'colocalizes_with'}
# evidence: {'IMP', 'IGI', 'EXP', 'IEP', 'IPI', 'IDA'}
# type: {'protein'}

# db: 1
# seqid: 88128
# symbol: 74328
# qual: 6
# go: 26035
# evidence: 6
# type: 1
# tax: 877

# ==QUAL COUNT==
# NOT: 3195
# colocalizes_with: 2290
# contributes_to: 2622
# NOT|colocalizes_with: 20
# NOT|contributes_to: 4
# Run time: 820.73 s
