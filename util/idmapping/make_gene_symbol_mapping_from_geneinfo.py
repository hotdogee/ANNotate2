import os
import re
import csv
import time
import gzip
import json
import errno
import struct
import argparse
from tqdm import tqdm
from glob import glob
from pathlib import Path
from pprint import pprint
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict
from ..verify_paths import verify_input_path, verify_output_path, verify_indir_path, verify_outdir_path

# Download: ftp://ftp.ncbi.nih.gov/gene/DATA/gene_info.gz
# #tax_id GeneID  Symbol  LocusTag        Synonyms        dbXrefs chromosome      map_location    description     type_of_gene    Symbol_from_nomenclature_authority      Full_name_from_nomenclature_authority   Nomenclature_status     Other_designations      Modification_date       Feature_type
# 9       1246500 repA1   pLeuDn_01       -       -       -       -       putative replication-associated protein protein-coding  -       -       -       -       20180129        -
# 9       1246501 repA2   pLeuDn_03       -       -       -       -       putative replication-associated protein protein-coding  -       -       -       -       20180129        -
# 9606    4576    TRNT    -       MTTT    MIM:590090|HGNC:HGNC:7499       MT      -       tRNA    tRNA    MT-TT   mitochondrially encoded tRNA threonine  O       -       20191019        -
# 1425170 17961592        TRNT    -       MTTT    -       MT      -       tRNA    tRNA    -       -       -       -       20131227        -
# 9913    532990  MACROH2A1       -       H2AFY   VGNC:VGNC:83608|BGD:BT26863|Ensembl:ENSBTAG00000016105  7       -       macroH2A.1 histone      protein-coding  MACROH2A1       macroH2A.1 histone      O       core histone macro-H2A.1|H2A histone family, member Y   20191104        -
# 10090   26914   Macroh2a1       -       H2AF12M|H2afy|mH2a1     MGI:MGI:1349392|Ensembl:ENSMUSG00000015937      13      13|13 B1        macroH2A.1 histone      protein-coding  Macroh2a1       macroH2A.1 histone      O       core histone macro-H2A.1|H2A histone family, member Y|H2A.y|H2A/y|MACROH2A1.2|histone macroH2A1 20191010        -
# 7897    102357682       GARS1   -       GARS    Ensembl:ENSLACG00000002810      Un      -       glycyl-tRNA synthetase 1        protein-coding  -       -       -       glycine--tRNA ligase    20190717        -
# 8469    102936147       GARS1   UY3_13464       GARS    -       Un      -       glycyl-tRNA synthetase 1        protein-coding  -       -       -       glycine--tRNA ligase    20190717        -


def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def explore_unfound_genes(path):
    # initialize
    unfound_genes = {
        'TRNN', 'HYMAI', 'DUX4L1', 'TRNS1', 'MT-TV', 'SPG16', 'GNAS-AS1',
        'KCNQ1OT1', 'MACROH2A1', 'USH1K', 'MT-TN', 'HELLPAR', 'CXORF56',
        'DYT13', 'MT-TK', 'USH1E', 'TRNS2', 'MT-TT', 'PWRN1', 'XIST', 'GARS1',
        'USH1H', 'MT-TF', 'SARS1', 'SNORD116-1', 'SPG37', 'FRA16E', 'MT-TW',
        'TRNC', 'SCA30', 'IARS1', 'AARS1', 'SPG34', 'SLC7A2-IT1', 'SPG14',
        'PWAR1', 'TRNQ', 'SCA37', 'MT-TS2', 'SNORD115-1', 'HBB-LCR', 'EPRS1',
        'KIFBP', 'MIR184', 'TRNL2', 'OPA2', 'C12ORF57', 'RNU4ATAC', 'MKRN3-AS1',
        'H1-4', 'MT-TP', 'H19-ICR', 'TRNK', 'SCA20', 'SNORD118', 'TRNW',
        'DARS1', 'TRNI', 'MT-TE', 'SPG25', 'RARS1', 'SPG23', 'TRNE', 'MIR96',
        'C12ORF4', 'MT-TQ', 'MIR204', 'SCA25', 'SPG41', 'SPG29', 'TRNT',
        'IL12A-AS1', 'DISC2', 'MT-TL1', 'IPW', 'FMR3', 'SPG36', 'SCA32',
        'C9ORF72', 'MT-TH', 'MARS1', 'RNU12', 'DYT17', 'WHCR', 'STING1',
        'YARS1', 'TRNV', 'KARS1', 'GINGF2', 'TRNF', 'DYT15', 'LARS1', 'ARSL',
        'SPG38', 'SPG19', 'DYT21', 'TRNP', 'SPG32', 'SPG27', 'QARS1', 'MT-TS1',
        'TRNL1', 'RMRP', 'C11ORF95', 'MT-TL2', 'HARS1', 'ADSS1', 'SPG24',
        'C15ORF41'
    }
    found_in = defaultdict(set)
    gene_types = set()
    unfound_genes_types = defaultdict(set)
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
        dynamic_ncols=True, ascii=True, desc=path.name, unit='lines'
    ) as t:
        rows = csv.reader(f, delimiter='\t')
        for row in rows:
            t.update()
            if row[0][0] == '#':
                continue
            tax_id = int(row[0].strip().upper())
            # if tax_id != 9606:
            #     continue
            symbol = row[2].strip().upper()
            assert (symbol != '-'), f'empty symbol: {row}'
            symbol_n = row[10].strip().upper()
            synonyms = row[4].strip().upper().split('|')
            gene_type = row[9].strip().upper()
            gene_types.add(gene_type)
            # check which fields the unfound genes are located
            if symbol in unfound_genes:
                found_in['symbol'].add(symbol)
                unfound_genes_types[symbol].add(gene_type)
            if symbol_n in unfound_genes:
                found_in['symbol_n'].add(symbol_n)
                unfound_genes_types[symbol_n].add(gene_type)
            for s in synonyms:
                if s in unfound_genes:
                    found_in['synonyms'].add(s)
                    unfound_genes_types[s].add(gene_type)
    print(f'Genes without a sequence ({len(unfound_genes)}): {unfound_genes}')
    print(f" - symbol ({len(found_in['symbol'])}): {found_in['symbol']}")
    print(f" - symbol_n ({len(found_in['symbol_n'])}): {found_in['symbol_n']}")
    print(f" - synonyms ({len(found_in['synonyms'])}): {found_in['synonyms']}")
    print(
        f" - symbol | symbol_n ({len(found_in['symbol'] | found_in['symbol_n'])}): {found_in['symbol'] | found_in['symbol_n']}"
    )
    print(
        f" - symbol | symbol_n | synonyms ({len(found_in['symbol'] | found_in['symbol_n'] | found_in['synonyms'])}): {found_in['symbol'] | found_in['symbol_n'] | found_in['synonyms']}"
    )
    print(f'Gene types ({len(gene_types)}): {gene_types}')
    # dict([(s, list(unfound_genes_types[s])) for s in unfound_genes_types])
    pprint(
        dict([(s, list(unfound_genes_types[s])) for s in unfound_genes_types])
    )


# gene_info: 25794322lines [01:37, 264460.62lines/s]
# Genes without a sequence (109): {'ADSS1', 'SCA30', 'SPG36', 'SLC7A2-IT1', 'SPG14', 'AARS1', 'MT-TH', 'XIST', 'SCA37', 'SPG25', 'MT-TQ', 'C12ORF57', 'SNORD115-1', 'MKRN3-AS1', 'MACROH2A1', 'HARS1',
# 'SPG24', 'IARS1', 'MT-TS2', 'MT-TF', 'RNU4ATAC', 'SPG19', 'HYMAI', 'PWRN1', 'TRNK', 'C15ORF41', 'SCA25', 'DYT13', 'SPG23', 'TRNF', 'TRNP', 'MIR96', 'SCA20', 'TRNE', 'H1-4', 'TRNC', 'TRNI', 'SPG37', 'DYT15', 'C11ORF95', 'WHCR', 'HELLPAR', 'SCA32', 'RNU12', 'MT-TL2', 'SPG41', 'USH1H', 'MIR184', 'KIFBP', 'SNORD118', 'YARS1', 'KCNQ1OT1', 'TRNL2', 'TRNT', 'TRNL1', 'SPG38', 'SPG32', 'STING1', 'MT-TL1', 'ARSL', 'USH1E', 'EPRS1', 'MT-TV', 'TRNV', 'H19-ICR', 'DYT17', 'MT-TK', 'C12ORF4', 'MIR204', 'DISC2', 'LARS1', 'TRNS2', 'GINGF2', 'HBB-LCR', 'DYT21', 'USH1K', 'SPG34', 'SPG27', 'OPA2', 'GARS1', 'CXORF56', 'TRNW', 'MT-TT', 'PWAR1', 'RMRP', 'MT-TE', 'MT-TW', 'SPG16', 'SNORD116-1', 'QARS1', 'KARS1', 'C9ORF72', 'MT-TN', 'MT-TP', 'SPG29', 'TRNS1', 'MT-TS1', 'SARS1', 'DARS1', 'TRNN', 'RARS1', 'FMR3', 'TRNQ', 'FRA16E', 'GNAS-AS1', 'IPW', 'MARS1', 'DUX4L1', 'IL12A-AS1'}
#  - symbol (91): {'ADSS1', 'SCA30', 'SPG36', 'SPG14', 'AARS1', 'XIST', 'SCA37', 'SPG25', 'C12ORF57', 'SNORD115-1', 'MKRN3-AS1', 'HARS1', 'MACROH2A1', 'SPG24', 'IARS1', 'RNU4ATAC', 'SPG19', 'HYMAI',
# 'TRNK', 'PWRN1', 'C15ORF41', 'SCA25', 'DYT13', 'TRNF', 'TRNP', 'MIR96', 'SCA20', 'TRNE', 'H1-4', 'TRNC', 'TRNI', 'SPG37', 'DYT15', 'C11ORF95', 'WHCR', 'HELLPAR', 'RNU12', 'SPG41', 'MIR184', 'USH1H', 'KIFBP', 'SNORD118', 'YARS1', 'TRNL2', 'KCNQ1OT1', 'TRNL1', 'TRNT', 'SPG38', 'SPG32', 'STING1', 'ARSL', 'USH1E', 'EPRS1', 'TRNV', 'H19-ICR', 'DYT17', 'C12ORF4', 'MIR204', 'DISC2', 'LARS1', 'TRNS2', 'GINGF2', 'DYT21', 'HBB-LCR', 'USH1K', 'SPG34', 'SPG27', 'OPA2', 'GARS1', 'CXORF56', 'TRNW', 'PWAR1', 'RMRP', 'SPG16', 'SNORD116-1', 'QARS1', 'KARS1', 'C9ORF72', 'SPG29', 'TRNS1', 'SARS1', 'DARS1', 'TRNN', 'RARS1', 'TRNQ', 'FRA16E', 'GNAS-AS1', 'IPW', 'MARS1', 'DUX4L1', 'IL12A-AS1'}
#  - symbol_n (61): {'STING1', 'SNORD116-1', 'MT-TL1', 'ADSS1', 'ARSL', 'QARS1', 'KARS1', 'C9ORF72', 'EPRS1', 'MT-TV', 'MIR96', 'MT-TN', 'MT-TH', 'AARS1', 'H1-4', 'MT-TP', 'XIST', 'MT-TK', 'MT-TS1',
# 'C11ORF95', 'MT-TQ', 'C12ORF4', 'HELLPAR', 'C12ORF57', 'SNORD115-1', 'MKRN3-AS1', 'HARS1', 'SARS1', 'MACROH2A1', 'MIR204', 'DARS1', 'DISC2', 'LARS1', 'RNU12', 'MT-TL2', 'IARS1', 'MIR184', 'RARS1',
# 'MT-TS2', 'MT-TF', 'RNU4ATAC', 'KIFBP', 'SNORD118', 'GARS1', 'HYMAI', 'PWRN1', 'YARS1', 'CXORF56', 'C15ORF41', 'MT-TT', 'PWAR1', 'RMRP', 'FRA16E', 'KCNQ1OT1', 'MT-TE', 'GNAS-AS1', 'IPW', 'MARS1', 'DUX4L1', 'IL12A-AS1', 'MT-TW'}
#  - synonyms (8): {'XIST', 'RNU12', 'DYT17', 'SPG29', 'SPG23', 'SCA37', 'TRNP', 'TRNE'}
#  - symbol | symbol_n (105): {'SPG38', 'SPG32', 'STING1', 'MT-TL1', 'ADSS1', 'ARSL', 'SCA30', 'USH1E', 'SPG36', 'EPRS1', 'TRNV', 'MT-TV', 'SPG14', 'H19-ICR', 'AARS1', 'MT-TH', 'XIST', 'DYT17', 'MT-TK', 'SCA37', 'SPG25', 'MT-TQ', 'C12ORF4', 'C12ORF57', 'SNORD115-1', 'MKRN3-AS1', 'HARS1', 'MACROH2A1', 'MIR204', 'SPG24', 'DISC2', 'LARS1', 'TRNS2', 'GINGF2', 'DYT21', 'HBB-LCR', 'USH1K', 'IARS1',
# 'SPG34', 'SPG27', 'MT-TS2', 'MT-TF', 'RNU4ATAC', 'SPG19', 'OPA2', 'GARS1', 'HYMAI', 'TRNK', 'PWRN1', 'CXORF56', 'C15ORF41', 'TRNW', 'MT-TT', 'PWAR1', 'RMRP', 'MT-TE', 'MT-TW', 'SCA25', 'DYT13', 'SPG16', 'SNORD116-1', 'QARS1', 'TRNF', 'KARS1', 'TRNP', 'C9ORF72', 'MIR96', 'SCA20', 'TRNE', 'MT-TN', 'MT-TP', 'H1-4', 'TRNC', 'TRNI', 'SPG37', 'DYT15', 'SPG29', 'C11ORF95', 'TRNS1', 'WHCR', 'MT-TS1', 'HELLPAR', 'SARS1', 'DARS1', 'RNU12', 'MT-TL2', 'SPG41', 'MIR184', 'USH1H', 'TRNN', 'RARS1', 'KIFBP', 'SNORD118', 'YARS1', 'TRNQ', 'FRA16E', 'TRNL2', 'KCNQ1OT1', 'GNAS-AS1', 'IPW', 'MARS1', 'TRNL1', 'TRNT', 'DUX4L1', 'IL12A-AS1'}
#  - symbol | symbol_n | synonyms (106): {'SPG38', 'SPG32', 'STING1', 'MT-TL1', 'ADSS1', 'ARSL', 'SCA30', 'USH1E', 'SPG36', 'EPRS1', 'TRNV', 'MT-TV', 'SPG14', 'H19-ICR', 'AARS1', 'MT-TH', 'XIST', 'DYT17', 'MT-TK', 'SCA37', 'SPG25', 'MT-TQ', 'C12ORF4', 'C12ORF57', 'SNORD115-1', 'MKRN3-AS1', 'HARS1', 'MACROH2A1', 'MIR204', 'SPG24', 'DISC2', 'LARS1', 'TRNS2', 'GINGF2', 'DYT21', 'HBB-LCR', 'USH1K', 'IARS1', 'SPG34', 'SPG27', 'MT-TS2', 'MT-TF', 'RNU4ATAC', 'SPG19', 'OPA2', 'GARS1', 'HYMAI', 'TRNK', 'PWRN1', 'CXORF56', 'C15ORF41', 'TRNW', 'MT-TT', 'PWAR1', 'RMRP', 'MT-TE', 'MT-TW', 'SCA25', 'DYT13', 'SPG16', 'SNORD116-1', 'SPG23', 'QARS1', 'TRNF', 'KARS1', 'TRNP', 'C9ORF72', 'MIR96', 'SCA20', 'TRNE', 'MT-TN', 'MT-TP', 'H1-4', 'TRNC', 'TRNI', 'SPG37', 'DYT15', 'SPG29', 'C11ORF95', 'TRNS1', 'WHCR', 'MT-TS1', 'HELLPAR', 'SARS1', 'DARS1', 'RNU12', 'MT-TL2', 'SPG41', 'MIR184', 'USH1H', 'TRNN', 'RARS1', 'KIFBP', 'SNORD118', 'YARS1', 'TRNQ', 'FRA16E', 'TRNL2', 'KCNQ1OT1', 'GNAS-AS1', 'IPW', 'MARS1', 'TRNL1', 'TRNT', 'DUX4L1', 'IL12A-AS1'}
# Gene types (11): {'PROTEIN-CODING', 'SCRNA', 'SNRNA', 'BIOLOGICAL-REGION', 'NCRNA', 'SNORNA', 'UNKNOWN', 'PSEUDO', 'RRNA', 'TRNA', 'OTHER'}
# {'SLC7A2-IT1', 'SCA32', 'FMR3'}

# {'AARS1': ['PROTEIN-CODING'],
#  'ADSS1': ['PROTEIN-CODING'],
#  'ARSL': ['PROTEIN-CODING'],
#  'C11ORF95': ['PROTEIN-CODING'],
#  'C12ORF4': ['PROTEIN-CODING'],
#  'C12ORF57': ['PROTEIN-CODING'],
#  'C15ORF41': ['PROTEIN-CODING'],
#  'C9ORF72': ['PROTEIN-CODING'],
#  'CXORF56': ['PROTEIN-CODING'],
#  'DARS1': ['PROTEIN-CODING'],
#  'DISC2': ['NCRNA'],
#  'DUX4L1': ['PSEUDO'],
#  'DYT13': ['UNKNOWN'],
#  'DYT15': ['UNKNOWN'],
#  'DYT17': ['PROTEIN-CODING', 'UNKNOWN'],
#  'DYT21': ['UNKNOWN'],
#  'EPRS1': ['PROTEIN-CODING'],
#  'FRA16E': ['OTHER'],
#  'GARS1': ['PROTEIN-CODING'],
#  'GINGF2': ['UNKNOWN'],
#  'GNAS-AS1': ['NCRNA'],
#  'H1-4': ['PROTEIN-CODING'],
#  'H19-ICR': ['BIOLOGICAL-REGION'],
#  'HARS1': ['PROTEIN-CODING'],
#  'HBB-LCR': ['BIOLOGICAL-REGION'],
#  'HELLPAR': ['NCRNA'],
#  'HYMAI': ['NCRNA'],
#  'IARS1': ['PROTEIN-CODING'],
#  'IL12A-AS1': ['NCRNA'],
#  'IPW': ['NCRNA'],
#  'KARS1': ['PROTEIN-CODING'],
#  'KCNQ1OT1': ['NCRNA'],
#  'KIFBP': ['PROTEIN-CODING'],
#  'LARS1': ['PROTEIN-CODING'],
#  'MACROH2A1': ['PROTEIN-CODING'],
#  'MARS1': ['PROTEIN-CODING'],
#  'MIR184': ['NCRNA'],
#  'MIR204': ['NCRNA'],
#  'MIR96': ['NCRNA'],
#  'MKRN3-AS1': ['NCRNA'],
#  'MT-TE': ['TRNA'],
#  'MT-TF': ['TRNA'],
#  'MT-TH': ['TRNA'],
#  'MT-TK': ['PROTEIN-CODING', 'TRNA'],
#  'MT-TL1': ['TRNA'],
#  'MT-TL2': ['TRNA'],
#  'MT-TN': ['TRNA'],
#  'MT-TP': ['TRNA'],
#  'MT-TQ': ['TRNA'],
#  'MT-TS1': ['TRNA'],
#  'MT-TS2': ['TRNA'],
#  'MT-TT': ['TRNA'],
#  'MT-TV': ['TRNA'],
#  'MT-TW': ['TRNA'],
#  'OPA2': ['UNKNOWN'],
#  'PWAR1': ['NCRNA'],
#  'PWRN1': ['NCRNA'],
#  'QARS1': ['PROTEIN-CODING'],
#  'RARS1': ['PROTEIN-CODING'],
#  'RMRP': ['NCRNA'],
#  'RNU12': ['SNRNA', 'PSEUDO'],
#  'RNU4ATAC': ['SNRNA'],
#  'SARS1': ['PROTEIN-CODING'],
#  'SCA20': ['UNKNOWN'],
#  'SCA25': ['UNKNOWN'],
#  'SCA30': ['UNKNOWN'],
#  'SCA37': ['PROTEIN-CODING', 'UNKNOWN'],
#  'SNORD115-1': ['SNORNA'],
#  'SNORD116-1': ['SNORNA'],
#  'SNORD118': ['SNORNA'],
#  'SPG14': ['UNKNOWN'],
#  'SPG16': ['UNKNOWN'],
#  'SPG19': ['UNKNOWN'],
#  'SPG23': ['PROTEIN-CODING'],
#  'SPG24': ['UNKNOWN'],
#  'SPG25': ['UNKNOWN'],
#  'SPG27': ['UNKNOWN'],
#  'SPG29': ['UNKNOWN'],
#  'SPG32': ['UNKNOWN'],
#  'SPG34': ['UNKNOWN'],
#  'SPG36': ['UNKNOWN'],
#  'SPG37': ['UNKNOWN'],
#  'SPG38': ['UNKNOWN'],
#  'SPG41': ['UNKNOWN'],
#  'STING1': ['PROTEIN-CODING'],
#  'TRNE': ['PROTEIN-CODING', 'TRNA', 'PSEUDO'],
#  'TRNF': ['NCRNA', 'TRNA', 'PSEUDO'],
#  'TRNI': ['TRNA', 'OTHER', 'RRNA', 'PSEUDO', 'PROTEIN-CODING'],
#  'TRNK': ['PROTEIN-CODING', 'TRNA', 'OTHER'],
#  'TRNL1': ['TRNA'],
#  'TRNL2': ['TRNA'],
#  'TRNN': ['TRNA', 'PSEUDO'],
#  'TRNP': ['PROTEIN-CODING', 'TRNA', 'PSEUDO'],
#  'TRNQ': ['PROTEIN-CODING', 'TRNA'],
#  'TRNS1': ['TRNA'],
#  'TRNS2': ['TRNA'],
#  'TRNT': ['TRNA', 'OTHER', 'PSEUDO'],
#  'TRNW': ['TRNA', 'PSEUDO'],
#  'USH1E': ['UNKNOWN'],
#  'USH1H': ['UNKNOWN'],
#  'USH1K': ['UNKNOWN'],
#  'WHCR': ['OTHER'],
#  'XIST': ['NCRNA', 'OTHER'],
#  'YARS1': ['PROTEIN-CODING']}


def get_gene_symbol_mapping(path):
    # initialize
    gene_symbol_mapping = {}
    gene_type_mapping = {}
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
        dynamic_ncols=True, ascii=True, desc=path.name, unit='lines'
    ) as t:
        rows = csv.reader(f, delimiter='\t')
        for row in rows:
            t.update()
            if row[0][0] == '#':
                continue
            tax_id = int(row[0].strip().upper())
            if tax_id != 9606:
                continue
            symbol = row[2].strip().upper()
            assert (symbol != '-'), f'empty symbol: {row}'
            symbol_n = row[10].strip().upper()
            synonyms = row[4].strip().upper().split('|')
            gene_type = row[9].strip().upper()
    #         symbol_list = []
    #         if symbol != '-':
    #             symbol_list.append(symbol)
    #         if symbol_n != '-':
    #             symbol_list.append(symbol_n)
    #         if synonyms[0] != '-':
    #             symbol_list += synonyms
    #         if len(symbol_list) == 0:
    #             continue
    #         found = set(
    #             [
    #                 gene_symbol_mapping[s]
    #                 for s in symbol_list if s in gene_symbol_mapping
    #             ]
    #         )
    #         # if 'REPZ' in symbol_list:
    #         #     print(row, found)
    #         # if 'REPZ' in found:
    #         #     print(
    #         #         row, found, gene_symbol_mapping['REPZ'],
    #         #         gene_symbol_mapping['REPA']
    #         #     )
    #         # if len(found) >= 2:
    #         #     print(
    #         #         f'found length: {len(found)}, expected: 0 or 1, {found}, {symbol_list}'
    #         #     )
    #         #     continue
    #         assert (
    #             len(found) < 2
    #         ), f'found length: {len(found)}, expected: 0 or 1, {found}, {symbol_list}'
    #         if len(found) == 0:
    #             main = symbol_list[0]
    #         else:
    #             main = list(found)[0]
    #         for s in symbol_list:
    #             if s not in gene_symbol_mapping:
    #                 gene_symbol_mapping[s] = main
    #             else:
    #                 assert (
    #                     gene_symbol_mapping[s] == main,
    #                     f'{row[1]} {symbol} gene_type conflict: {gene_symbol_mapping[s]} != {main}'
    #                 )

    #             if s not in gene_type_mapping:
    #                 if gene_type != '-':
    #                     gene_type_mapping[s] = gene_type
    #             else:
    #                 assert (
    #                     gene_type_mapping[s] == gene_type,
    #                     f'{row[1]} {symbol} gene_type conflict: {gene_type_mapping[s]} != {gene_type}'
    #                 )
    # return gene_symbol_mapping, gene_type_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate gene_symbol_mapping.json from gene_info[.gz].'
    )
    parser.add_argument(
        '-i',
        '--info',
        type=str,
        required=True,
        help="Path to read gene_info[.gz] file, required."
    )
    parser.add_argument(
        '-s',
        '--symbol',
        type=str,
        required=True,
        help="Path to write gene_symbol_mapping.json file, required."
    )
    parser.add_argument(
        '-t',
        '--type',
        type=str,
        required=True,
        help="Path to write gene_type_mapping.json file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    info_path = verify_input_path(args.info)
    print(f'Processing: {info_path.name}')

    explore_unfound_genes(info_path)

    # gene_symbol_mapping = get_gene_symbol_mapping(info_path)
    # # chr(10) = '\n'
    # print(f'Total mappings: ({len(gene_symbol_mapping)})')
    # out_path = verify_output_path(args.out)
    # out_path.write_text(
    #     json.dumps(gene_symbol_mapping, indent=2, sort_keys=True)
    # )

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# windows
# python -m util.idmapping.make_gene_symbol_mapping_from_geneinfo --info E:\gene\gene_info --symbol E:\gene\gene_info\gene_symbol_mapping.json --type E:\gene\gene_info\gene_type_mapping.json

# idmapping.dat: 2,442,456,179 lines [59:43, 681539 lines/s]
# ID Types (104):
# Allergome
# ArachnoServer
# Araport
# BioCyc
# BioGrid
# BioMuta
# CCDS
# CGD
# CPTAC
# CRC64
# ChEMBL
# ChiTaRS
# CollecTF
# ComplexPortal
# ConoServer
# DIP
# DMDM
# DNASU
# DisProt
# DrugBank
# EMBL
# EMBL-CDS
# ESTHER
# EchoBASE
# EcoGene
# Ensembl
# EnsemblGenome
# EnsemblGenome_PRO
# EnsemblGenome_TRS
# Ensembl_PRO
# Ensembl_TRS
# EuPathDB
# FlyBase
# GI
# GeneCards
# GeneDB
# GeneID
# GeneReviews
# GeneTree
# GeneWiki
# Gene_Name
# Gene_ORFName
# Gene_OrderedLocusName
# Gene_Synonym
# GenomeRNAi
# GlyConnect
# GuidetoPHARMACOLOGY
# HGNC
# HOGENOM
# HPA
# KEGG
# KO
# LegioList
# Leproma
# MEROPS
# MGI
# MIM
# MINT
# MaizeGDB
# NCBI_TaxID
# OMA
# Orphanet
# OrthoDB
# PATRIC
# PDB
# PeroxiBase
# PharmGKB
# PlantReactome
# PomBase
# ProteomicsDB
# PseudoCAP
# REBASE
# RGD
# Reactome
# RefSeq
# RefSeq_NT
# SGD
# STRING
# SwissLipids
# TAIR
# TCDB
# TreeFam
# TubercuList
# UCSC
# UniParc
# UniPathway
# UniProtKB-ID
# UniRef100
# UniRef50
# UniRef90
# VGNC
# VectorBase
# WBParaSite
# World-2DPAGE
# WormBase
# WormBase_PRO
# WormBase_TRS
# Xenbase
# ZFIN
# dictyBase
# eggNOG
# euHCVdb
# mycoCLAP
# neXtProt
# Run time: 3583.82 s
