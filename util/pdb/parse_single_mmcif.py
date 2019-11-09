import os
import re
import time
import gzip
import errno
import argparse
from glob import glob
from pathlib import Path

# windows
# python .\util\pdb\parse_single_mmcif.py --mmcif D:/pdb/1zni.cif.gz
# python .\util\pdb\parse_single_mmcif.py --mmcif D:/pdb/1zni.cif

# Amino acid FASTA codification dictionary
codification = { "ALA" : 'A',
                 "CYS" : 'C',
                 "ASP" : 'D',
                 "GLU" : 'E',
                 "PHE" : 'F',
                 "GLY" : 'G',
                 "HIS" : 'H',
                 "ILE" : 'I',
                 "LYS" : 'K',
                 "LEU" : 'L',
                 "MET" : 'M',
                 "ASN" : 'N',
                 "PYL" : 'O',
                 "PRO" : 'P',
                 "GLN" : 'Q',
                 "ARG" : 'R',
                 "SER" : 'S',
                 "THR" : 'T',
                 "SEC" : 'U',
                 "VAL" : 'V',
                 "TRP" : 'W',
                 "TYR" : 'Y' }

def parse_mmcif(path):
    path = Path(path)
    # if gzipped
    if path.suffix == '.gz':
        f = gzip.open(path, mode='rt', encoding='utf-8')
    else:
        f = path.open(mode='r', encoding='utf-8')

    with f:
        for line in f:
            # print(line)
            # break
            # ID
            if line[:len('_entry.id')] == '_entry.id':
                id = line.split()[1].strip()
                print(id)
                break
            # SEQ
            # _struct_ref.pdbx_seq_one_letter_code 74.0 % of entries
            # ;MALWTRLLPLLALLALWAPAPAQAFVNQHLCGSHLVEALYLVCGERGFFYTPKARREAENPQAGAVELGGGLGGLQALAL
            # EGPPQKRGIVEQCCTSICSLYQLENYCN
            # _entity_poly.pdbx_seq_one_letter_code 100.0 % of entries
            # _entity_poly.pdbx_seq_one_letter_code_can 100.0 % of entries
            # 1 'polypeptide(L)' no no GIVEQCCTSICSLYQLENYCN          GIVEQCCTSICSLYQLENYCN          A,C ?
            # 2 'polypeptide(L)' no no FVNQHLCGSHLVEALYLVCGERGFFYTPKA FVNQHLCGSHLVEALYLVCGERGFFYTPKA B,D ?
            # ATOM
            # _atom_site
            # MASK?
            # SECONDARY


if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description='Compares the contents of mmCIF and PDB directories.')
    parser.add_argument('-m', '--mmcif', type=str, required=True,
        help="Path to .cif.gz file, required.")

    args, unparsed = parser.parse_known_args()
    # print(f'mmCIF: {args.mmcif}\n')

    start_time = time.time()

    # get absolute path to dataset directory
    mmcif_path = Path(os.path.abspath(os.path.expanduser(args.mmcif)))
    # print(f'mmCIF: {mmcif_path}\n')

    # doesn't exist
    if not mmcif_path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), mmcif_path)
    # is dir
    if mmcif_path.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), mmcif_path)
    print(f'mmCIF: {mmcif_path}')

    parse_mmcif(mmcif_path)
