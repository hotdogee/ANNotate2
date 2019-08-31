import os
import re
import time
import gzip
import errno
import argparse
from glob import glob
from pathlib import Path
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict

# windows
# python .\util\uniprot\compare_sprot_trembl.py --tsv1 D:/uniprot/uniprot-20190211/uniprot_sprot.dat.gz --tsv2 D:/uniprot/uniprot-20190211/uniprot_trembl.dat.gz
# ubuntu
# python compare_pfam_regions_tsv.py --tsv1 --tsv2

# uniprot_acc seq_version crc64            md5                              pfamA_acc seq_start seq_end
# A0A103YDS2  1           3C510D74FCA8479C 31e3d17a0f6766e2a545214e475d72a8 PF04863   52        107
# A0A087Y7L3  1           E79AF82467BBFA7D bb8465c4f3cd34f56f6c60817d43f551 PF09014   261       348
# A0A103YDS2  1           3C510D74FCA8479C 31e3d17a0f6766e2a545214e475d72a8 PF04863   433       488
# A0A103YDS2  1           3C510D74FCA8479C 31e3d17a0f6766e2a545214e475d72a8 PF04863   870       925
# A0A199V6R7  1           F37C9E162DD24DD7 0656490991e5645a3e2b402a861e450e PF04863   66        120
# A0A199V6R7  1           F37C9E162DD24DD7 0656490991e5645a3e2b402a861e450e PF04863   468       522
# A0A0D9V4Z5  1           25C8FDA89273F894 1ddc7bbbfec83e195473a9e51b763b16 PF04863   56        111
# W5N889      1           509B93CB8368603C 0e92ab8939197d40e00e0238de6b8a69 PF09014   264       351
# A0A0D9V4Z5  1           25C8FDA89273F894 1ddc7bbbfec83e195473a9e51b763b16 PF04863   552       607


FT_KEYS = ['TRANSMEM', 'DNA_BIND', 'REPEAT', 'REGION', 'COMPBIAS', 'INTRAMEM', 'DOMAIN', 'CA_BIND', 'COILED', 'ZN_FING', 'NP_BIND', 'MOTIF', 'TOPO_DOM', 'HELIX', 'STRAND', 'TURN', 'ACT_SITE', 'METAL', 'BINDING', 'SITE', 'NON_STD', 'MOD_RES', 'LIPID', 'CARBOHYD', 'DISULFID', 'CROSSLNK', 'VAR_SEQ', 'VARIANT', 'MUTAGEN', 'CONFLICT', 'UNSURE', 'NON_CONS', 'NON_TER', 'INIT_MET', 'SIGNAL', 'TRANSIT', 'PROPEP', 'CHAIN', 'PEPTIDE']

FT = namedtuple('FT', ['key', 'start', 'end'])

def stats_uniprot_dat(path):
    path = Path(path)
    # if gzipped
    if path.suffix == '.gz':
        f = gzip.open(path, mode='rt', encoding='utf-8')
    else:
        f = path.open(mode='r', encoding='utf-8')
    # initialize
    overlap = dict([(k, dict([(k, 0) for k in FT_KEYS])) for k in FT_KEYS])
    vocab = defaultdict(set)
    count = defaultdict(int)
    id = ''
    mp_e = 0 # Molecule processing
    ptm_e = 0 # PTM
    sf_e = 0 # Structure feathers
    entry = defaultdict(list)
    with f:
        for line in f:
            # line code
            lc = line[:2]

            # empty line
            line = line.strip()
            if line.strip() == '':
                continue

            # termination
            if lc == '//':
                count[lc] += 1
                # check overlapping features
                for i in range(len(entry['FT'])):
                    for j in range(i+1, len(entry['FT'])):
                        f1 = entry['FT'][i]
                        f2 = entry['FT'][j]
                        if f2.start <= f1.end and f2.end >= f1.start:
                            overlap[f1.key][f2.key] = 1
                            overlap[f2.key][f1.key] = 1
                # clear entry
                entry = defaultdict(list)
            elif lc == 'ID':
                count[lc] += 1
                id = line.split()[1]
                vocab[lc].add(id)
            elif lc == 'AC':
                for ac in [a.strip() for a in line[5:].split(';') if a != '']:
                    vocab[lc].add(ac)
            elif lc == 'OX':
                count[lc] += 1
            elif lc == 'PE':
                count[lc] += 1
            elif lc == 'SQ':
                count[lc] += 1
            elif lc == 'FT':
                vocab['FTID'].add(id)
                key = line[5:13].strip()
                start = line[14:20].strip()
                end = line[21:27].strip()
                if key == '' or start == '' or end == '':
                    continue
                if start[0] in ['>', '<', '?']:
                    start = start[1:]
                if end[0] in ['>', '<', '?']:
                    end = end[1:]
                if start == '' or end == '':
                    continue
                start = int(start)
                end = int(end)
                entry['FT'].append(FT(key, start, end))
                # if count[key] < 2:
                #     print(f'FT {key} - {id}')
                vocab[lc].add(key)
                count[key] += 1
    return vocab, count, overlap

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
    print(f'''
{n1}: {len(s1)} {unit}
{n2}: {len(s2)} {unit}
{n1} & {n2}: {len(s1 & s2)} {unit}
{n1} | {n2}: {len(s1 | s2)} {unit}
{n1} - {n2}: {len(s1 - s2)} {unit}
{n2} - {n1}: {len(s2 - s1)} {unit}
''')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Print the differences of two Pfam-A.regions.uniprot.tsv.gz files.')
    parser.add_argument('-1', '--tsv1', type=str, required=True,
        help="Path to the first Pfam-A.regions.uniprot.tsv.gz file, required.")
    parser.add_argument('-2', '--tsv2', type=str, required=True,
        help="Path to the second Pfam-A.regions.uniprot.tsv.gz file, required.")
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    tsv1_path = verify_input_path(args.tsv1)
    tsv2_path = verify_input_path(args.tsv2)
    print(f'Comparing: {tsv1_path.name} and {tsv2_path.name}')

    sprot_vocab, sprot_count, sprot_overlap = stats_uniprot_dat(sprot_path)
    trembl_vocab, trembl_count, trembl_overlap = stats_uniprot_dat(trembl_path)

    print('==ID set stats==')
    print_set_stats('sprot', sprot_vocab['ID'], 'trembl', trembl_vocab['ID'])

    print('==AC set stats==')
    print_set_stats('sprot', sprot_vocab['AC'], 'trembl', trembl_vocab['AC'])

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# Comparing: uniprot_sprot.dat.gz and uniprot_trembl.dat.gz==ID set stats==

# sprot: 559077
# trembl: 139694261
# sprot & trembl: 0
# sprot | trembl: 140253338
# sprot - trembl: 559077
# trembl - sprot: 139694261

# ==AC set stats==

# sprot: 774874
# trembl: 140179159
# sprot & trembl: 0
# sprot | trembl: 140954033
# sprot - trembl: 774874
# trembl - sprot: 140179159

# Run time: 10610.75 s
