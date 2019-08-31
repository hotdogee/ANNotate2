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
# python .\util\uniprot\stats_uniprot_dat.py --dat D:/uniprot/uniprot-20190211/uniprot_sprot.dat.gz
# python .\util\uniprot\stats_uniprot_dat.py --dat D:/uniprot/uniprot-20190211/uniprot_trembl.dat.gz

# ID   001R_FRG3G              Reviewed;         256 AA.
# AC   Q6GZX4;
# DT   28-JUN-2011, integrated into UniProtKB/Swiss-Prot.
# DT   19-JUL-2004, sequence version 1.
# DT   16-JAN-2019, entry version 35.
# DE   RecName: Full=Putative transcription factor 001R;
# GN   ORFNames=FV3-001R;
# OS   Frog virus 3 (isolate Goorha) (FV-3).
# OC   Viruses; dsDNA viruses, no RNA stage; Iridoviridae; Alphairidovirinae;
# OC   Ranavirus.
# OX   NCBI_TaxID=654924;
# OH   NCBI_TaxID=8295; Ambystoma (mole salamanders).
# OH   NCBI_TaxID=30343; Dryophytes versicolor (chameleon treefrog).
# OH   NCBI_TaxID=8404; Lithobates pipiens (Northern leopard frog) (Rana pipiens).
# OH   NCBI_TaxID=8316; Notophthalmus viridescens (Eastern newt) (Triturus viridescens).
# OH   NCBI_TaxID=45438; Rana sylvatica (Wood frog).
# RN   [1]
# RP   NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA].
# RX   PubMed=15165820; DOI=10.1016/j.virol.2004.02.019;
# RA   Tan W.G., Barkman T.J., Gregory Chinchar V., Essani K.;
# RT   "Comparative genomic analyses of frog virus 3, type species of the
# RT   genus Ranavirus (family Iridoviridae).";
# RL   Virology 323:70-84(2004).
# CC   -!- FUNCTION: Transcription activation. {ECO:0000305}.
# DR   EMBL; AY548484; AAT09660.1; -; Genomic_DNA.
# DR   RefSeq; YP_031579.1; NC_005946.1.
# DR   ProteinModelPortal; Q6GZX4; -.
# DR   SwissPalm; Q6GZX4; -.
# DR   GeneID; 2947773; -.
# DR   KEGG; vg:2947773; -.
# DR   Proteomes; UP000008770; Genome.
# DR   GO; GO:0046782; P:regulation of viral transcription; IEA:InterPro.
# DR   InterPro; IPR007031; Poxvirus_VLTF3.
# DR   Pfam; PF04947; Pox_VLTF3; 1.
# PE   4: Predicted;
# KW   Activator; Complete proteome; Reference proteome; Transcription;
# KW   Transcription regulation.
# FT   CHAIN         1    256       Putative transcription factor 001R.
# FT                                /FTId=PRO_0000410512.
# FT   COMPBIAS     14     17       Poly-Arg.
# SQ   SEQUENCE   256 AA;  29735 MW;  B4840739BF7D4121 CRC64;
#      MAFSAEDVLK EYDRRRRMEA LLLSLYYPND RKLLDYKEWS PPRVQVECPK APVEWNNPPS
#      EKGLIVGHFS GIKYKGEKAQ ASEVDVNKMC CWVSKFKDAM RRYQGIQTCK IPGKVLSDLD
#      AKIKAYNLTV EGVEGFVRYS RVTKQHVAAF LKELRHSKQY ENVNLIHYIL TDKRVDIQHL
#      EKDLVKDFKA LVESAHRMRQ GHMINVKYIL YQLLKKHGHG PDGPDILTVK TGSKGVLYDD
#      SFRKIYTDLG WKFTPL
# //
# ID   002L_FRG3G              Reviewed;         320 AA.
# AC   Q6GZX3;
# DT   28-JUN-2011, integrated into UniProtKB/Swiss-Prot.
# DT   19-JUL-2004, sequence version 1.
# DT   16-JAN-2019, entry version 36.
# DE   RecName: Full=Uncharacterized protein 002L;
# GN   ORFNames=FV3-002L;

# Line code	Content	Occurrence in an entry
# ID	Identification	Once; starts the entry
# AC	Accession number(s)	Once or more
# DT	Date	Three times
# DE	Description	Once or more
# GN	Gene name(s)	Optional
# OS	Organism species	Once or more
# OG	Organelle	Optional
# OC	Organism classification	Once or more
# OX	Taxonomy cross-reference	Once
# OH	Organism host	Optional
# RN	Reference number	Once or more
# RP	Reference position	Once or more
# RC	Reference comment(s)	Optional
# RX	Reference cross-reference(s)	Optional
# RG	Reference group	Once or more (Optional if RA line)
# RA	Reference authors	Once or more (Optional if RG line)
# RT	Reference title	Optional
# RL	Reference location	Once or more
# CC	Comments or notes	Optional
# DR	Database cross-references	Optional
# PE	Protein existence	Once
# KW	Keywords	Optional
# FT	Feature table data	Once or more in Swiss-Prot, optional in TrEMBL
# SQ	Sequence header	Once
# (blanks)	Sequence data	Once or more
# //	Termination line	Once; ends the entry

FT_KEYS = ['TRANSMEM', 'DNA_BIND', 'REPEAT', 'REGION', 'COMPBIAS', 'INTRAMEM', 'DOMAIN', 'CA_BIND', 'COILED', 'ZN_FING', 'NP_BIND', 'MOTIF', 'TOPO_DOM', 'HELIX', 'STRAND', 'TURN', 'ACT_SITE', 'METAL', 'BINDING', 'SITE', 'NON_STD', 'MOD_RES', 'LIPID', 'CARBOHYD', 'DISULFID', 'CROSSLNK', 'VAR_SEQ', 'VARIANT', 'MUTAGEN', 'CONFLICT', 'UNSURE', 'NON_CONS', 'NON_TER', 'INIT_MET', 'SIGNAL', 'TRANSIT', 'PROPEP', 'CHAIN', 'PEPTIDE']

FT = namedtuple('FT', ['key', 'start', 'end'])
DR = namedtuple('DR', ['rdb', 'rid', 'rinfo'])

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
    pdbid_count = defaultdict(set)
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
                # PDB ID and HELIX STRAND TURN features
                has_pdb_id = len([r for r in entry['DR'] if r.rdb == 'PDB']) > 0
                if has_pdb_id:
                    vocab['HAS_PDB_ID'].add(id)
                has_struct_ft = len([f for f in entry['FT'] if f.key in ['HELIX', 'STRAND', 'TURN']]) > 0
                if has_struct_ft:
                    vocab['HAS_STRUCT_FT'].add(id)
                # clear entry
                entry = defaultdict(list)
            elif lc == 'ID':
                count[lc] += 1
                id = line.split()[1]
                vocab[lc].add(id)
            elif lc == 'DR':
                rdb, rid, *rinfo = [a.strip() for a in line[5:].split(';') if a != '']
                entry['DR'].append(DR(rdb, rid, rinfo))
                if rdb == 'PDB':
                    vocab[rdb].add(rid)
                    pdbid_count[rid].add(id)
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
    return vocab, count, overlap, pdbid_count

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
        description='Print some statistics of the uniprot dat.gz file.')
    parser.add_argument('-d', '--dat', type=str, required=True,
        help="Path to .dat.gz file, required.")
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    dat_path = verify_input_path(args.dat)
    print(f'Processing: {dat_path.name}')

    vocab, count, overlap, pdbid_count = stats_uniprot_dat(dat_path)

    print('==COUNT==')
    for k, v in count.items():
        print(f'{k}: {v}')

    print('==VOCAB LENGTH==')
    for k, v in vocab.items():
        print(f'{k}: {len(v)}')

    print('==VOCAB==')
    print(f'''FT: {vocab['FT']}''')

    print('==OVERLAP==')
    for k in FT_KEYS:
        print(f'''{k:8} {' '.join([str(overlap[k][j]) for j in FT_KEYS])}''')

    print('==HAS_PDB_ID vs HAS_STRUCT_FT set stats==')
    print_set_stats('HAS_PDB_ID', vocab['HAS_PDB_ID'], 'HAS_STRUCT_FT', vocab['HAS_STRUCT_FT'])

    print('==Number of PDB IDs with more than one entry==')
    mep = sorted([(k,s) for k, s in pdbid_count.items() if len(s) > 1], key=lambda x: len(x[1]), reverse=True)
    print(f'''{len(mep)}: {mep[:3]}''')

    # Number of unique PDB IDs in sprot
    # Number of entries with at least one PDB ID
    # Number of entries with at least one PDB ID and has HELIX STRAND TURN features
    # Number of PDB IDs with more than one entry

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# >>> s-t
# {'MUTAGEN', 'VARIANT', 'NON_CONS', 'VAR_SEQ', 'STRAND', 'TURN', 'HELIX', 'CONFLICT'}

# GAF: uniprot_sprot.dat.gz
# //: 559077
# ID: 559077
# OX: 559077
# PE: 559077
# SQ: 559077
# ID: 559077
# Run time: 55.07 s

# GAF: uniprot_trembl.dat.gz
# //: 139694261
# ID: 139694261
# OX: 139694261
# PE: 139694261
# SQ: 139694261
# ID: 139694261
# Run time: 7264.90 s

# GAF: uniprot_sprot.dat.gz
# ==COUNT==
# ID: 559077
# OX: 559077
# PE: 559077
# CHAIN: 567012
# COMPBIAS: 59057
# SQ: 559077
# //: 559077
# TRANSMEM: 370742
# SIGNAL: 41698
# DOMAIN: 194557
# COILED: 22034
# NP_BIND: 157134
# MOTIF: 43131
# ACT_SITE: 163831
# BINDING: 405195
# INIT_MET: 17164
# LIPID: 13037
# REGION: 198127
# METAL: 381033
# ZN_FING: 30558
# DISULFID: 125136
# PROPEP: 14220
# CARBOHYD: 115929
# TOPO_DOM: 140348
# REPEAT: 105511
# NON_TER: 12403
# MOD_RES: 248764
# CONFLICT: 135285
# STRAND: 266064
# HELIX: 255508
# TURN: 61535
# VAR_SEQ: 52010
# NON_CONS: 2258
# SITE: 57940
# MUTAGEN: 70268
# CROSSLNK: 23621
# VARIANT: 96908
# UNSURE: 4455
# CA_BIND: 4177
# TRANSIT: 9098
# PEPTIDE: 11579
# DNA_BIND: 11671
# INTRAMEM: 2668
# NON_STD: 357
# ==VOCAB LENGTH==
# ID: 559077
# FTID: 559077
# FT: 39
# ==VOCAB==
# FT: {'NON_CONS', 'MOD_RES', 'BINDING', 'NON_TER', 'CARBOHYD', 'CONFLICT', 'NON_STD', 'TRANSMEM', 'DISULFID', 'TURN', 'DNA_BIND', 'REPEAT', 'VAR_SEQ', 'REGION', 'COMPBIAS', 'METAL', 'PROPEP', 'CROSSLNK', 'INTRAMEM', 'UNSURE', 'DOMAIN', 'SIGNAL', 'ACT_SITE', 'CA_BIND', 'COILED', 'PEPTIDE', 'ZN_FING', 'CHAIN', 'TRANSIT', 'HELIX', 'INIT_MET', 'MUTAGEN', 'LIPID', 'NP_BIND', 'VARIANT', 'MOTIF', 'SITE', 'TOPO_DOM', 'STRAND'}
# Run time: 65.53 s

# GAF: uniprot_trembl.dat.gz
# ==COUNT==
# ID: 139694261
# OX: 139694261
# PE: 139694261
# DOMAIN: 99615682
# SQ: 139694261
# //: 139694261
# TRANSMEM: 124352189
# NON_TER: 20194449
# METAL: 15222911
# BINDING: 19191997
# COILED: 20471047
# MOTIF: 1726499
# REGION: 6269332
# ACT_SITE: 9062027
# NP_BIND: 7873143
# SITE: 2403797
# PROPEP: 19514
# MOD_RES: 2824442
# DISULFID: 2262561
# REPEAT: 5392898
# ZN_FING: 482470
# TOPO_DOM: 344615
# UNSURE: 92944
# CROSSLNK: 39681
# INIT_MET: 59565
# DNA_BIND: 1116388
# NON_STD: 7159
# COMPBIAS: 5067
# SIGNAL: 10228844
# CHAIN: 10256807
# CA_BIND: 290582
# INTRAMEM: 1300
# LIPID: 375529
# CARBOHYD: 21573
# TRANSIT: 140
# PEPTIDE: 756
# ==VOCAB LENGTH==
# ID: 139694261
# FTID: 103799165
# FT: 31
# ==VOCAB==
# FT: {'INTRAMEM', 'ZN_FING', 'MOTIF', 'DNA_BIND', 'UNSURE', 'DISULFID', 'CHAIN', 'SITE', 'TRANSMEM', 'CA_BIND', 'REPEAT', 'INIT_MET', 'COILED', 'BINDING', 'ACT_SITE', 'REGION', 'MOD_RES', 'TRANSIT', 'COMPBIAS', 'CARBOHYD', 'LIPID', 'PEPTIDE', 'NON_STD', 'NP_BIND', 'CROSSLNK', 'METAL', 'DOMAIN', 'SIGNAL', 'TOPO_DOM', 'NON_TER', 'PROPEP'}
# Run time: 8128.38 s

# FT CHAIN - 001R_FRG3G
# FT COMPBIAS - 001R_FRG3G
# FT CHAIN - 002L_FRG3G
# FT TRANSMEM - 002L_FRG3G
# FT COMPBIAS - 002L_FRG3G
# FT SIGNAL - 003R_FRG3G
# FT TRANSMEM - 004R_FRG3G
# FT SIGNAL - 005L_IIV3
# FT DOMAIN - 006L_IIV6
# FT COILED - 006L_IIV6
# FT DOMAIN - 009L_FRG3G
# FT NP_BIND - 009L_FRG3G
# FT MOTIF - 009L_FRG3G
# FT COILED - 014R_FRG3G
# FT NP_BIND - 019R_FRG3G
# FT ACT_SITE - 019R_FRG3G
# FT BINDING - 019R_FRG3G
# FT ACT_SITE - 044L_IIV3
# FT BINDING - 044L_IIV3
# FT INIT_MET - 053R_FRG3G
# FT LIPID - 053R_FRG3G
# FT MOTIF - 055L_FRG3G
# FT REGION - 075L_FRG3G
# FT METAL - 075L_FRG3G
# FT METAL - 075L_FRG3G
# FT ZN_FING - 081R_FRG3G
# FT DISULFID - 088R_FRG3G
# FT PROPEP - 095L_IIV3
# FT ZN_FING - 095L_IIV6
# FT PROPEP - 104K_THEAN
# FT LIPID - 104K_THEAN
# FT DISULFID - 108_SOLLC
# FT CARBOHYD - 11011_ASFK5
# FT CARBOHYD - 11011_ASFM2
# FT TOPO_DOM - 1101L_ASFB7
# FT TOPO_DOM - 1101L_ASFB7
# FT REPEAT - 1101L_ASFB7
# FT REPEAT - 1101L_ASFB7
# FT REGION - 110KD_PLAKN
# FT NON_TER - 110KD_PLAKN
# FT MOD_RES - 11SB_CUCMA
# FT CONFLICT - 11SB_CUCMA
# FT CONFLICT - 11SB_CUCMA
# FT STRAND - 11SB_CUCMA
# FT STRAND - 11SB_CUCMA
# FT HELIX - 11SB_CUCMA
# FT TURN - 11SB_CUCMA
# FT HELIX - 11SB_CUCMA
# FT TURN - 11SB_CUCMA
# FT NON_TER - 12AH_CLOS4
# FT INIT_MET - 12S_PROFR
# FT MOD_RES - 14310_ARATH
# FT VAR_SEQ - 14310_ARATH
# FT VAR_SEQ - 14311_ARATH
# FT NON_CONS - 14331_PSEMZ
# FT NON_CONS - 14331_PSEMZ
# FT SITE - 14332_CAEEL
# FT MUTAGEN - 14332_CAEEL
# FT MUTAGEN - 14336_ARATH
# FT SITE - 1433B_BOVIN
# FT CROSSLNK - 1433B_BOVIN
# FT CROSSLNK - 1433B_HUMAN
# FT VARIANT - 1433B_HUMAN
# FT VARIANT - 1433G_HUMAN
# FT UNSURE - 28KD_TRIFO
# FT UNSURE - 28KD_TRIFO
# FT CA_BIND - 2AB2A_ARATH
# FT CA_BIND - 2AB2B_ARATH
# FT TRANSIT - 3CAR1_PICAB
# FT TRANSIT - 3CAR1_PICGL
# FT PEPTIDE - 3CP1_STRS9
# FT PEPTIDE - 3CP2_STRSQ
# FT DNA_BIND - 7UP1_DROME
# FT DNA_BIND - 7UP2_DROME
# FT INTRAMEM - AAAT_BOVIN
# FT INTRAMEM - AAAT_BOVIN
# FT NON_STD - BTHD_DROME
# FT NON_STD - DHGL_DROME

# >>> s-t
# {'MUTAGEN', 'VARIANT', 'NON_CONS', 'VAR_SEQ', 'STRAND', 'TURN', 'HELIX', 'CONFLICT'}

# ==OVERLAP==
# TRANSMEM 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 0 0 1 1 1
# DNA_BIND 0 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 0 1 0 1 0 1 1 0
# REPEAT   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1
# REGION   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# COMPBIAS 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# INTRAMEM 1 0 1 1 1 0 1 0 0 0 0 1 0 1 1 1 0 1 1 1 0 1 1 1 1 0 1 1 1 1 0 0 1 0 0 0 0 1 1
# DOMAIN   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1
# CA_BIND  0 0 1 1 0 0 1 1 1 0 0 0 1 1 1 1 0 1 0 0 0 1 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
# COILED   1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1
# ZN_FING  1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 0 0 1 0 1 0
# NP_BIND  1 1 1 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 0
# MOTIF    1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1
# TOPO_DOM 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# HELIX    1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1
# STRAND   1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1
# TURN     1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1
# ACT_SITE 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1
# METAL    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1
# BINDING  1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1
# SITE     1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1
# NON_STD  1 0 0 1 1 0 1 0 0 0 0 0 1 1 1 0 1 1 0 1 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 1
# MOD_RES  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# LIPID    1 0 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 0 0 1 0 0 0 1 1 1
# CARBOHYD 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1
# DISULFID 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1
# CROSSLNK 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1
# VAR_SEQ  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# VARIANT  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# MUTAGEN  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1
# CONFLICT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# UNSURE   0 0 0 1 1 0 1 1 0 0 0 0 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 0 1 0 1 1 0 1 0 1 1 1
# NON_CONS 1 0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1 0 0 0 0 1 1 1
# NON_TER  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 0 0 0 1 1 1 1 1
# INIT_MET 0 0 0 1 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 0
# SIGNAL   0 1 1 1 1 0 0 0 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 1 0 1 0 0 1 0
# TRANSIT  0 0 1 1 1 0 0 0 1 1 1 1 1 1 1 1 0 0 1 1 0 1 0 1 0 1 1 1 1 1 0 0 1 0 0 1 0 1 0
# PROPEP   1 1 1 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1
# CHAIN    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# PEPTIDE  1 0 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1

# uniprot_trembl.dat.gz
# ==OVERLAP==
# TRANSMEM 0 1 1 1 0 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 0 1 0 0 0 0 0 1 0 0 1 1 0 1 1 0
# DNA_BIND 1 1 1 1 0 0 1 1 1 0 0 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# REPEAT   1 1 1 1 0 0 1 1 1 1 1 0 1 0 0 0 1 1 1 0 0 1 0 1 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# REGION   1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 1 1 1 1 1 0 0 0 0 1 0 1 1 1 0 0 1 0
# COMPBIAS 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
# INTRAMEM 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
# DOMAIN   1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0 1 1 1 1 0 1 0
# CA_BIND  1 1 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# COILED   1 1 1 1 1 0 1 1 0 1 1 1 1 0 0 0 1 1 1 1 0 1 1 1 1 1 0 0 0 0 1 0 1 0 1 0 1 1 0
# ZN_FING  1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# NP_BIND  1 0 1 1 0 0 1 0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# MOTIF    1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# TOPO_DOM 1 1 1 1 0 1 1 0 1 0 1 1 1 0 0 0 1 1 1 1 0 1 1 1 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0
    # HELIX    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # STRAND   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # TURN     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# ACT_SITE 1 0 1 1 0 0 1 0 1 0 1 1 1 0 0 0 1 1 1 1 1 1 0 0 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0
# METAL    1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0
# BINDING  1 0 1 1 0 0 1 0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 1 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0
# SITE     1 1 0 1 0 0 1 0 1 0 1 1 1 0 0 0 1 1 1 1 0 1 0 1 1 0 0 0 0 0 1 0 1 0 1 0 1 1 1
# NON_STD  1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0
# MOD_RES  1 1 1 1 0 0 1 1 1 0 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 1 1
# LIPID    1 0 0 1 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0
# CARBOHYD 0 0 1 1 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0
# DISULFID 1 1 1 1 0 0 1 1 1 1 1 0 1 0 0 0 1 1 1 1 1 1 0 1 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0
# CROSSLNK 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 1 1 1 0 0 1 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 1 0
    # VAR_SEQ  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # VARIANT  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # MUTAGEN  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # CONFLICT 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# UNSURE   1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 1 1 0
    # NON_CONS 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# NON_TER  0 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 0 0 0 0 1 0 0 1 1 0 1 1 0
# INIT_MET 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0
# SIGNAL   1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 0 1 0 0 0 0 0 1 0 1 0 1 1 1 1 0
# TRANSIT  0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0
# PROPEP   1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# CHAIN    1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0 1 1 1 1 1 1 0
# PEPTIDE  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

# ==VOCAB LENGTH==
# ID: 559077
# FTID: 559077
# FT: 39
# PDB: 108730
# HAS_PDB_ID: 27117
# HAS_STRUCT_FT: 24484

# ==HAS_PDB_ID vs HAS_STRUCT_FT set stats==
# HAS_PDB_ID: 27117
# HAS_STRUCT_FT: 24484
# HAS_PDB_ID & HAS_STRUCT_FT: 24484
# HAS_PDB_ID | HAS_STRUCT_FT: 27117
# HAS_PDB_ID - HAS_STRUCT_FT: 2633
# HAS_STRUCT_FT - HAS_PDB_ID: 0

# ==Number of PDB IDs with more than one entry==
# 16836: [
#     ('4V6X', {'RL22_HUMAN', 'RL28_HUMAN', 'RL21_HUMAN', 'RL36A_HUMAN', 'RL11_HUMAN', 'PAIRB_HUMAN', 'RL26_HUMAN', 'RL34_HUMAN', 'RL36_HUMAN', 'RS17_HUMAN', 'RL35A_HUMAN', 'RS19_HUMAN', 'RL3_HUMAN', 'RS11_HUMAN', 'RL23A_HUMAN', 'RS3_HUMAN', 'RS16_HUMAN', 'RL30_HUMAN', 'RL4_HUMAN', 'RS18_HUMAN', 'RL41_HUMAN', 'RLA0_HUMAN', 'RS21_HUMAN', 'RL37A_HUMAN', 'RL14_HUMAN', 'RS6_HUMAN', 'RSSA_HUMAN', 'RL13A_HUMAN', 'RL10A_HUMAN', 'RS29_HUMAN', 'RS27_HUMAN', 'RL23_HUMAN', 'RL24_HUMAN', 'RL39_HUMAN', 'RS13_HUMAN', 'RL19_HUMAN', 'RS14_HUMAN', 'RS8_HUMAN', 'RL10L_HUMAN', 'RS10_HUMAN', 'RS4X_HUMAN', 'RL7A_HUMAN', 'RL18A_HUMAN', 'RS20_HUMAN', 'RS26_HUMAN', 'RL12_HUMAN', 'RS24_HUMAN', 'RL13_HUMAN', 'RL17_HUMAN', 'RS9_HUMAN', 'RS15A_HUMAN', 'RS25_HUMAN', 'RS3A_HUMAN', 'RL32_HUMAN', 'RL6_HUMAN', 'RL38_HUMAN', 'RL35_HUMAN', 'RS30_HUMAN', 'RS28_HUMAN', 'RS23_HUMAN', 'RL18_HUMAN', 'RACK1_HUMAN', 'RL7_HUMAN', 'RLA2_HUMAN', 'RS15_HUMAN', 'RS7_HUMAN', 'RS2_HUMAN', 'RL5_HUMAN', 'RL37_HUMAN', 'RL31_HUMAN', 'RS27A_HUMAN', 'RL27_HUMAN', 'RS5_HUMAN', 'RL40_HUMAN', 'EF2_HUMAN', 'RL29_HUMAN', 'RL8_HUMAN', 'RL27A_HUMAN', 'RL15_HUMAN', 'RL9_HUMAN', 'RS12_HUMAN', 'RLA1_HUMAN'}),
#     ('5DGE', {'RS22A_YEAST', 'RS30A_YEAST', 'RL22A_YEAST', 'RL37A_YEAST', 'RS4A_YEAST', 'RS31_YEAST', 'RL4A_YEAST', 'GBLP_YEAST', 'RL27A_YEAST', 'RL29_YEAST', 'RL28_YEAST', 'RL13A_YEAST', 'RL39_YEAST', 'RL3_YEAST', 'RL44A_YEAST', 'RL5_YEAST', 'RL24A_YEAST', 'RL40A_YEAST', 'RL8A_YEAST', 'RS14A_YEAST', 'RL26A_YEAST', 'RL36A_YEAST', 'RS20_YEAST', 'RL10_YEAST', 'RS26A_YEAST', 'RL32_YEAST', 'RL38_YEAST', 'RL6A_YEAST', 'RL2A_YEAST', 'RL31A_YEAST', 'RL41A_YEAST', 'RS25A_YEAST', 'RLA0_YEAST', 'RL25_YEAST', 'RL35A_YEAST', 'RL33A_YEAST', 'RS8A_YEAST', 'RL30_YEAST', 'STM1_YEAST', 'RS3A1_YEAST', 'RL14A_YEAST', 'RS13_YEAST', 'RL16A_YEAST', 'RS16A_YEAST', 'IF5A1_YEAST', 'RL19A_YEAST', 'RL43A_YEAST', 'RS9A_YEAST', 'RL12A_YEAST', 'RS28A_YEAST', 'RL7A_YEAST', 'RL20A_YEAST', 'RS7A_YEAST', 'RS10A_YEAST', 'RS6A_YEAST', 'RL11A_YEAST', 'RL18A_YEAST', 'RL34A_YEAST', 'RS23A_YEAST', 'RS29A_YEAST', 'RL21A_YEAST', 'RS2_YEAST', 'RS21A_YEAST', 'RS27A_YEAST', 'RSSA1_YEAST', 'RS3_YEAST', 'RS24A_YEAST', 'RL23A_YEAST', 'RL15A_YEAST', 'RS18A_YEAST', 'RL9A_YEAST', 'RLA1_YEAST', 'RS19A_YEAST', 'RS11A_YEAST', 'RS15_YEAST', 'RS5_YEAST', 'RS17A_YEAST', 'RL17A_YEAST', 'RS12_YEAST'}),
#     ('5JUO', {'RS22A_YEAST', 'RS30A_YEAST', 'RL22A_YEAST', 'RL37A_YEAST', 'RS4A_YEAST', 'RL1A_YEAST', 'RL4A_YEAST', 'GBLP_YEAST', 'RS31_YEAST', 'RL27A_YEAST', 'RL29_YEAST', 'RL28_YEAST', 'RL13A_YEAST', 'RL39_YEAST', 'RL3_YEAST', 'RL44A_YEAST', 'RL5_YEAST', 'RL24A_YEAST', 'RL40A_YEAST', 'RL8A_YEAST', 'RS14A_YEAST', 'RL26A_YEAST', 'RL36A_YEAST', 'RS20_YEAST', 'RL10_YEAST', 'RS26A_YEAST', 'RL32_YEAST', 'RL38_YEAST', 'RL6A_YEAST', 'RL2A_YEAST', 'RL31A_YEAST', 'RL41A_YEAST', 'RS25A_YEAST', 'RLA0_YEAST', 'RL25_YEAST', 'RL35A_YEAST', 'RL33A_YEAST', 'RS8A_YEAST', 'RL30_YEAST', 'EF2_YEAST', 'RS3A1_YEAST', 'RL14A_YEAST', 'RS13_YEAST', 'RL16A_YEAST', 'RS16A_YEAST', 'RL19A_YEAST', 'RL43A_YEAST', 'RS9A_YEAST', 'RL12A_YEAST', 'RS28A_YEAST', 'RL7A_YEAST', 'RL20A_YEAST', 'RS7A_YEAST', 'RS10A_YEAST', 'RS6A_YEAST', 'RL11A_YEAST', 'RL18A_YEAST', 'RL34A_YEAST', 'RS23A_YEAST', 'RS29A_YEAST', 'RL21A_YEAST', 'RS2_YEAST', 'RS21A_YEAST', 'RS27A_YEAST', 'RSSA1_YEAST', 'RS3_YEAST', 'RS24A_YEAST', 'RL23A_YEAST', 'RL15A_YEAST', 'RS18A_YEAST', 'RL9A_YEAST', 'RS19A_YEAST', 'RS11A_YEAST', 'RS15_YEAST', 'RS5_YEAST', 'RS17A_YEAST', 'RL17A_YEAST', 'RS12_YEAST'})]


# Processing: uniprot_trembl.dat.gz==COUNT==
# ID: 139694261
# OX: 139694261
# PE: 139694261
# DOMAIN: 99615682
# SQ: 139694261
# //: 139694261
# TRANSMEM: 124352189
# NON_TER: 20194449
# METAL: 15222911
# BINDING: 19191997
# COILED: 20471047
# MOTIF: 1726499
# REGION: 6269332
# ACT_SITE: 9062027
# NP_BIND: 7873143
# SITE: 2403797
# PROPEP: 19514
# MOD_RES: 2824442
# DISULFID: 2262561
# REPEAT: 5392898
# ZN_FING: 482470
# TOPO_DOM: 344615
# UNSURE: 92944
# CROSSLNK: 39681
# INIT_MET: 59565
# DNA_BIND: 1116388
# NON_STD: 7159
# COMPBIAS: 5067
# SIGNAL: 10228844
# CHAIN: 10256807
# CA_BIND: 290582
# INTRAMEM: 1300
# LIPID: 375529
# CARBOHYD: 21573
# TRANSIT: 140
# PEPTIDE: 756
# ==VOCAB LENGTH==
# ID: 139694261
# FTID: 103799165
# FT: 31
# PDB: 33638
# HAS_PDB_ID: 19039
# ==VOCAB==
# FT: {'TRANSIT', 'MOD_RES', 'COMPBIAS', 'METAL', 'CARBOHYD', 'MOTIF', 'NON_STD', 'UNSURE', 'SIGNAL', 'TOPO_DOM', 'PEPTIDE', 'INTRAMEM', 'CA_BIND', 'INIT_MET', 'BINDING', 'ZN_FING', 'PROPEP', 'CROSSLNK', 'TRANSMEM', 'LIPID', 'NP_BIND', 'CHAIN', 'DOMAIN', 'DISULFID', 'NON_TER', 'REPEAT', 'DNA_BIND', 'COILED', 'SITE', 'REGION', 'ACT_SITE'}
# ==OVERLAP==
# TRANSMEM 0 1 1 1 0 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 0 1 0 0 0 0 0 1 0 0 1 1 0 1 1 0
# DNA_BIND 1 1 1 1 0 0 1 1 1 0 0 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# REPEAT   1 1 1 1 0 0 1 1 1 1 1 0 1 0 0 0 1 1 1 0 0 1 0 1 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# REGION   1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 1 1 1 1 1 0 0 0 0 1 0 1 1 1 0 0 1 0
# COMPBIAS 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
# INTRAMEM 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
# DOMAIN   1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0 1 1 1 1 0 1 0
# CA_BIND  1 1 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# COILED   1 1 1 1 1 0 1 1 0 1 1 1 1 0 0 0 1 1 1 1 0 1 1 1 1 1 0 0 0 0 1 0 1 0 1 0 1 1 0
# ZN_FING  1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# NP_BIND  1 0 1 1 0 0 1 0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# MOTIF    1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# TOPO_DOM 1 1 1 1 0 1 1 0 1 0 1 1 1 0 0 0 1 1 1 1 0 1 1 1 1 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# HELIX    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# STRAND   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# TURN     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# ACT_SITE 1 0 1 1 0 0 1 0 1 0 1 1 1 0 0 0 1 1 1 1 1 1 0 0 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0
# METAL    1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0
# BINDING  1 0 1 1 0 0 1 0 1 1 1 1 1 0 0 0 1 1 1 1 0 1 0 1 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0
# SITE     1 1 0 1 0 0 1 0 1 0 1 1 1 0 0 0 1 1 1 1 0 1 0 1 1 0 0 0 0 0 1 0 1 0 1 0 1 1 1
# NON_STD  1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0
# MOD_RES  1 1 1 1 0 0 1 1 1 0 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1 1 1
# LIPID    1 0 0 1 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0
# CARBOHYD 0 0 1 1 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0
# DISULFID 1 1 1 1 0 0 1 1 1 1 1 0 1 0 0 0 1 1 1 1 1 1 0 1 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0
# CROSSLNK 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 1 1 1 0 0 1 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 1 0
# VAR_SEQ  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# VARIANT  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# MUTAGEN  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# CONFLICT 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# UNSURE   1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 1 1 0
# NON_CONS 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# NON_TER  0 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 0 0 0 0 1 0 0 1 1 0 1 1 0
# INIT_MET 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0
# SIGNAL   1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 0 1 0 0 0 0 0 1 0 1 0 1 1 1 1 0
# TRANSIT  0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0
# PROPEP   1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 1 0
# CHAIN    1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0 1 1 1 1 1 1 0
# PEPTIDE  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# ==HAS_PDB_ID vs HAS_STRUCT_FT set stats==

# HAS_PDB_ID: 19039
# HAS_STRUCT_FT: 0
# HAS_PDB_ID & HAS_STRUCT_FT: 0
# HAS_PDB_ID | HAS_STRUCT_FT: 19039
# HAS_PDB_ID - HAS_STRUCT_FT: 19039
# HAS_STRUCT_FT - HAS_PDB_ID: 0

# ==Number of PDB IDs with more than one entry==
# 1757: [('6HIV', {'Q38BP1_TRYB2', 'Q57Z82_TRYB2', 'A0A1G4HYD1_TRYEQ', 'Q587C2_TRYB2', 'Q387G3_TRYB2', 'Q38C96_TRYB2', 'Q38EP7_TRYB2', 'Q584T8_TRYB2', 'Q57ZP8_TRYB2', 'D0A8P6_TRYB9', 'Q382E8_TRYB2', 'Q57WW0_TRYB2', 'C9ZR63_TRYB9', 'C9ZPP1_TRYB9', 'C9ZK52_TRYB9', 'Q57YD4_TRYB2', 'Q38ET1_TRYB2', 'C9ZPD1_TRYB9', 'D0A4T0_TRYB9', 'Q4GZ99_TRYB2', 'Q386Q7_TRYB2', 'C9ZU82_TRYB9', 'Q585A3_TRYB2', 'D0A3P2_TRYB9', 'Q38AX6_TRYB2', 'Q57VB2_TRYB2', 'D0A1K1_TRYB9', 'D0A8I6_TRYB9', 'Q581Q0_TRYB2', 'Q587C4_TRYB2', 'Q580I0_TRYB2', 'Q57UZ6_TRYB2', 'Q583E5_TRYB2', 'C9ZUW3_TRYB9', 'D0A0S6_TRYB9', 'C9ZR91_TRYB9', 'Q385V2_TRYB2', 'C9ZJE4_TRYB9', 'Q38BW5_TRYB2',
# 'Q38DK6_TRYB2', 'Q38AS2_TRYB2', 'A0A0E3J9R6_TRYVI', 'D0A7Z9_TRYB9', 'Q383B7_TRYB2', 'Q38DP8_TRYB2', 'Q388L8_TRYB2', 'Q57U68_TRYB2', 'Q586G5_TRYB2', 'C9ZQR6_TRYB9', 'Q57WF8_TRYB2', 'Q381W9_TRYB2', 'Q388L7_TRYB2', 'Q38AM5_TRYB2', 'Q4GZ98_TRYB2', 'D0A755_TRYB9', 'Q586A2_TRYB2', 'Q57Z45_TRYB2', 'Q383M2_TRYB2', 'Q383D1_TRYB2', 'E0A3K1_LEIAM', 'A0A1G4I8T3_TRYEQ', 'Q383R4_TRYB2', 'Q383Y4_TRYB2', 'Q57UC5_TRYB2', 'C9ZPU8_TRYB9', 'Q57WW5_TRYB2', 'D0A511_TRYB9', 'D0A4P5_TRYB9', 'Q381T7_TRYB2', 'Q57WG1_TRYB2', 'Q580D5_TRYB2', 'Q580U5_TRYB2', 'Q385L8_TRYB2', 'Q57VQ9_TRYB2', 'Q38FG8_TRYB2', 'C9ZT11_TRYB9', 'Q587H8_TRYB2', 'Q57UK0_TRYB2', 'Q38AB0_TRYB2', 'Q584V5_TRYB2', 'Q388M2_TRYB2', 'Q38F25_TRYB2', 'Q389T7_TRYB2', 'C9ZSF8_TRYB9', 'C9ZT91_TRYB9', 'Q384N9_TRYB2', 'Q388R7_TRYB2', 'Q585P1_TRYB2', 'Q585I1_TRYB2', 'C9ZY77_TRYB9', 'C9ZRZ4_TRYB9', 'Q383S1_TRYB2', 'Q381N5_TRYB2', 'Q57YA9_TRYB2', 'A0A1G4HYZ0_TRYEQ', 'Q584F4_TRYB2', 'Q584U8_TRYB2', 'Q389L3_TRYB2', 'A0A1G4I0W4_TRYEQ', 'Q38CK0_TRYB2', 'D0A5V6_TRYB9', 'Q585C2_TRYB2', 'C9ZKC1_TRYB9', 'Q383G5_TRYB2', 'Q38CI0_TRYB2', 'Q383R2_TRYB2', 'C9ZSI8_TRYB9', 'Q57W62_TRYB2', 'Q580V1_TRYB2', 'Q38EM7_TRYB2', 'Q580M9_TRYB2', 'Q38DR3_TRYB2', 'C9ZSK8_TRYB9', 'Q57UJ2_TRYB2', 'Q57Y49_TRYB2', 'Q38D60_TRYB2', 'Q389K3_TRYB2', 'C9ZMA9_TRYB9', 'Q580R4_TRYB2', 'Q57YI7_TRYB2', 'Q38BS2_TRYB2', 'Q57UM6_TRYB2', 'Q387C7_TRYB2', 'Q586A6_TRYB2', 'Q586R9_TRYB2'}), ('6D90', {'G1U7M4_RABIT', 'G1TJW1_RABIT', 'G1T0C1_RABIT', 'G1TJR3_RABIT', 'G1SGX4_RABIT', 'G1SYJ6_RABIT', 'G1SKZ8_RABIT', 'G1TDB3_RABIT', 'G1SVB0_RABIT',
# 'G1TTQ5_RABIT', 'G1TPG3_RABIT', 'G1TT27_RABIT', 'G1U0Q2_RABIT', 'G1SS70_RABIT', 'G1SZ47_RABIT', 'G1TL06_RABIT', 'G1SE28_RABIT', 'G1SIZ2_RABIT', 'G1TVT6_RABIT', 'U3KPD5_RABIT', 'G1U344_RABIT', 'G1TNM3_RABIT', 'G1TRM4_RABIT', 'G1SE76_RABIT', 'G1SF08_RABIT', 'G1SK22_RABIT', 'G1SHQ2_RABIT', 'G1TIB4_RABIT', 'G1TG89_RABIT', 'G1U7L1_RABIT', 'G1U472_RABIT', 'G1TFM5_RABIT', 'B7NZQ2_RABIT', 'G1SKF7_RABIT', 'G1TU13_RABIT', 'G1TX33_RABIT', 'B7NZS8_RABIT', 'G1SGR6_RABIT', 'G1U945_RABIT', 'G1U6N2_RABIT', 'G1SY53_RABIT', 'G1SQH0_RABIT', 'G1SWM1_RABIT', 'G1SIT5_RABIT', 'G1TX70_RABIT', 'G1U3J0_RABIT', 'G1SJB4_RABIT', 'G1TDL2_RABIT', 'G1T6D1_RABIT', 'G1U437_RABIT', 'G1TPV3_RABIT', 'G1T8A2_RABIT', 'G1STW0_RABIT', 'G1SHG0_RABIT', 'G1TM82_RABIT', 'G1SMR7_RABIT', 'G1SNY0_RABIT', 'G1TXF6_RABIT', 'G1TZ76_RABIT', 'G1SVW5_RABIT', 'G1TUB8_RABIT', 'G1SV32_RABIT', 'G1TFE8_RABIT', 'G1SFR8_RABIT', 'G1TTN1_RABIT', 'G1SP51_RABIT', 'U3KNW6_RABIT', 'G1TK17_RABIT', 'G1TM55_RABIT', 'G1TTY7_RABIT', 'G1TSG1_RABIT', 'G1TN62_RABIT', 'G1T3D8_RABIT', 'G1TWL4_RABIT'}), ('6GAW', {'A0A286ZJR2_PIG', 'A0A287AY01_PIG', 'F1RY70_PIG', 'F1RG19_PIG', 'I3LTC4_PIG', 'F1S001_PIG', 'A0A287B3R8_PIG', 'F1RRN2_PIG', 'F1S9B7_PIG', 'F1RU54_PIG', 'A0A140UHW4_PIG', 'A0A287A119_PIG', 'I3LNJ0_PIG', 'I3LU08_PIG', 'A0A286ZP98_PIG', 'F1STE5_PIG', 'A0A287AWS0_PIG', 'A0A286ZU56_PIG', 'A0A287BMZ6_PIG', 'F1SG95_PIG', 'A0A286ZNB3_PIG', 'A0A0M3KL56_PIG', 'F1SVC5_PIG', 'F1RWJ0_PIG', 'I3LGM5_PIG', 'I3L9J0_PIG', 'A0A0M3KL52_PIG', 'A0A287A731_PIG', 'A0A0M3KL55_PIG', 'A0A286ZXA6_PIG', 'A0A287BP93_PIG', 'A0A286ZU68_PIG', 'A0A287AQT5_PIG', 'A0A0R4J8D6_PIG', 'F1SGC7_PIG', 'A0A286ZJ25_PIG', 'I3LR45_PIG', 'F1SHP6_PIG', 'F1RIU0_PIG', 'F1SU49_PIG', 'F1RSH0_PIG', 'F1S8U4_PIG', 'F1RI89_PIG', 'F1RHJ1_PIG', 'F1S8G3_PIG', 'F1RW03_PIG', 'I3LN63_PIG', 'F1RMQ4_PIG', 'F1SR64_PIG', 'K7GP19_PIG', 'I3LEJ9_PIG', 'I3L9H6_PIG', 'A0A0M3KL54_PIG', 'F1RUU2_PIG', 'A0A287BMP0_PIG', 'F1SU66_PIG', 'A0A286ZTW8_PIG', 'F1RQC9_PIG', 'A0A287AEW3_PIG', 'I3LS10_PIG', 'A0A287BIV6_PIG', 'F1RT79_PIG', 'A0A286ZZC6_PIG', 'A0A287ANU4_PIG', 'A0A0M3KL53_PIG', 'F1SRT0_PIG', 'K7GKS8_PIG', 'A0A287ARV8_PIG', 'F1SUS9_PIG', 'A0A286ZNK1_PIG', 'A0A286ZJJ6_PIG', 'F1RRH6_PIG'})]
# Run time: 14240.93 s
