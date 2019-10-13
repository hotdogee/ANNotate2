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
# python .\util\uniprot\compare_sprot_trembl.py --sprot D:/uniprot/uniprot-20190211/uniprot_sprot.dat.gz --trembl D:/uniprot/uniprot-20190211/uniprot_trembl.dat.gz

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

FT_KEYS = [
    'TRANSMEM', 'DNA_BIND', 'REPEAT', 'REGION', 'COMPBIAS', 'INTRAMEM',
    'DOMAIN', 'CA_BIND', 'COILED', 'ZN_FING', 'NP_BIND', 'MOTIF', 'TOPO_DOM',
    'HELIX', 'STRAND', 'TURN', 'ACT_SITE', 'METAL', 'BINDING', 'SITE',
    'NON_STD', 'MOD_RES', 'LIPID', 'CARBOHYD', 'DISULFID', 'CROSSLNK',
    'VAR_SEQ', 'VARIANT', 'MUTAGEN', 'CONFLICT', 'UNSURE', 'NON_CONS',
    'NON_TER', 'INIT_MET', 'SIGNAL', 'TRANSIT', 'PROPEP', 'CHAIN', 'PEPTIDE'
]

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
    mp_e = 0  # Molecule processing
    ptm_e = 0  # PTM
    sf_e = 0  # Structure feathers
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
                    for j in range(i + 1, len(entry['FT'])):
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
        description='Print the differences of two uniprot dat.gz files.'
    )
    parser.add_argument(
        '-s',
        '--sprot',
        type=str,
        required=True,
        help="Path to uniprot_sprot.dat.gz file, required."
    )
    parser.add_argument(
        '-t',
        '--trembl',
        type=str,
        required=True,
        help="Path to uniprot_trembl.dat.gz file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    sprot_path = verify_input_path(args.sprot)
    trembl_path = verify_input_path(args.trembl)
    print(f'Comparing: {sprot_path.name} and {trembl_path.name}')

    sprot_vocab, sprot_count, sprot_overlap = stats_uniprot_dat(sprot_path)
    trembl_vocab, trembl_count, trembl_overlap = stats_uniprot_dat(trembl_path)

    print('==ID set stats==')
    print_set_stats('sprot', sprot_vocab['ID'], 'trembl', trembl_vocab['ID'])

    print('==AC set stats==')
    print_set_stats('sprot', sprot_vocab['AC'], 'trembl', trembl_vocab['AC'])

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# Swiss-Prot (created in 1986) is a high quality manually annotated and non-redundant protein sequence database, which brings together experimental results, computed features and scientific conclusions. UniProtKB/Swiss-Prot is now the reviewed section of the UniProt Knowledgebase.
# The TrEMBL section of UniProtKB was introduced in 1996 in response to the increased dataflow resulting from genome projects. It was already recognized at that time that the traditional time- and labour-intensive manual curation process which is the hallmark of Swiss-Prot could not be broadened to encompass all available protein sequences. UniProtKB/TrEMBL contains high quality computationally analyzed records that are enriched with automatic annotation and classification. These UniProtKB/TrEMBL unreviewed entries are kept separated from the UniProtKB/Swiss-Prot manually reviewed entries so that the high quality data of the latter is not diluted in any way. Automatic processing of the data enables the records to be made available to the public quickly.

# Comparing: uniprot_sprot.dat.gz and uniprot_trembl.dat.gz
# ==ID set stats==
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
