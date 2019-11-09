import os
import re
import csv
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
from xml.etree import ElementTree as ET
from .orpha.stats_orpha_xml import parse_orpha_gene_xml, parse_orpha_phenotype_xml
from ..verify_paths import verify_input_path, verify_output_path, verify_indir_path, verify_outdir_path

# Download phenotype.hpoa: https://hpo.jax.org/app/download/annotation
# #description: HPO annotations for rare diseases [7623: OMIM; 47: DECIPHER; 3702 ORPHANET]
# #date: 2019-10-11
# #tracker: https://github.com/obophenotype/human-phenotype-ontology
# #HPO-version: http://purl.obolibrary.org/obo/hp.obo/hp/releases/2019-09-06/hp.obo.owl
# DatabaseID	DiseaseName	Qualifier	HPO_ID	Reference	Evidence	Onset	Frequency	Sex	Modifier	Aspect	Biocuration
# DECIPHER:58	Leri-Weill dyschondrostosis (LWD) - SHOX deletion		HP:0003067	DECIPHER:58	IEA					P	HPO:skoehler[2013-05-29]
# OMIM:210100	BETA-AMINOISOBUTYRIC ACID, URINARY EXCRETION OF		HP:0000007	OMIM:210100	IEA					I	HPO:iea[2009-02-17]
# ORPHA:166024	Multiple epiphyseal dysplasia, Al-Gazali type		HP:0030084	ORPHA:166024	TAS		HP:0040281			P	ORPHA:orphadata[2019-10-11]
# ORPHA:166024	Multiple epiphyseal dysplasia, Al-Gazali type		HP:0000007	ORPHA:166024	TAS					I	ORPHA:orphadata[2019-10-11]


def parse_hpoa(hpoa_path):
    Phenotype = namedtuple('Phenotype', ['hpo', 'evidence'])
    hpo_disorder_phenotype = defaultdict(set)
    with hpoa_path.open(mode='r', encoding='utf-8') as f:
        for line in f:
            # empty line
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            # parse
            tokens = line.split('\t')
            if tokens[0] == 'DatabaseID':
                continue
            disorder = tokens[0].strip()
            qualifier = tokens[2].strip()
            hpo = tokens[3].strip()
            evidence = tokens[5].strip()
            if qualifier == 'NOT':
                continue
            hpo_disorder_phenotype[disorder].add(Phenotype(hpo, evidence))
    return hpo_disorder_phenotype


# ORPHA_ALL_FREQUENCIES_genes_to_phenotype.txt
# #Format: entrez-gene-id<tab>entrez-gene-symbol<tab>HPO-Term-Name<tab>HPO-Term-ID
# 4099	MAG	Intellectual disability	HP:0001249
# 4099	MAG	Astigmatism	HP:0000483
def parse_hpo_genes_to_phenotype(path):
    gene_phenotype = defaultdict(set)
    with path.open(mode='r', encoding='utf-8') as f:
        for line in f:
            # empty line
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            # parse
            tokens = line.split('\t')
            gene_id = tokens[0].strip()
            gene_symbol = tokens[1].strip()
            hpo = tokens[3].strip()
            gene_phenotype[gene_symbol].add(hpo)
    return gene_phenotype


# ORPHA_ALL_FREQUENCIES_phenotype_to_genes.txt
# #Format: HPO-ID<tab>HPO-Name<tab>Gene-ID<tab>Gene-Name
# HP:0010708	1-5 finger syndactyly	6469	SHH
# HP:0010708	1-5 finger syndactyly	64327	LMBR1
def parse_hpo_phenotype_to_genes(path):
    gene_phenotype = defaultdict(set)
    with path.open(mode='r', encoding='utf-8') as f:
        for line in f:
            # empty line
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            # parse
            tokens = line.split('\t')
            gene_id = tokens[2].strip()
            gene_symbol = tokens[3].strip()
            hpo = tokens[0].strip()
            gene_phenotype[gene_symbol].add(hpo)
    return gene_phenotype


# mim2gene.txt
# # Copyright (c) 1966-2019 Johns Hopkins University. Use of this file adheres to the terms specified at https://omim.org/help/agreement.
# # Generated: 2019-10-31
# # This file provides links between the genes in OMIM and other gene identifiers.
# # THIS IS NOT A TABLE OF GENE-PHENOTYPE RELATIONSHIPS.
# # MIM Number	MIM Entry Type (see FAQ 1.3 at https://omim.org/help/faq)	Entrez Gene ID (NCBI)	Approved Gene Symbol (HGNC)	Ensembl Gene ID (Ensembl)
# 100050	predominantly phenotypes
# 100070	phenotype	100329167
# 100640	gene	216	ALDH1A1	ENSG00000165092
# 100650	gene/phenotype	217	ALDH2	ENSG00000111275
def parse_omim_to_genes(path):
    omim_gene = defaultdict(set)
    with path.open(mode='r', encoding='utf-8') as f:
        for line in f:
            # empty line
            line_s = line.strip()
            if len(line_s) == 0 or line_s[0] == '#':
                continue
            # parse
            tokens = line.split('\t')
            gene_id = tokens[2].strip()
            gene_symbol = tokens[3].strip()
            omim = tokens[0].strip()
            if not gene_symbol:
                continue
            omim_gene[omim].add(gene_symbol)
    return omim_gene


# morbidmap.txt
# # Copyright (c) 1966-2019 Johns Hopkins University. Use of this file adheres to the terms specified at https://omim.org/help/agreement.
# # Generated: 2019-11-08
# # See end of file for additional documentation on specific fields
# # Phenotype	Gene Symbols	MIM Number	Cyto Location
# 17,20-lyase deficiency, isolated, 202110 (3)	CYP17A1, CYP17, P450C17	609300	10q24.32
# 17-alpha-hydroxylase/17,20-lyase deficiency, 202110 (3)	CYP17A1, CYP17, P450C17	609300	10q24.32
# 2-aminoadipic 2-oxoadipic aciduria, 204750 (3)	DHTKD1, KIAA1630, AMOXAD, CMT2Q	614984	10p14
# 2-methylbutyrylglycinuria, 610006 (3)	ACADSB, SBCAD	600301	10q26.13
# 3-M syndrome 1, 273750 (3)	CUL7, 3M1	609577	6p21.1
# 3-M syndrome 2, 612921 (3)	OBSL1, KIAA0657, 3M2	610991	2q35
# 3-M syndrome 3, 614205 (3)	CCDC8, 3M3	614145	19q13.32
# {Yao syndrome}, 617321 (3)	NOD2, CARD15, IBD1, CD, YAOS, BLAUS	605956	16q12.1
# {von Hippel-Lindau syndrome, modifier of}, 193300 (3)	CCND1, PRAD1, BCL1	168461	11q13.3
# [Beta-glycopyranoside tasting], 617956 (3) {Alcohol dependence, susceptibility to}, 103780 (3)	TAS2R16, T2R16, BGLPT	604867	7q31.32
# Wilms tumor, type 1, 194070 (3)	WT1, NPHS4	607102	11p13
# Wilms tumor, type 3 (2)	WT3	194090	16q
# Wilms tumor, type 4 (2)	WT4	601363	17q12-q21
# (?:, (\d*))? \(\d\)
def parse_omim_morbidmap(path):
    parse_phenotype_mim = re.compile(r'(?:, (\d*))? \(\d\)')
    omim_gene = defaultdict(set)
    with path.open(mode='r', encoding='utf-8') as f:
        for line in f:
            # empty line
            line_s = line.strip()
            if len(line_s) == 0 or line_s[0] == '#':
                continue
            # parse
            tokens = line.split('\t')
            phenotype = tokens[0].strip()
            gene_symbols = [t.strip() for t in tokens[1].strip().split(',')]
            mims = [tokens[2].strip()] + parse_phenotype_mim.findall(phenotype)
            for m in mims:
                if not m:
                    continue
                # if m in omim_gene:
                #     print(f'==DEBUG: {m} mim already in omim_gene')
                for g in gene_symbols:
                    omim_gene[m].add(g)
    return omim_gene


# DDG2P_1_11_2019.csv
# "gene symbol","gene mim","disease name","disease mim","DDD category","allelic requirement","mutation consequence",phenotypes,"organ specificity list",pmids,panel,"prev symbols","hgnc id","gene disease pair entry date"
# HMX1,142992,"OCULOAURICULAR SYNDROME",612109,probable,biallelic,"loss of function",HP:0000007;HP:0000482;HP:0000647;HP:0007906;HP:0000568;HP:0000589;HP:0000639;HP:0000518;HP:0001104,Eye;Ear,18423520,DD,,5017,"2015-07-22 16:14:07"
# CEP290,610142,"BARDET-BIEDL SYNDROME TYPE 14",615991,confirmed,biallelic,"loss of function",HP:0000256;HP:0001829;HP:0009806;HP:0002167;HP:0001162;HP:0000639;HP:0000501;HP:0001328;HP:0001773;HP:0000483;HP:0000486;HP:0000135;HP:0000545;HP:0009466;HP:0000077;HP:0000668;HP:0001395;HP:0000218;HP:0000819;HP:0001007;HP:0001080;HP:0001156;HP:0000518;HP:0000054;HP:0000556;HP:0000137;HP:0001249;HP:0001251;HP:0001263;HP:0002370;HP:0001513;HP:0000365;HP:0002099;HP:0002141;HP:0002251;HP:0008734;HP:0000822;HP:0000678;HP:0007707;HP:0000007;HP:0001712;HP:0000510;HP:0001159;HP:0000148;HP:0000750,"Kidney Renal Tract;Brain/Cognition",,DD,,29021,"2015-07-22 16:14:07"
# for row in rows:
#     print(row)
#     count += 1
#     if count == 10:
#         break
def parse_decipher_ddg2p(path):
    fieldnames = [
        'gene symbol', 'gene mim', 'disease name', 'disease mim',
        'DDD category', 'allelic requirement', 'mutation consequence',
        'phenotypes', 'organ specificity list', 'pmids', 'panel',
        'prev symbols', 'hgnc id', 'gene disease pair entry date'
    ]
    decipher_gene_phenotype = defaultdict(set)
    with path.open(mode='r', encoding='utf-8') as f:
        rows = csv.reader(f)
        count = 0
        for row in rows:
            count += 1
            if count == 1:
                continue
            # assert length == 14
            assert (len(row) == 14), f'token length: {len(row)}, expected: 14'
            gene_symbol = row[0].strip()
            if not gene_symbol:
                continue
            phenotypes = row[7].strip()
            if not phenotypes:
                continue
            decipher_gene_phenotype[gene_symbol] |= set(
                [h.strip() for h in phenotypes.split(';')]
            )
    return decipher_gene_phenotype


def print_set_stats(n1, s1, n2, s2, unit=''):
    print(
        f'''
{n1}: {len(s1)} {unit} ({list(s1)[:5]})
{n2}: {len(s2)} {unit} ({list(s2)[:5]})
{n1} & {n2}: {len(s1 & s2)} {unit} ({list(s1 & s2)[:5]})
{n1} | {n2}: {len(s1 | s2)} {unit} ({list(s1 | s2)[:5]})
{n1} - {n2}: {len(s1 - s2)} {unit} ({list(s1 - s2)[:5]})
{n2} - {n1}: {len(s2 - s1)} {unit} ({list(s2 - s1)[:5]})
'''
    )


def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def index_uniprot_fa(path, gene_seq, whitelist):
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
    # a = '>tr|V4S029|V4S029_9ROSI Uncharacterized protein OS=Citrus clementina OX=85681 GN=CICLE_v10027472mg PE=4 SV=1'
    # b = '>tr|V4S029|V4S029_9ROSI Uncharacterized protein OS=Citrus clementina OX=85681 PE=4 SV=1'
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
    pattern = re.compile(r'^.+?GN=([^= ]+)')
    with f, tqdm(
        total=target,
        dynamic_ncols=True,
        ascii=True,
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
                    if symbol in whitelist:
                        gene_seq[symbol].add(sequence)
                    sequence = ''
                # parse header
                match = pattern.match(line)
                if match:
                    symbol = match.group(1).upper()
                else:
                    symbol = None
            else:
                sequence += line
        if sequence:
            if symbol in whitelist:
                gene_seq[symbol].add(sequence)
            sequence = ''
    return


if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description=
        'Print some statistics of the HPO phenotype.hpoa and orpha en_product6.xml file.'
    )
    parser.add_argument(
        '-a',
        '--hpoa',
        type=str,
        required=True,
        help="Path to HPO phenotype.hpoa file, required."
    )
    parser.add_argument(
        '-x6',
        '--orpha6',
        type=str,
        required=True,
        help="Path to ORPHA en_product6.xml file, required."
    )
    parser.add_argument(
        '-x4',
        '--orpha4',
        type=str,
        required=True,
        help="Path to ORPHA en_product4_HPO.xml file, required."
    )
    parser.add_argument(
        '-gp',
        '--hpog2p',
        type=str,
        required=True,
        help=
        "Path to HPO ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt file, required."
    )
    parser.add_argument(
        '-pg',
        '--hpop2g',
        type=str,
        required=True,
        help=
        "Path to HPO ALL_SOURCES_ALL_FREQUENCIES_phenotype_to_genes.txt file, required."
    )
    parser.add_argument(
        '-o',
        '--omim',
        type=str,
        required=True,
        help="Path to OMIM mim2gene.txt file, required."
    )
    parser.add_argument(
        '-om',
        '--omim_morbidmap',
        type=str,
        required=True,
        help="Path to OMIM morbidmap.txt file, required."
    )
    parser.add_argument(
        '-d',
        '--ddg2p',
        type=str,
        required=True,
        help="Path to DECIPHER DDG2P_1_11_2019.csv file, required."
    )
    parser.add_argument(
        '-f1',
        '--fa1',
        type=str,
        required=True,
        help="Path to the uniprot_trembl.fasta[.gz] file, required."
    )
    parser.add_argument(
        '-f2',
        '--fa2',
        type=str,
        required=True,
        help="Path to the uniprot_sprot.fasta[.gz] file, required."
    )
    parser.add_argument(
        '-f3',
        '--fa3',
        type=str,
        required=True,
        help="Path to the uniprot_sprot_varsplic.fasta[.gz] file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    hpoa_path = verify_input_path(args.hpoa)
    orpha6_path = verify_input_path(args.orpha6)
    orpha4_path = verify_input_path(args.orpha4)
    hpo_g2p_path = verify_input_path(args.hpog2p)
    hpo_p2g_path = verify_input_path(args.hpop2g)
    omim_path = verify_input_path(args.omim)
    omim_morbidmap_path = verify_input_path(args.omim_morbidmap)
    ddg2p_path = verify_input_path(args.ddg2p)
    fa1_path = verify_input_path(args.fa1)
    fa2_path = verify_input_path(args.fa2)
    fa3_path = verify_input_path(args.fa3)
    print(
        f'Processing: {hpoa_path.name}, {orpha6_path.name}, {orpha4_path.name}, {hpo_g2p_path.name}, {hpo_p2g_path.name}, {omim_path.name}, {ddg2p_path.name}'
    )

    hpo_disorder_phenotype = parse_hpoa(hpoa_path)
    print('==HPO: {hpoa_path.name}==')
    # Disorders: 11372
    print(f'Disorders: {len(hpo_disorder_phenotype)}')
    hpo_phenotype_annotations = [
        phenotype for disorder in hpo_disorder_phenotype
        for phenotype in hpo_disorder_phenotype[disorder]
    ]
    # Disorder to phenotype annotations: 190530
    print(
        f'Disorder to phenotype annotations: {len(hpo_phenotype_annotations)}'
    )
    print( # IEA (inferred from electronic annotation): 53279
        f' - IEA (inferred from electronic annotation): {len([a for a in hpo_phenotype_annotations if a.evidence == "IEA"])}'
    )
    print( # PCS (published clinical study): 9724
        f' - PCS (published clinical study): {len([a for a in hpo_phenotype_annotations if a.evidence == "PCS"])}'
    )
    print( # ICE (individual clinical experience): 0
        f' - ICE (individual clinical experience): {len([a for a in hpo_phenotype_annotations if a.evidence == "ICE"])}'
    )
    print( # TAS (traceable author statement): 127527
        f' - TAS (traceable author statement): {len([a for a in hpo_phenotype_annotations if a.evidence == "TAS"])}'
    )
    print( # evidence list: {'IEA', 'TAS', 'PCS'}
        f' - evidence list: {set([a.evidence for a in hpo_phenotype_annotations])}'
    )
    hpo_decipher_disorder_phenotype = set(
        [
            d.split(':')[1]
            for d in hpo_disorder_phenotype if d.split(':')[0] == 'DECIPHER'
        ]
    )
    hpo_omim_disorder_phenotype = set(
        [
            d.split(':')[1]
            for d in hpo_disorder_phenotype if d.split(':')[0] == 'OMIM'
        ]
    )
    hpo_orpha_disorder_phenotype = set(
        [
            d.split(':')[1]
            for d in hpo_disorder_phenotype if d.split(':')[0] == 'ORPHA'
        ]
    )
    print(f' - decipher disorders: {len(hpo_decipher_disorder_phenotype)}')
    print(f' - omim disorders: {len(hpo_omim_disorder_phenotype)}')
    print(f' - orpha disorders: {len(hpo_orpha_disorder_phenotype)}')
    # - decipher disorders: 47
    # - omim disorders: 7623
    # - orpha disorders: 3702

    print('')
    print('==ORPHA_GENE: {orpha6_path.name}==')
    orpha_gene = parse_orpha_gene_xml(orpha6_path)
    print(f'Disorders: {len(orpha_gene)}')
    gene_annotations = [
        gene for disorder in orpha_gene for gene in orpha_gene[disorder]
    ]
    print(f'Disorder to gene annotations: {len(gene_annotations)}')
    print(
        f' - no SwissProt reference: {len([a for a in gene_annotations if "SwissProt" not in a])}'
    )
    print(
        f' - no Ensembl reference: {len([a for a in gene_annotations if "Ensembl" not in a])}'
    )

    print('')
    print('==ORPHA_HPO: {orpha4_path.name}==')
    orpha_phenotype = parse_orpha_phenotype_xml(orpha4_path)
    print(f'Disorders: {len(orpha_phenotype)}')

    print('')
    # build gene_phenotype from hpo_disorder_phenotype and orpha_gene
    hpo_orpha_gene_phenotype = defaultdict(set)
    for orpha in orpha_gene:
        phenotypes = hpo_disorder_phenotype.get(f'ORPHA:{orpha}')
        if not phenotypes:
            continue
        phenotypes_set = set([p.hpo for p in phenotypes])
        for gene in orpha_gene[orpha]:
            gene_symbol = gene['symbol']
            hpo_orpha_gene_phenotype[gene_symbol] |= phenotypes_set

    # build gene_phenotype from orpha_phenotype and orpha_gene
    orpha_gene_phenotype = defaultdict(set)
    for orpha in orpha_gene:
        phenotypes = orpha_phenotype.get(orpha)
        if not phenotypes:
            continue
        for gene in orpha_gene[orpha]:
            gene_symbol = gene['symbol']
            orpha_gene_phenotype[gene_symbol] |= phenotypes

    print('')
    # print('==OMIM: {omim_path.name}==')
    # omim_gene = parse_omim_to_genes(omim_path)
    # print(f'Disorders: {len(omim_gene)}') # 16104
    print('==OMIM: {omim_morbidmap_path.name}==')
    omim_gene = parse_omim_morbidmap(omim_morbidmap_path)
    print(f'Disorders: {len(omim_gene)}')  # 10905
    print(
        f'Genes: {len(set([g for o in omim_gene for g in omim_gene[o]]))}'
    )  # 14280
    # build gene_phenotype from hpo_disorder_phenotype and omim_gene
    count = 0
    hpo_omim_gene_phenotype = defaultdict(set)
    for omim in omim_gene:
        phenotypes = hpo_disorder_phenotype.get(f'OMIM:{omim}')
        if not phenotypes:
            continue
        count += 1
        phenotypes_set = set([p.hpo for p in phenotypes])
        for gene_symbol in omim_gene[omim]:
            hpo_omim_gene_phenotype[gene_symbol] |= phenotypes_set
    # print(f'count {count}')

    hpo_g2p_gene_phenotype = parse_hpo_genes_to_phenotype(hpo_g2p_path)
    hpo_p2g_gene_phenotype = parse_hpo_phenotype_to_genes(hpo_p2g_path)
    decipher_gene_phenotype = parse_decipher_ddg2p(ddg2p_path)
    print('==HPO_G2P==')
    print(f'Genes: {len(hpo_g2p_gene_phenotype)}')  # 4231
    print('==HPO_P2G==')
    print(f'Genes: {len(hpo_p2g_gene_phenotype)}')  # 4231
    print('==HPO_ORPHA==')
    print(f'Genes: {len(hpo_orpha_gene_phenotype)}')  # 2730
    print('==ORPHA==')
    print(f'Genes: {len(orpha_gene_phenotype)}')  # 2763
    print('==HPO_OMIM==')
    print(f'Genes: {len(hpo_omim_gene_phenotype)}')  # 23 -> 13003
    print('==DECIPHER==')
    print(f'Genes: {len(decipher_gene_phenotype)}')  # 1370
    # print_set_stats(
    #     'ORPHA|HPO_ORPHA|HPO_G2P',
    #     set(
    #         orpha_gene_phenotype
    #     ) | set(
    #         hpo_orpha_gene_phenotype
    #     ) | set(
    #         hpo_g2p_gene_phenotype
    #     ), 'DECIPHER',
    #     set(
    #         decipher_gene_phenotype
    #     )
    # )
    hpo_omim_gene_phenotype_set = set(
        [
            (g, p) for g in hpo_omim_gene_phenotype
            for p in hpo_omim_gene_phenotype[g]
        ]
    )
    hpo_g2p_gene_phenotype_set = set(
        [
            (g, p) for g in hpo_g2p_gene_phenotype
            for p in hpo_g2p_gene_phenotype[g]
        ]
    )
    orpha_gene_phenotype_set = set(
        [(g, p) for g in orpha_gene_phenotype for p in orpha_gene_phenotype[g]]
    )
    hpo_orpha_gene_phenotype_set = set(
        [
            (g, p) for g in hpo_orpha_gene_phenotype
            for p in hpo_orpha_gene_phenotype[g]
        ]
    )
    decipher_gene_phenotype_set = set(
        [
            (g, p) for g in decipher_gene_phenotype
            for p in decipher_gene_phenotype[g]
        ]
    )
    print_set_stats(
        'HPO_OMIM', hpo_omim_gene_phenotype_set, 'HPO_G2P',
        hpo_g2p_gene_phenotype_set
    )
    gene_phenotype_set = hpo_omim_gene_phenotype_set | hpo_g2p_gene_phenotype_set
    print_set_stats(
        'HPO_OMIM|HPO_G2P', gene_phenotype_set, 'ORPHA',
        orpha_gene_phenotype_set
    )
    gene_phenotype_set |= orpha_gene_phenotype_set
    print_set_stats(
        'HPO_OMIM|HPO_G2P|ORPHA', gene_phenotype_set, 'HPO_ORPHA',
        hpo_orpha_gene_phenotype_set
    )
    gene_phenotype_set |= hpo_orpha_gene_phenotype_set
    print_set_stats(
        'HPO_OMIM|HPO_G2P|ORPHA|HPO_ORPHA', gene_phenotype_set, 'DECIPHER',
        decipher_gene_phenotype_set
    )
    gene_phenotype_set |= decipher_gene_phenotype_set

    # total number of genes with phenotype annotations
    gene_phenotype = defaultdict(set)
    for g, p in gene_phenotype_set:
        gene_phenotype[g.upper()].add(p.upper())
    print(f'gene_phenotype ({len(gene_phenotype_set)})')
    print(f' - Genes ({len(gene_phenotype)})')
    print(
        f' - Phenotypes ({len(set([p for g in gene_phenotype for p in gene_phenotype[g]]))})'
    )
    # print(f'{set(gene_phenotype)}')

    # print('')
    # print('==SET STATS==')
    # print_set_stats(
    #     'HPO_ORPHA', hpo_orpha_disorder_phenotype, 'ORPHA_GENE', set(orpha_gene)
    # )
    # print_set_stats(
    #     'HPO_ORPHA', hpo_orpha_disorder_phenotype, 'ORPHA_HPO', set(orpha_phenotype)
    # )
    # print_set_stats('HPO_OMIM', hpo_omim_disorder_phenotype, 'OMIM_GENE', set(omim_gene))
    # print_set_stats(
    #     'HPO_G2P',
    #     set(
    #         [
    #             (g, p) for g in hpo_g2p_gene_phenotype
    #             for p in hpo_g2p_gene_phenotype[g]
    #         ]
    #     ), 'HPO_ORPHA_P2G',
    #     set(
    #         [
    #             (g, p) for g in hpo_p2g_gene_phenotype
    #             for p in hpo_p2g_gene_phenotype[g]
    #         ]
    #     )
    # )
    # print_set_stats(
    #     'HPO_ORPHA',
    #     set(
    #         [
    #             (g, p) for g in hpo_orpha_gene_phenotype
    #             for p in hpo_orpha_gene_phenotype[g]
    #         ]
    #     ), 'ORPHA',
    #     set(
    #         [
    #             (g, p) for g in orpha_gene_phenotype
    #             for p in orpha_gene_phenotype[g]
    #         ]
    #     )
    # )
    # print_set_stats(
    #     'HPO_ORPHA',
    #     set(
    #         [
    #             (g, p) for g in hpo_orpha_gene_phenotype
    #             for p in hpo_orpha_gene_phenotype[g]
    #         ]
    #     ), 'HPO_G2P',
    #     set(
    #         [
    #             (g, p) for g in hpo_g2p_gene_phenotype
    #             for p in hpo_g2p_gene_phenotype[g]
    #         ]
    #     )
    # )
    # print_set_stats(
    #     'ORPHA',
    #     set(
    #         [
    #             (g, p) for g in orpha_gene_phenotype
    #             for p in orpha_gene_phenotype[g]
    #         ]
    #     ), 'HPO_G2P',
    #     set(
    #         [
    #             (g, p) for g in hpo_g2p_gene_phenotype
    #             for p in hpo_g2p_gene_phenotype[g]
    #         ]
    #     )
    # )
    # print_set_stats(
    #     'ORPHA|HPO_ORPHA',
    #     set(
    #         [
    #             (g, p) for g in orpha_gene_phenotype
    #             for p in orpha_gene_phenotype[g]
    #         ]
    #     ) | set(
    #         [
    #             (g, p) for g in hpo_orpha_gene_phenotype
    #             for p in hpo_orpha_gene_phenotype[g]
    #         ]
    #     ), 'HPO_G2P',
    #     set(
    #         [
    #             (g, p) for g in hpo_g2p_gene_phenotype
    #             for p in hpo_g2p_gene_phenotype[g]
    #         ]
    #     )
    # )
    # print_set_stats(
    #     'ORPHA|HPO_ORPHA|HPO_G2P',
    #     set(
    #         [
    #             (g, p) for g in orpha_gene_phenotype
    #             for p in orpha_gene_phenotype[g]
    #         ]
    #     ) | set(
    #         [
    #             (g, p) for g in hpo_orpha_gene_phenotype
    #             for p in hpo_orpha_gene_phenotype[g]
    #         ]
    #     ) | set(
    #         [
    #             (g, p) for g in hpo_g2p_gene_phenotype
    #             for p in hpo_g2p_gene_phenotype[g]
    #         ]
    #     ), 'DECIPHER',
    #     set(
    #         [
    #             (g, p) for g in decipher_gene_phenotype
    #             for p in decipher_gene_phenotype[g]
    #         ]
    #     )
    # )

    # read sequences
    gene_seq = defaultdict(set)
    index_uniprot_fa(fa3_path, gene_seq, gene_phenotype)
    # make a gene list sorted by number of sequences from most to least
    gene_list = sorted(
        gene_seq, key=lambda k: (len(gene_seq[k]), k), reverse=True
    )
    print(f'{fa3_path.name}')
    print(f'Genes:')
    print(f' - Total: {len(gene_seq)}')
    print(f'Seqs:')
    print(f' - Total: {len(set([s for g in gene_seq for s in gene_seq[g]]))}')
    print(
        f' - Max, Median, Min: {len(gene_seq[gene_list[0]])}, {len(gene_seq[gene_list[len(gene_list)//2]])}, {len(gene_seq[gene_list[-1]])}'
    )
    index_uniprot_fa(fa2_path, gene_seq, gene_phenotype)
    print(f'+ {fa2_path.name}')
    print(f'Genes:')
    print(f' - Total: {len(gene_seq)}')
    print(f'Seqs:')
    print(f' - Total: {len(set([s for g in gene_seq for s in gene_seq[g]]))}')
    print(
        f' - Max, Median, Min: {len(gene_seq[gene_list[0]])}, {len(gene_seq[gene_list[len(gene_list)//2]])}, {len(gene_seq[gene_list[-1]])}'
    )
    index_uniprot_fa(fa1_path, gene_seq, gene_phenotype)
    print(f'+ {fa1_path.name}')
    print(f'Genes:')
    print(f' - Total: {len(gene_seq)}')
    print(f'Seqs:')
    print(f' - Total: {len(set([s for g in gene_seq for s in gene_seq[g]]))}')
    print(
        f' - Max, Median, Min: {len(gene_seq[gene_list[0]])}, {len(gene_seq[gene_list[len(gene_list)//2]])}, {len(gene_seq[gene_list[-1]])}'
    )

    # list some genes without a protein sequence
    genes_without_sequence = set(gene_phenotype) - set(gene_seq)
    print(
        f'Genes without a sequence ({len(genes_without_sequence)}): {genes_without_sequence}'
    )

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# windows
# python -m util.hpo.stats_hpoa --hpoa E:\hpo\hpo-20191011\phenotype.hpoa --orpha6 E:\hpo\orpha-20191101\en_product6.xml --orpha4 E:\hpo\orpha-20191101\en_product4_HPO.xml --hpog2p E:\hpo\hpo-20191011\annotation\ORPHA_ALL_FREQUENCIES_genes_to_phenotype.txt --hpop2g E:\hpo\hpo-20191011\annotation\ORPHA_ALL_FREQUENCIES_phenotype_to_genes.txt --omim E:\hpo\omim-20191101\mim2gene.txt --ddg2p E:\hpo\decipher-20191101\DDG2P_1_11_2019.csv

# python -m util.hpo.stats_hpoa --hpoa E:\hpo\hpo-20191011\phenotype.hpoa --orpha6 E:\hpo\orpha-20191101\en_product6.xml --orpha4 E:\hpo\orpha-20191101\en_product4_HPO.xml --hpog2p E:\hpo\hpo-20191011\annotation\ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt --hpop2g E:\hpo\hpo-20191011\annotation\ALL_SOURCES_ALL_FREQUENCIES_phenotype_to_genes.txt --omim E:\hpo\omim-20191101\mim2gene.txt --ddg2p E:\hpo\decipher-20191101\DDG2P_1_11_2019.csv

# python -m util.hpo.stats_hpoa --hpoa E:\hpo\hpo-20191011\phenotype.hpoa --orpha6 E:\hpo\orpha-20191101\en_product6.xml --orpha4 E:\hpo\orpha-20191101\en_product4_HPO.xml --hpog2p E:\hpo\hpo-20191011\annotation\ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt --hpop2g E:\hpo\hpo-20191011\annotation\ALL_SOURCES_ALL_FREQUENCIES_phenotype_to_genes.txt --omim E:\hpo\omim-20191101\mim2gene.txt --ddg2p E:\hpo\decipher-20191101\DDG2P_1_11_2019.csv --fa1 F:\uniprot\uniprot-20191015\uniprot_trembl.fasta --fa2 F:\uniprot\uniprot-20191015\uniprot_sprot.fasta --fa3 F:\uniprot\uniprot-20191015\uniprot_sprot_varsplic.fasta --omim_morbidmap E:\hpo\omim-20191101\morbidmap.txt

# Use ORPHA|HPO_ORPHA|HPO_G2P for Orpha dataset

# Processing: phenotype.hpoa, en_product6.xml, en_product4_HPO.xml, ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt, ALL_SOURCES_ALL_FREQUENCIES_phenotype_to_genes.txt, mim2gene.txt, DDG2P_1_11_2019.csv
# ==HPO==
# Disorders: 11372
# Disorder to phenotype annotations: 190530
#  - IEA (inferred from electronic annotation): 53279
#  - PCS (published clinical study): 9724
#  - ICE (individual clinical experience): 0
#  - TAS (traceable author statement): 127527
#  - evidence list: {'PCS', 'IEA', 'TAS'}
#  - decipher disorders: 47
#  - omim disorders: 7623
#  - orpha disorders: 3702

# ==ORPHA_GENE==
# Disorders: 3792
# Disorder to gene annotations: 7537
#  - no SwissProt reference: 127
#  - no Ensembl reference: 50

# ==ORPHA_HPO==
# Disorders: 3771

# ==OMIM==
# Disorders: 16104

# ==HPO_G2P==
# Genes: 4231
# ==HPO_ORPHA_P2G==
# Genes: 4231
# ==HPO_ORPHA==
# Genes: 2730
# ==ORPHA==
# Genes: 2763
# ==OMIM==
# Genes: 23
# ==DECIPHER==
# Genes: 1370

# ORPHA|HPO_ORPHA|HPO_G2P: 4313  (['CAV3', 'FANCE', 'ZDHHC15', 'PTPN1', 'TINF2'])
# DECIPHER: 1370  (['FANCE', 'ZDHHC15', 'MTO1', 'UROC1', 'HPRT1'])
# ORPHA|HPO_ORPHA|HPO_G2P & DECIPHER: 1326  (['FANCE', 'ZDHHC15', 'MTO1', 'UROC1', 'HPRT1'])
# ORPHA|HPO_ORPHA|HPO_G2P | DECIPHER: 4357  (['CAV3', 'FANCE', 'ZDHHC15', 'PTPN1', 'TINF2'])
# ORPHA|HPO_ORPHA|HPO_G2P - DECIPHER: 2987  (['IL2RB', 'CAV3', 'PTPN1', 'TINF2', 'HLA-DQB1'])
# DECIPHER - ORPHA|HPO_ORPHA|HPO_G2P: 44  (['NRXN3', 'C8orf37', 'EPRS', 'APOPT1', 'MED13'])

# ==SET STATS==

# HPO_ORPHA: 3702  (['281090', '261476', '99750', '2698', '254361'])
# ORPHA_GENE: 3792  (['79246', '228179', '281090', '2698', '254361'])
# HPO_ORPHA & ORPHA_GENE: 2008  (['281090', '2698', '254361', '319199', '220295'])
# HPO_ORPHA | ORPHA_GENE: 5486  (['79246', '281090', '261476', '2698', '254361'])
# HPO_ORPHA - ORPHA_GENE: 1694  (['261476', '99750', '293987', '436003', '85319'])
# ORPHA_GENE - HPO_ORPHA: 1784  (['79246', '228179', '97234', '276603', '157716'])

# HPO_ORPHA: 3702  (['281090', '261476', '99750', '2698', '254361'])
# ORPHA_HPO: 3771  (['281090', '261476', '99750', '2698', '254361'])
# HPO_ORPHA & ORPHA_HPO: 3702  (['281090', '261476', '99750', '2698', '254361'])
# HPO_ORPHA | ORPHA_HPO: 3771  (['281090', '261476', '2698', '254361', '319199'])
# HPO_ORPHA - ORPHA_HPO: 0  ([])
# ORPHA_HPO - HPO_ORPHA: 69  (['357001', '247262', '331206', '95455', '275555'])

# HPO_OMIM: 7623  (['300310', '120433', '601216', '616562', '136580'])
# OMIM_GENE: 16104  (['607729', '612377', '604258', '605096', '616254'])
# HPO_OMIM & OMIM_GENE: 23  (['152200', '151430', '400003', '610271', '240400'])
# HPO_OMIM | OMIM_GENE: 23704  (['300310', '607729', '612377', '605096', '618338'])
# HPO_OMIM - OMIM_GENE: 7600  (['300310', '120433', '601216', '616562', '136580'])
# OMIM_GENE - HPO_OMIM: 16081  (['607729', '612377', '604258', '605096', '616254'])

# HPO_G2P: 161003  ([('MED25', 'HP:0001263'), ('PRKRA', 'HP:0002451'), ('ARCN1', 'HP:0000218'), ('FOXG1', 'HP:0100490'), ('SCN4A', 'HP:0002203')])
# HPO_ORPHA_P2G: 552814  ([('MED25', 'HP:0001263'), ('STAT1', 'HP:0004348'), ('SLC34A3', 'HP:0025031'), ('DPF2', 'HP:0003549'), ('ARCN1', 'HP:0000218')])
# HPO_G2P & HPO_ORPHA_P2G: 161003  ([('MED25', 'HP:0001263'), ('PRKRA', 'HP:0002451'), ('ARCN1', 'HP:0000218'), ('FOXG1', 'HP:0100490'), ('SCN4A', 'HP:0002203')])
# HPO_G2P | HPO_ORPHA_P2G: 552814  ([('MED25', 'HP:0001263'), ('SLC34A3', 'HP:0025031'), ('ARCN1', 'HP:0000218'), ('COX3', 'HP:0025142'), ('SC5D', 'HP:0008428')])
# HPO_G2P - HPO_ORPHA_P2G: 0  ([])
# HPO_ORPHA_P2G - HPO_G2P: 391811  ([('STAT1', 'HP:0004348'), ('SLC34A3', 'HP:0025031'), ('DPF2', 'HP:0003549'), ('COX3', 'HP:0025142'), ('SC5D', 'HP:0008428')])

# HPO_ORPHA: 111511  ([('MED25', 'HP:0001263'), ('KRT14', 'HP:0001053'), ('PRKRA', 'HP:0002451'), ('SCN4A', 'HP:0002203'), ('FOXG1', 'HP:0100490')])
# ORPHA: 109591  ([('MED25', 'HP:0001263'), ('KRT14', 'HP:0001053'), ('PRKRA', 'HP:0002451'), ('SCN4A', 'HP:0002203'), ('FOXG1', 'HP:0100490')])
# HPO_ORPHA & ORPHA: 106750  ([('MED25', 'HP:0001263'), ('KRT14', 'HP:0001053'), ('PRKRA', 'HP:0002451'), ('SCN4A', 'HP:0002203'), ('FOXG1', 'HP:0100490')])
# HPO_ORPHA | ORPHA: 114352  ([('MED25', 'HP:0001263'), ('PRKRA', 'HP:0002451'), ('SCN4A', 'HP:0002203'), ('FOXG1', 'HP:0100490'), ('AP4M1', 'HP:0001263')])
# HPO_ORPHA - ORPHA: 4761  ([('EDAR', 'HP:0000007'), ('GDF5', 'HP:0000006'), ('SEC24D', 'HP:0000006'), ('DISC1', 'HP:0000007'), ('DPP9', 'HP:0001426')])
# ORPHA - HPO_ORPHA: 2841  ([('KCNQ1OT1', 'HP:0002564'), ('PIK3CA', 'HP:0100843'), ('PHKG2', 'HP:0001395'), ('KMT2A', 'HP:0007930'), ('PHKA2', 'HP:0002240')])

# HPO_ORPHA: 111511  ([('MED25', 'HP:0001263'), ('KRT14', 'HP:0001053'), ('PRKRA', 'HP:0002451'), ('SCN4A', 'HP:0002203'), ('FOXG1', 'HP:0100490')])
# HPO_G2P: 161003  ([('MED25', 'HP:0001263'), ('PRKRA', 'HP:0002451'), ('ARCN1', 'HP:0000218'), ('FOXG1', 'HP:0100490'), ('SCN4A', 'HP:0002203')])
# HPO_ORPHA & HPO_G2P: 98315  ([('MED25', 'HP:0001263'), ('KRT14', 'HP:0001053'), ('PRKRA', 'HP:0002451'), ('SCN4A', 'HP:0002203'), ('FOXG1', 'HP:0100490')])
# HPO_ORPHA | HPO_G2P: 174199  ([('MED25', 'HP:0001263'), ('ARCN1', 'HP:0000218'), ('SCN4A', 'HP:0002203'), ('IL10', 'HP:0001289'), ('VPS13C', 'HP:0001268')])
# HPO_ORPHA - HPO_G2P: 13196  ([('POLG2', 'HP:0001251'), ('PUF60', 'HP:0040019'), ('MT-ATP6', 'HP:0001285'), ('MT-ND4', 'HP:0002076'), ('SUFU', 'HP:0011442')])
# HPO_G2P - HPO_ORPHA: 62688  ([('CANT1', 'HP:0004233'), ('PTS', 'HP:0000737'), ('MATN3', 'HP:0002983'), ('CRYAA', 'HP:0000568'), ('ALDH18A1', 'HP:0001328')])

# ORPHA: 109591  ([('MED25', 'HP:0001263'), ('KRT14', 'HP:0001053'), ('PRKRA', 'HP:0002451'), ('SCN4A', 'HP:0002203'), ('FOXG1', 'HP:0100490')])
# HPO_G2P: 161003  ([('MED25', 'HP:0001263'), ('PRKRA', 'HP:0002451'), ('ARCN1', 'HP:0000218'), ('FOXG1', 'HP:0100490'), ('SCN4A', 'HP:0002203')])
# ORPHA & HPO_G2P: 95795  ([('MED25', 'HP:0001263'), ('KRT14', 'HP:0001053'), ('PRKRA', 'HP:0002451'), ('SCN4A', 'HP:0002203'), ('FOXG1', 'HP:0100490')])
# ORPHA | HPO_G2P: 174799  ([('MED25', 'HP:0001263'), ('ARCN1', 'HP:0000218'), ('SCN4A', 'HP:0002203'), ('IL10', 'HP:0001289'), ('VPS13C', 'HP:0001268')])
# ORPHA - HPO_G2P: 13796  ([('PIK3CA', 'HP:0100843'), ('POLG2', 'HP:0001251'), ('PHKG2', 'HP:0001395'), ('PUF60', 'HP:0040019'), ('MT-ATP6', 'HP:0001285')])
# HPO_G2P - ORPHA: 65208  ([('CANT1', 'HP:0004233'), ('PTS', 'HP:0000737'), ('MATN3', 'HP:0002983'), ('CRYAA', 'HP:0000568'), ('ALDH18A1', 'HP:0001328')])

# ORPHA|HPO_ORPHA: 114352  ([('MED25', 'HP:0001263'), ('PRKRA', 'HP:0002451'), ('SCN4A', 'HP:0002203'), ('FOXG1', 'HP:0100490'), ('AP4M1', 'HP:0001263')])
# HPO_G2P: 161003  ([('MED25', 'HP:0001263'), ('PRKRA', 'HP:0002451'), ('ARCN1', 'HP:0000218'), ('FOXG1', 'HP:0100490'), ('SCN4A', 'HP:0002203')])
# ORPHA|HPO_ORPHA & HPO_G2P: 98663  ([('MED25', 'HP:0001263'), ('KRT14', 'HP:0001053'), ('PRKRA', 'HP:0002451'), ('SCN4A', 'HP:0002203'), ('FOXG1', 'HP:0100490')])
# ORPHA|HPO_ORPHA | HPO_G2P: 176692  ([('MED25', 'HP:0001263'), ('ARCN1', 'HP:0000218'), ('SCN4A', 'HP:0002203'), ('IL10', 'HP:0001289'), ('VPS13C', 'HP:0001268')])
# ORPHA|HPO_ORPHA - HPO_G2P: 15689  ([('PHKG2', 'HP:0001395'), ('PIK3CA', 'HP:0100843'), ('POLG2', 'HP:0001251'), ('PUF60', 'HP:0040019'), ('MT-ATP6', 'HP:0001285')])
# HPO_G2P - ORPHA|HPO_ORPHA: 62340  ([('CANT1', 'HP:0004233'), ('PTS', 'HP:0000737'), ('MATN3', 'HP:0002983'), ('CRYAA', 'HP:0000568'), ('ALDH18A1', 'HP:0001328')])

# ORPHA|HPO_ORPHA|HPO_G2P: 176692  ([('MED25', 'HP:0001263'), ('ARCN1', 'HP:0000218'), ('SCN4A', 'HP:0002203'), ('IL10', 'HP:0001289'), ('VPS13C', 'HP:0001268')])
# DECIPHER: 28710  ([('BMP4', 'HP:0000202'), ('PTS', 'HP:0000737'), ('ALDH18A1', 'HP:0001328'), ('RAPSN', 'HP:0001260'), ('SRD5A3', 'HP:0000007')])
# ORPHA|HPO_ORPHA|HPO_G2P & DECIPHER: 24992  ([('BMP4', 'HP:0000202'), ('PTS', 'HP:0000737'), ('ALDH18A1', 'HP:0001328'), ('SRD5A3', 'HP:0000007'), ('FTO', 'HP:0001276')])
# ORPHA|HPO_ORPHA|HPO_G2P | DECIPHER: 180410  ([('MED25', 'HP:0001263'), ('PRKRA', 'HP:0002451'), ('ARCN1', 'HP:0000218'), ('SCN4A', 'HP:0002203'), ('FOXG1', 'HP:0100490')])
# ORPHA|HPO_ORPHA|HPO_G2P - DECIPHER: 151700  ([('MED25', 'HP:0001263'), ('PRKRA', 'HP:0002451'), ('ARCN1', 'HP:0000218'), ('SCN4A', 'HP:0002203'), ('FOXG1', 'HP:0100490')])
# DECIPHER - ORPHA|HPO_ORPHA|HPO_G2P: 3718  ([('GJA1', 'HP:0006482'), ('SMARCA4', 'HP:0003083'), ('GJA1', 'HP:0002645'), ('TTC8', 'HP:0001773'), ('RPGRIP1L', 'HP:0000272')])

# gene_phenotype Genes: 4357
# uniprot_sprot_varsplic.fasta
# Genes:
#  - Total: 2700
# Seqs:
#  - Total: 6836
#  - Max, Median, Min: 40, 2, 1
# uniprot_sprot.fasta: 100%|###########################################################################################################################################################################################################################################################################################################################| 277017792/277017792 [00:07<00:00, 38662486.39bytes/s]
# + uniprot_sprot.fasta
# Genes:
#  - Total: 4235
# Seqs:
#  - Total: 23024
#  - Max, Median, Min: 44, 3, 2
# uniprot_trembl.fasta.gz: 100%|###################################################################################################################################################################################################################################################################################################################| 80312776613/80312776613 [49:11<00:00, 27209386.06bytes/s]
# + uniprot_trembl.fasta.gz
# Genes:
#  - Total: 4248
# Seqs:
#  - Total: 1028942
#  - Max, Median, Min: 393, 230, 107
# Genes without a sequence (109): {'TRNN', 'HYMAI', 'DUX4L1', 'TRNS1', 'MT-TV', 'SPG16', 'GNAS-AS1', 'KCNQ1OT1', 'MACROH2A1', 'USH1K', 'MT-TN', 'HELLPAR', 'CXORF56', 'DYT13', 'MT-TK', 'USH1E', 'TRNS2', 'MT-TT', 'PWRN1', 'XIST', 'GARS1', 'USH1H', 'MT-TF', 'SARS1', 'SNORD116-1', 'SPG37', 'FRA16E', 'MT-TW', 'TRNC', 'SCA30', 'IARS1', 'AARS1', 'SPG34', 'SLC7A2-IT1', 'SPG14', 'PWAR1', 'TRNQ', 'SCA37', 'MT-TS2', 'SNORD115-1', 'HBB-LCR', 'EPRS1', 'KIFBP', 'MIR184', 'TRNL2', 'OPA2', 'C12ORF57', 'RNU4ATAC', 'MKRN3-AS1', 'H1-4', 'MT-TP', 'H19-ICR', 'TRNK', 'SCA20', 'SNORD118', 'TRNW', 'DARS1', 'TRNI', 'MT-TE', 'SPG25', 'RARS1', 'SPG23', 'TRNE', 'MIR96', 'C12ORF4', 'MT-TQ', 'MIR204', 'SCA25', 'SPG41', 'SPG29', 'TRNT', 'IL12A-AS1', 'DISC2', 'MT-TL1', 'IPW', 'FMR3', 'SPG36', 'SCA32', 'C9ORF72', 'MT-TH', 'MARS1', 'RNU12', 'DYT17', 'WHCR', 'STING1', 'YARS1', 'TRNV', 'KARS1', 'GINGF2', 'TRNF', 'DYT15', 'LARS1', 'ARSL', 'SPG38', 'SPG19', 'DYT21', 'TRNP', 'SPG32', 'SPG27', 'QARS1', 'MT-TS1', 'TRNL1', 'RMRP', 'C11ORF95', 'MT-TL2', 'HARS1', 'ADSS1', 'SPG24', 'C15ORF41'}
# Run time: 2968.80 s
