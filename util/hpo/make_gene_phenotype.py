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


if __name__ == "__main__":
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
        '-o',
        '--out',
        type=str,
        required=True,
        help="Path to gene_phenotype.json file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    hpoa_path = verify_input_path(args.hpoa)
    orpha6_path = verify_input_path(args.orpha6)
    orpha4_path = verify_input_path(args.orpha4)
    hpo_g2p_path = verify_input_path(args.hpog2p)
    omim_morbidmap_path = verify_input_path(args.omim_morbidmap)
    ddg2p_path = verify_input_path(args.ddg2p)
    print(
        f'Processing: {hpoa_path.name}, {orpha6_path.name}, {orpha4_path.name}, {hpo_g2p_path.name}, {omim_morbidmap_path.name}, {ddg2p_path.name}'
    )

    hpo_disorder_phenotype = parse_hpoa(hpoa_path)
    print('==HPO: {hpoa_path.name}==')
    # Disorders: 11372
    print(f'Disorders: {len(hpo_disorder_phenotype)}')

    print('')
    print('==ORPHA_GENE: {orpha6_path.name}==')
    orpha_gene = parse_orpha_gene_xml(orpha6_path)
    print(f'Disorders: {len(orpha_gene)}')
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

    print('')
    print('==ORPHA_HPO: {orpha4_path.name}==')
    orpha_phenotype = parse_orpha_phenotype_xml(orpha4_path)
    print(f'Disorders: {len(orpha_phenotype)}')
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

    print('==HPO_OMIM==')
    print(f'Genes: {len(hpo_omim_gene_phenotype)}')  # 23 -> 13003
    print('==HPO_G2P==')
    hpo_g2p_gene_phenotype = parse_hpo_genes_to_phenotype(hpo_g2p_path)
    print(f'Genes: {len(hpo_g2p_gene_phenotype)}')  # 4231
    print('==ORPHA==')
    print(f'Genes: {len(orpha_gene_phenotype)}')  # 2763
    print('==HPO_ORPHA==')
    print(f'Genes: {len(hpo_orpha_gene_phenotype)}')  # 2730
    print('==DECIPHER==')
    decipher_gene_phenotype = parse_decipher_ddg2p(ddg2p_path)
    print(f'Genes: {len(decipher_gene_phenotype)}')  # 1370

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
    # save to file
    out_path = verify_output_path(args.out)
    out_path.write_text(
        json.dumps(
            dict([(g, tuple(gene_phenotype[g])) for g in gene_phenotype]),
            indent=2,
            sort_keys=True
        )
    )

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# windows

# python -m util.hpo.make_gene_phenotype --hpoa E:\hpo\hpo-20191011\phenotype.hpoa --orpha6 E:\hpo\orpha-20191101\en_product6.xml --orpha4 E:\hpo\orpha-20191101\en_product4_HPO.xml --hpog2p E:\hpo\hpo-20191011\annotation\ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt --ddg2p E:\hpo\decipher-20191101\DDG2P_1_11_2019.csv --omim_morbidmap E:\hpo\omim-20191101\morbidmap.txt --out E:\hpo\hpo-20191011\hpo-20191011-gene-phenotype.json

# Processing: phenotype.hpoa, en_product6.xml, en_product4_HPO.xml, ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt, morbidmap.txt, DDG2P_1_11_2019.csv
# ==HPO: {hpoa_path.name}==
# Disorders: 11372

# ==ORPHA_GENE: {orpha6_path.name}==
# Disorders: 3792

# ==ORPHA_HPO: {orpha4_path.name}==
# Disorders: 3771

# ==OMIM: {omim_morbidmap_path.name}==
# Disorders: 10905
# Genes: 14280
# ==HPO_OMIM==
# Genes: 13003
# ==HPO_G2P==
# Genes: 4231
# ==ORPHA==
# Genes: 2763
# ==HPO_ORPHA==
# Genes: 2730
# ==DECIPHER==
# Genes: 1370

# HPO_OMIM: 293857  ([('MTDPS2', 'HP:0003690'), ('FJHN', 'HP:0005584'), ('CORTRD2', 'HP:0000956'), ('RCHTS', 'HP:0002119'), ('NFE2L2', 'HP:0002721')])
# HPO_G2P: 161003  ([('NFE2L2', 'HP:0002721'), ('NSD2', 'HP:0000384'), ('RNF168', 'HP:0006530'), ('KCNJ10', 'HP:0000007'), ('FBN1', 'HP:0025586')])
# HPO_OMIM & HPO_G2P: 76990  ([('CACNA1D', 'HP:0000822'), ('NDN', 'HP:0000750'), ('NFE2L2', 'HP:0002721'), ('PDP1', 'HP:0001263'), ('PRKCG', 'HP:0000639')])
# HPO_OMIM | HPO_G2P: 377870  ([('MTDPS2', 'HP:0003690'), ('FJHN', 'HP:0005584'), ('NFE2L2', 'HP:0002721'), ('MRT60', 'HP:0000007'), ('NSD2', 'HP:0000384')])
# HPO_OMIM - HPO_G2P: 216867  ([('MTDPS2', 'HP:0003690'), ('FJHN', 'HP:0005584'), ('CORTRD2', 'HP:0000956'), ('RCHTS', 'HP:0002119'), ('FRODO', 'HP:0000136')])
# HPO_G2P - HPO_OMIM: 84013  ([('GPC3', 'HP:0000303'), ('NSD2', 'HP:0000384'), ('RNF168', 'HP:0006530'), ('XRCC2', 'HP:0012041'), ('MUSK', 'HP:0031108')])


# HPO_OMIM|HPO_G2P: 377870  ([('MTDPS2', 'HP:0003690'), ('FJHN', 'HP:0005584'), ('NFE2L2', 'HP:0002721'), ('MRT60', 'HP:0000007'), ('NSD2', 'HP:0000384')])
# ORPHA: 109591  ([('GPC3', 'HP:0000303'), ('WHRN', 'HP:0000359'), ('RNF168', 'HP:0006530'), ('XRCC2', 'HP:0012041'), ('MUSK', 'HP:0031108')])
# HPO_OMIM|HPO_G2P & ORPHA: 96190  ([('GPC3', 'HP:0000303'), ('RNF168', 'HP:0006530'), ('XRCC2', 'HP:0012041'), ('MUSK', 'HP:0031108'), ('KCNJ5', 'HP:0200114')])
# HPO_OMIM|HPO_G2P | ORPHA: 391271  ([('MTDPS2', 'HP:0003690'), ('FJHN', 'HP:0005584'), ('NFE2L2', 'HP:0002721'), ('MRT60', 'HP:0000007'), ('NSD2', 'HP:0000384')])
# HPO_OMIM|HPO_G2P - ORPHA: 281680  ([('MTDPS2', 'HP:0003690'), ('FJHN', 'HP:0005584'), ('NFE2L2', 'HP:0002721'), ('CORTRD2', 'HP:0000956'), ('RCHTS', 'HP:0002119')])
# ORPHA - HPO_OMIM|HPO_G2P: 13401  ([('WHRN', 'HP:0000359'), ('CACNA1B', 'HP:0001290'), ('KIAA0753', 'HP:0001252'), ('TRPM4', 'HP:0012722'), ('NHLRC1', 'HP:0007359')])


# HPO_OMIM|HPO_G2P|ORPHA: 391271  ([('MTDPS2', 'HP:0003690'), ('FJHN', 'HP:0005584'), ('NFE2L2', 'HP:0002721'), ('MRT60', 'HP:0000007'), ('NSD2', 'HP:0000384')])
# HPO_ORPHA: 111511  ([('GPC3', 'HP:0000303'), ('WHRN', 'HP:0000359'), ('RNF168', 'HP:0006530'), ('XRCC2', 'HP:0012041'), ('MUSK', 'HP:0031108')])
# HPO_OMIM|HPO_G2P|ORPHA & HPO_ORPHA: 109688  ([('GPC3', 'HP:0000303'), ('WHRN', 'HP:0000359'), ('RNF168', 'HP:0006530'), ('XRCC2', 'HP:0012041'), ('MUSK', 'HP:0031108')])
# HPO_OMIM|HPO_G2P|ORPHA | HPO_ORPHA: 393094  ([('MTDPS2', 'HP:0003690'), ('FJHN', 'HP:0005584'), ('NFE2L2', 'HP:0002721'), ('MRT60', 'HP:0000007'), ('NSD2', 'HP:0000384')])
# HPO_OMIM|HPO_G2P|ORPHA - HPO_ORPHA: 281583  ([('MTDPS2', 'HP:0003690'), ('FJHN', 'HP:0005584'), ('NFE2L2', 'HP:0002721'), ('CORTRD2', 'HP:0000956'), ('RCHTS', 'HP:0002119')])
# HPO_ORPHA - HPO_OMIM|HPO_G2P|ORPHA: 1823  ([('NUP133', 'HP:0001419'), ('PCARE', 'HP:0000006'), ('FGG', 'HP:0000006'), ('MKS1', 'HP:0010983'), ('RIMS1', 'HP:0001419')])


# HPO_OMIM|HPO_G2P|ORPHA|HPO_ORPHA: 393094  ([('MTDPS2', 'HP:0003690'), ('FJHN', 'HP:0005584'), ('NFE2L2', 'HP:0002721'), ('MRT60', 'HP:0000007'), ('NSD2', 'HP:0000384')])
# DECIPHER: 28710  ([('LRPPRC', 'HP:0001250'), ('FKRP', 'HP:0001249'), ('KCNJ10', 'HP:0000007'), ('GRIN2A', 'HP:0000006'), ('MTHFR', 'HP:0000252')])
# HPO_OMIM|HPO_G2P|ORPHA|HPO_ORPHA & DECIPHER: 25314  ([('LRPPRC', 'HP:0001250'), ('FKRP', 'HP:0001249'), ('KCNJ10', 'HP:0000007'), ('GRIN2A', 'HP:0000006'), ('MTHFR', 'HP:0000252')])
# HPO_OMIM|HPO_G2P|ORPHA|HPO_ORPHA | DECIPHER: 396490  ([('MTDPS2', 'HP:0003690'), ('FJHN', 'HP:0005584'), ('NFE2L2', 'HP:0002721'), ('MRT60', 'HP:0000007'), ('NSD2', 'HP:0000384')])
# HPO_OMIM|HPO_G2P|ORPHA|HPO_ORPHA - DECIPHER: 367780  ([('MTDPS2', 'HP:0003690'), ('FJHN', 'HP:0005584'), ('NFE2L2', 'HP:0002721'), ('MRT60', 'HP:0000007'), ('NSD2', 'HP:0000384')])
# DECIPHER - HPO_OMIM|HPO_G2P|ORPHA|HPO_ORPHA: 3396  ([('FREM2', 'HP:0000057'), ('LIAS', 'HP:0002154'), ('PEX1', 'HP:0000846'), ('MAB21L1', 'HP:0000046'), ('COQ8A', 'HP:0003652')])

# gene_phenotype (396490)
#  - Genes (13478)
#  - Phenotypes (8076)
# Run time: 7.44 s