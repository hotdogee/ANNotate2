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
    disorder_phenotype = defaultdict(set)
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
            disorder_phenotype[disorder].add(Phenotype(hpo, evidence))
    return disorder_phenotype


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
    fieldnames=['gene symbol', 'gene mim', 'disease name', 'disease mim', 'DDD category', 'allelic requirement', 'mutation consequence', 'phenotypes', 'organ specificity list', 'pmids', 'panel', 'prev symbols', 'hgnc id', 'gene disease pair entry date']
    decipher_gene_phenotype = defaultdict(set)
    with path.open(mode='r', encoding='utf-8') as f:
        rows = csv.reader(f)
        count = 0
        for row in rows:
            count += 1
            if count == 1:
                continue
            # assert length == 14
            assert (
                len(row) == 14
            ), f'token length: {len(row)}, expected: 14'
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
        unit='bytes',
        dynamic_ncols=True,
        ascii=True,
        desc=path.name
    ) as t:
        for line in f:
            t.update(len(line))
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
                    symbol = match.group(1)
                else:
                    symbol = None
            else:
                sequence += line
        if sequence:
            if uid in whitelist:
                id_seq[uid] = sequence
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
        help="Path to .hpoa file, required."
    )
    parser.add_argument(
        '-x6',
        '--xml6',
        type=str,
        required=True,
        help="Path to en_product6.xml file, required."
    )
    parser.add_argument(
        '-x4',
        '--xml4',
        type=str,
        required=True,
        help="Path to en_product4_HPO.xml file, required."
    )
    parser.add_argument(
        '-rp',
        '--org2p',
        type=str,
        required=True,
        help=
        "Path to ORPHA_ALL_FREQUENCIES_genes_to_phenotype.txt file, required."
    )
    parser.add_argument(
        '-rg',
        '--orp2g',
        type=str,
        required=True,
        help=
        "Path to ORPHA_ALL_FREQUENCIES_phenotype_to_genes.txt file, required."
    )
    parser.add_argument(
        '-o',
        '--omim',
        type=str,
        required=True,
        help="Path to mim2gene.txt file, required."
    )
    parser.add_argument(
        '-d',
        '--ddg2p',
        type=str,
        required=True,
        help="Path to DDG2P_1_11_2019.csv file, required."
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
    xml6_path = verify_input_path(args.xml6)
    xml4_path = verify_input_path(args.xml4)
    org2p_path = verify_input_path(args.org2p)
    orp2g_path = verify_input_path(args.orp2g)
    omim_path = verify_input_path(args.omim)
    ddg2p_path = verify_input_path(args.ddg2p)
    fa1_path = verify_input_path(args.fa1)
    fa2_path = verify_input_path(args.fa2)
    fa3_path = verify_input_path(args.fa3)
    print(
        f'Processing: {hpoa_path.name}, {xml6_path.name}, {xml4_path.name}, {org2p_path.name}, {orp2g_path.name}, {omim_path.name}, {ddg2p_path.name}'
    )

    disorder_phenotype = parse_hpoa(hpoa_path)
    print('==HPO==')
    print(f'Disorders: {len(disorder_phenotype)}')
    phenotype_annotations = [
        phenotype for disorder in disorder_phenotype
        for phenotype in disorder_phenotype[disorder]
    ]
    print(f'Disorder to phenotype annotations: {len(phenotype_annotations)}')
    print(
        f' - IEA (inferred from electronic annotation): {len([a for a in phenotype_annotations if a.evidence == "IEA"])}'
    )
    print(
        f' - PCS (published clinical study): {len([a for a in phenotype_annotations if a.evidence == "PCS"])}'
    )
    print(
        f' - ICE (individual clinical experience): {len([a for a in phenotype_annotations if a.evidence == "ICE"])}'
    )
    print(
        f' - TAS (traceable author statement): {len([a for a in phenotype_annotations if a.evidence == "TAS"])}'
    )
    print(
        f' - evidence list: {set([a.evidence for a in phenotype_annotations])}'
    )
    hpo_decipher_disorders = set(
        [
            d.split(':')[1]
            for d in disorder_phenotype if d.split(':')[0] == 'DECIPHER'
        ]
    )
    hpo_omim_disorders = set(
        [
            d.split(':')[1]
            for d in disorder_phenotype if d.split(':')[0] == 'OMIM'
        ]
    )
    hpo_orpha_disorders = set(
        [
            d.split(':')[1]
            for d in disorder_phenotype if d.split(':')[0] == 'ORPHA'
        ]
    )
    print(f' - decipher disorders: {len(hpo_decipher_disorders)}')
    print(f' - omim disorders: {len(hpo_omim_disorders)}')
    print(f' - orpha disorders: {len(hpo_orpha_disorders)}')

    print('')
    orpha_gene = parse_orpha_gene_xml(xml6_path)
    print('==ORPHA_GENE==')
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
    orpha_phenotype = parse_orpha_phenotype_xml(xml4_path)
    print('==ORPHA_HPO==')
    print(f'Disorders: {len(orpha_phenotype)}')

    print('')
    omim_gene = parse_omim_to_genes(omim_path)
    print('==OMIM==')
    print(f'Disorders: {len(omim_gene)}')

    print('')
    # build gene_phenotype from disorder_phenotype and orpha_gene
    hpo_orpha_gene_phenotype = defaultdict(set)
    for orpha in orpha_gene:
        phenotypes = disorder_phenotype.get(f'ORPHA:{orpha}')
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

    # build gene_phenotype from disorder_phenotype and omim_gene
    count = 0
    hpo_omim_gene_phenotype = defaultdict(set)
    for omim in omim_gene:
        phenotypes = disorder_phenotype.get(f'OMIM:{omim}')
        if not phenotypes:
            continue
        count += 1
        phenotypes_set = set([p.hpo for p in phenotypes])
        for gene_symbol in omim_gene[omim]:
            hpo_omim_gene_phenotype[gene_symbol] |= phenotypes_set
    print(f'count {count}')

    org2p_gene_phenotype = parse_hpo_genes_to_phenotype(org2p_path)
    orp2g_gene_phenotype = parse_hpo_phenotype_to_genes(orp2g_path)
    decipher_gene_phenotype = parse_decipher_ddg2p(ddg2p_path)
    print('==HPO_ORPHA_G2P==')
    print(f'Genes: {len(org2p_gene_phenotype)}')
    print('==HPO_ORPHA_P2G==')
    print(f'Genes: {len(orp2g_gene_phenotype)}')
    print('==HPO_ORPHA==')
    print(f'Genes: {len(hpo_orpha_gene_phenotype)}')
    print('==ORPHA==')
    print(f'Genes: {len(orpha_gene_phenotype)}')
    print('==OMIM==')
    print(f'Genes: {len(hpo_omim_gene_phenotype)}')
    print('==DECIPHER==')
    print(f'Genes: {len(decipher_gene_phenotype)}')
    print_set_stats(
        'ORPHA|HPO_ORPHA|HPO_ORPHA_G2P',
        set(
            orpha_gene_phenotype
        ) | set(
            hpo_orpha_gene_phenotype
        ) | set(
            org2p_gene_phenotype
        ), 'DECIPHER',
        set(
            decipher_gene_phenotype
        )
    )

    print('')
    print('==SET STATS==')
    print_set_stats(
        'HPO_ORPHA', hpo_orpha_disorders, 'ORPHA_GENE', set(orpha_gene)
    )
    print_set_stats(
        'HPO_ORPHA', hpo_orpha_disorders, 'ORPHA_HPO', set(orpha_phenotype)
    )
    print_set_stats('HPO_OMIM', hpo_omim_disorders, 'OMIM_GENE', set(omim_gene))
    print_set_stats(
        'HPO_ORPHA_G2P',
        set(
            [
                (g, p) for g in org2p_gene_phenotype
                for p in org2p_gene_phenotype[g]
            ]
        ), 'HPO_ORPHA_P2G',
        set(
            [
                (g, p) for g in orp2g_gene_phenotype
                for p in orp2g_gene_phenotype[g]
            ]
        )
    )
    print_set_stats(
        'HPO_ORPHA',
        set(
            [
                (g, p) for g in hpo_orpha_gene_phenotype
                for p in hpo_orpha_gene_phenotype[g]
            ]
        ), 'ORPHA',
        set(
            [
                (g, p) for g in orpha_gene_phenotype
                for p in orpha_gene_phenotype[g]
            ]
        )
    )
    print_set_stats(
        'HPO_ORPHA',
        set(
            [
                (g, p) for g in hpo_orpha_gene_phenotype
                for p in hpo_orpha_gene_phenotype[g]
            ]
        ), 'HPO_ORPHA_G2P',
        set(
            [
                (g, p) for g in org2p_gene_phenotype
                for p in org2p_gene_phenotype[g]
            ]
        )
    )
    print_set_stats(
        'ORPHA',
        set(
            [
                (g, p) for g in orpha_gene_phenotype
                for p in orpha_gene_phenotype[g]
            ]
        ), 'HPO_ORPHA_G2P',
        set(
            [
                (g, p) for g in org2p_gene_phenotype
                for p in org2p_gene_phenotype[g]
            ]
        )
    )
    print_set_stats(
        'ORPHA|HPO_ORPHA',
        set(
            [
                (g, p) for g in orpha_gene_phenotype
                for p in orpha_gene_phenotype[g]
            ]
        ) | set(
            [
                (g, p) for g in hpo_orpha_gene_phenotype
                for p in hpo_orpha_gene_phenotype[g]
            ]
        ), 'HPO_ORPHA_G2P',
        set(
            [
                (g, p) for g in org2p_gene_phenotype
                for p in org2p_gene_phenotype[g]
            ]
        )
    )
    print_set_stats(
        'ORPHA|HPO_ORPHA|HPO_ORPHA_G2P',
        set(
            [
                (g, p) for g in orpha_gene_phenotype
                for p in orpha_gene_phenotype[g]
            ]
        ) | set(
            [
                (g, p) for g in hpo_orpha_gene_phenotype
                for p in hpo_orpha_gene_phenotype[g]
            ]
        ) | set(
            [
                (g, p) for g in org2p_gene_phenotype
                for p in org2p_gene_phenotype[g]
            ]
        ), 'DECIPHER',
        set(
            [
                (g, p) for g in decipher_gene_phenotype
                for p in decipher_gene_phenotype[g]
            ]
        )
    )
    gene_phenotype_set = set(
            [
                (g, p) for g in orpha_gene_phenotype
                for p in orpha_gene_phenotype[g]
            ]
        ) | set(
            [
                (g, p) for g in hpo_orpha_gene_phenotype
                for p in hpo_orpha_gene_phenotype[g]
            ]
        ) | set(
            [
                (g, p) for g in org2p_gene_phenotype
                for p in org2p_gene_phenotype[g]
            ]
        )|set(
            [
                (g, p) for g in decipher_gene_phenotype
                for p in decipher_gene_phenotype[g]
            ]
        )
    gene_phenotype = defaultdict(set)
    for g, p in gene_phenotype_set:
        gene_phenotype[g].add(p)
    print(f'gene_phenotype Genes: {len(gene_phenotype)}')

    # read sequences
    gene_seq = defaultdict(set)
    index_uniprot_fa(fa1_path, gene_seq, gene_phenotype)
    index_uniprot_fa(fa2_path, gene_seq, gene_phenotype)
    index_uniprot_fa(fa3_path, gene_seq, gene_phenotype)
    print(f'gene_seq Genes: {len(gene_seq)}')
    print(f'gene_seq Seqs: {set([s for g in gene_seq for s in gene_seq[g]])}')

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# windows
# python -m util.hpo.stats_hpoa --hpoa E:\hpo\hpo-20191011\phenotype.hpoa --xml6 E:\hpo\orpha-20191101\en_product6.xml --xml4 E:\hpo\orpha-20191101\en_product4_HPO.xml --org2p E:\hpo\hpo-20191011\annotation\ORPHA_ALL_FREQUENCIES_genes_to_phenotype.txt --orp2g E:\hpo\hpo-20191011\annotation\ORPHA_ALL_FREQUENCIES_phenotype_to_genes.txt --omim E:\hpo\omim-20191101\mim2gene.txt --ddg2p E:\hpo\decipher-20191101\DDG2P_1_11_2019.csv

# python -m util.hpo.stats_hpoa --hpoa E:\hpo\hpo-20191011\phenotype.hpoa --xml6 E:\hpo\orpha-20191101\en_product6.xml --xml4 E:\hpo\orpha-20191101\en_product4_HPO.xml --org2p E:\hpo\hpo-20191011\annotation\ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt --orp2g E:\hpo\hpo-20191011\annotation\ALL_SOURCES_ALL_FREQUENCIES_phenotype_to_genes.txt --omim E:\hpo\omim-20191101\mim2gene.txt --ddg2p E:\hpo\decipher-20191101\DDG2P_1_11_2019.csv

# python -m util.hpo.stats_hpoa --hpoa E:\hpo\hpo-20191011\phenotype.hpoa --xml6 E:\hpo\orpha-20191101\en_product6.xml --xml4 E:\hpo\orpha-20191101\en_product4_HPO.xml --org2p E:\hpo\hpo-20191011\annotation\ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt --orp2g E:\hpo\hpo-20191011\annotation\ALL_SOURCES_ALL_FREQUENCIES_phenotype_to_genes.txt --omim E:\hpo\omim-20191101\mim2gene.txt --ddg2p E:\hpo\decipher-20191101\DDG2P_1_11_2019.csv --fa1 F:\uniprot\uniprot-20191015\uniprot_trembl.fasta.gz --fa2 F:\uniprot\uniprot-20191015\uniprot_sprot.fasta --fa3 F:\uniprot\uniprot-20191015\uniprot_sprot_varsplic.fasta


# Use ORPHA|HPO_ORPHA|HPO_ORPHA_G2P for Orpha dataset
