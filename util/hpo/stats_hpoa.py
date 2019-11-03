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
        '-6',
        '--xml6',
        type=str,
        required=True,
        help="Path to en_product6.xml file, required."
    )
    parser.add_argument(
        '-4',
        '--xml4',
        type=str,
        required=True,
        help="Path to en_product4_HPO.xml file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    hpoa_path = verify_input_path(args.hpoa)
    xml6_path = verify_input_path(args.xml6)
    xml4_path = verify_input_path(args.xml4)
    print(f'Processing: {hpoa_path.name}, {xml6_path.name}, {xml4_path.name}')

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

    orpha_gene = parse_orpha_gene_xml(xml6_path)
    orpha_phenotype = parse_orpha_phenotype_xml(xml4_path)

    print('')
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
    print('==ORPHA_HPO==')
    print(f'Disorders: {len(orpha_phenotype)}')

    print('')
    print('==SET STATS==')
    print_set_stats(
        'HPO_ORPHA', hpo_orpha_disorders, 'ORPHA_GENE', set(orpha_gene)
    )
    print_set_stats(
        'HPO_ORPHA', hpo_orpha_disorders, 'ORPHA_HPO', set(orpha_phenotype)
    )

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# windows
# python -m util.hpo.stats_hpoa --hpoa E:\hpo\hpo-20191011\phenotype.hpoa --xml6 E:\hpo\orpha-20191101\en_product6.xml --xml4 E:\hpo\orpha-20191101\en_product4_HPO.xml
