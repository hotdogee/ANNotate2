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
from .orpha.stats_orpha_xml import parse_orpha_xml
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
    return {}


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
        '-x',
        '--xml',
        type=str,
        required=True,
        help="Path to .xml file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    hpoa_path = verify_input_path(args.hpoa)
    xml_path = verify_input_path(args.xml)
    print(f'Processing: {hpoa_path.name}, {xml_path.name}')

    disorder_phenotype = parse_hpoa(hpoa_path)

    disorder_gene = parse_orpha_xml(xml_path)

    print('==COUNT==')
    print(f'Disorders: {len(disorder_gene)}')
    annotations = [
        gene for disorder in disorder_gene for gene in disorder_gene[disorder]
    ]
    print(f'Disorder to Gene annotations: {len(annotations)}')
    print(
        f' - no SwissProt reference: {len([a for a in annotations if "SwissProt" not in a])}'
    )
    print(
        f' - no Ensembl reference: {len([a for a in annotations if "Ensembl" not in a])}'
    )

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# windows
# python -m util.hpo.stats_hpoa --hpoa E:\hpo\hpo-20191011\phenotype.hpoa --xml E:\hpo\orpha-20191101\en_product6.xml
