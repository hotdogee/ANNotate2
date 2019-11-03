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

# Download: http://www.orphadata.org/data/xml/en_product6.xml
# <?xml version="1.0" encoding="ISO-8859-1"?>
# <JDBOR date="2019-11-01 05:00:27" version="1.2.11 / 4.1.6 [2018-04-12] (orientdb version)" copyright="Orphanet (c) 2019">
# <Availability>
#  <Licence>
#     <FullName lang="en">Creative Commons Attribution 4.0 International</FullName>
#     <ShortIdentifier>CC-BY-4.0</ShortIdentifier>
#     <LegalCode>https://creativecommons.org/licenses/by/4.0/legalcode</LegalCode>
#  </Licence>
# </Availability>
#   <DisorderList count="3792">
#     <Disorder id="17601">
#       <OrphaNumber>166024</OrphaNumber>
#       <Name lang="en">Multiple epiphyseal dysplasia, Al-Gazali type</Name>
#       <DisorderGeneAssociationList count="1">
#         <DisorderGeneAssociation>
#           <SourceOfValidation>22587682[PMID]</SourceOfValidation>
#           <Gene id="20160">
#             <OrphaNumber>268061</OrphaNumber>
#             <Name lang="en">kinesin family member 7</Name>
#             <Symbol>KIF7</Symbol>
#             <SynonymList count="1">
#               <Synonym lang="en">JBTS12</Synonym>
#             </SynonymList>
#             <ExternalReferenceList count="6">
#               <ExternalReference id="57240">
#                 <Source>Ensembl</Source>
#                 <Reference>ENSG00000166813</Reference>
#               </ExternalReference>
#               <ExternalReference id="51758">
#                 <Source>Genatlas</Source>
#                 <Reference>KIF7</Reference>
#               </ExternalReference>
#               <ExternalReference id="51756">
#                 <Source>HGNC</Source>
#                 <Reference>30497</Reference>
#               </ExternalReference>
#               <ExternalReference id="51757">
#                 <Source>OMIM</Source>
#                 <Reference>611254</Reference>
#               </ExternalReference>
#               <ExternalReference id="97306">
#                 <Source>Reactome</Source>
#                 <Reference>Q2M1P5</Reference>
#               </ExternalReference>
#               <ExternalReference id="51759">
#                 <Source>SwissProt</Source>
#                 <Reference>Q2M1P5</Reference>
#               </ExternalReference>
#             </ExternalReferenceList>
#             <LocusList count="1">
#               <Locus id="16859">
#               </Locus>
#             </LocusList>
#           </Gene>
#           <DisorderGeneAssociationType id="17949">
#             <Name lang="en">Disease-causing germline mutation(s) in</Name>
#           </DisorderGeneAssociationType>
#           <DisorderGeneAssociationStatus id="17991">
#             <Name lang="en">Assessed</Name>
#           </DisorderGeneAssociationStatus>
#         </DisorderGeneAssociation>
#       </DisorderGeneAssociationList>
#     </Disorder>


class OrphaCollector:
    def __init__(self):
        self.tag = None
        self.disorder_gene = {}
        self.disorder = None
        self.gene = {}
        self.source = None

    def start(self, tag, attr):
        self.tag = tag
        if tag == 'Disorder':
            self.disorder = None
        elif tag == 'JDBOR':
            self.disorder_gene = {}
            self.disorder = None
            self.gene = {}
            self.source = None

    def end(self, tag):
        pass

    def data(self, data):
        data = data.strip()
        if not data:
            return
        if self.tag == 'OrphaNumber' and self.disorder is None:
            self.disorder = data
            self.disorder_gene[self.disorder] = []
        elif self.tag == 'Symbol':
            self.gene = {'symbol': data}
            self.disorder_gene[self.disorder].append(self.gene)
        elif self.tag == 'Source':
            self.source = data
        elif self.tag == 'Reference':
            self.gene[self.source] = data

    def close(self):
        return self.disorder_gene


def parse_orpha_xml(xml_path):
    with xml_path.open(encoding='ISO-8859-1') as stream:
        # stream=xml_path.open(encoding='ISO-8859-1')
        collector = OrphaCollector()
        parser = ET.XMLParser(target=collector)
        ET.parse(stream, parser=parser)
        # a=collector.disorder_gene
    return collector.disorder_gene


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


def verify_output_path(p):
    # get absolute path to dataset directory
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # existing file
    if path.exists():
        raise FileExistsError(errno.ENOENT, os.strerror(errno.EEXIST), path)
    # is dir
    if path.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
    # assert dirs
    path.parent.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
    return path


def verify_indir_path(p):
    # get absolute path
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # doesn't exist
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    # existing file or directory
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(
                errno.ENOTDIR, os.strerror(errno.ENOTDIR), path
            )
        else:  # got existing directory
            assert len([x for x in path.iterdir()]) != 0, 'Directory is empty'
    return path


def verify_outdir_path(p, required_empty=True):
    # get absolute path
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # existing file or directory
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(
                errno.ENOTDIR, os.strerror(errno.ENOTDIR), path
            )
        elif required_empty:  # got existing directory
            assert len([x for x in path.iterdir()]) == 0, 'Directory not empty'
    # directory does not exist, create it
    path.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
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
        description='Print some statistics of the orpha en_product6.xml file.'
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

    xml_path = verify_input_path(args.xml)
    print(f'Processing: {xml_path.name}')

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
# python .\util\hpo\orpha\stats_orpha_xml.py --xml E:\hpo\orpha-20191101\en_product6.xml
