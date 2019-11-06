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
from pprint import pprint
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict
from ..verify_paths import verify_input_path, verify_output_path, verify_indir_path, verify_outdir_path

# Download: ftp://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/idmapping/idmapping.dat.gz
# 1. UniProtKB-AC
# 2. ID_type
# 3. ID
# Q6GZX4	UniProtKB-ID	001R_FRG3G
# Q6GZX4	Gene_ORFName	FV3-001R
# Q6GZX4	GI	81941549
# Q6GZX4	GI	49237298
# Q6GZX4	UniRef100	UniRef100_Q6GZX4
# Q6GZX4	UniRef90	UniRef90_Q6GZX4
# Q6GZX4	UniRef50	UniRef50_Q6GZX4
# Q6GZX4	UniParc	UPI00003B0FD4
# Q6GZX4	EMBL	AY548484
# Q6GZX4	EMBL-CDS	AAT09660.1
# Q6GZX4	NCBI_TaxID	654924
# Q6GZX4	RefSeq	YP_031579.1
# Q6GZX4	RefSeq_NT	NC_005946.1
# Q6GZX4	GeneID	2947773
# Q6GZX4	KEGG	vg:2947773
# Q6GZX4	CRC64	B4840739BF7D4121

# >db|UniqueIdentifier|EntryName ProteinName OS=OrganismName OX=OrganismIdentifier [GN=GeneName ]PE=ProteinExistence SV=SequenceVersion
# db is ‘sp’ for UniProtKB/Swiss-Prot and ‘tr’ for UniProtKB/TrEMBL.
# UniqueIdentifier is the primary accession number of the UniProtKB entry.
# EntryName is the entry name of the UniProtKB entry.
# ProteinName is the recommended name of the UniProtKB entry as annotated in the RecName field. For UniProtKB/TrEMBL entries without a RecName field, the SubName field is used. In case of multiple SubNames, the first one is used. The ‘precursor’ attribute is excluded, ‘Fragment’ is included with the name if applicable.
# OrganismName is the scientific name of the organism of the UniProtKB entry.
# OrganismIdentifier is the unique identifier of the source organism, assigned by the NCBI.
# GeneName is the first gene name of the UniProtKB entry. If there is no gene name, OrderedLocusName or ORFname, the GN field is not listed.
# ProteinExistence is the numerical value describing the evidence for the existence of the protein.
# SequenceVersion is the version number of the sequence.
# >sp|Q6GZX4|001R_FRG3G Putative transcription factor 001R OS=Frog virus 3 (isolate Goorha) OX=654924 GN=FV3-001R PE=4 SV=1
# >sp|Q6GZX3|002L_FRG3G Uncharacterized protein 002L OS=Frog virus 3 (isolate Goorha) OX=654924 GN=FV3-002L PE=4 SV=1


def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def explore_unfound_genes(path):
    # initializes
    unfound_genes = {
        'TRNN', 'HYMAI', 'DUX4L1', 'TRNS1', 'MT-TV', 'SPG16', 'GNAS-AS1',
        'KCNQ1OT1', 'MACROH2A1', 'USH1K', 'MT-TN', 'HELLPAR', 'CXORF56',
        'DYT13', 'MT-TK', 'USH1E', 'TRNS2', 'MT-TT', 'PWRN1', 'XIST', 'GARS1',
        'USH1H', 'MT-TF', 'SARS1', 'SNORD116-1', 'SPG37', 'FRA16E', 'MT-TW',
        'TRNC', 'SCA30', 'IARS1', 'AARS1', 'SPG34', 'SLC7A2-IT1', 'SPG14',
        'PWAR1', 'TRNQ', 'SCA37', 'MT-TS2', 'SNORD115-1', 'HBB-LCR', 'EPRS1',
        'KIFBP', 'MIR184', 'TRNL2', 'OPA2', 'C12ORF57', 'RNU4ATAC', 'MKRN3-AS1',
        'H1-4', 'MT-TP', 'H19-ICR', 'TRNK', 'SCA20', 'SNORD118', 'TRNW',
        'DARS1', 'TRNI', 'MT-TE', 'SPG25', 'RARS1', 'SPG23', 'TRNE', 'MIR96',
        'C12ORF4', 'MT-TQ', 'MIR204', 'SCA25', 'SPG41', 'SPG29', 'TRNT',
        'IL12A-AS1', 'DISC2', 'MT-TL1', 'IPW', 'FMR3', 'SPG36', 'SCA32',
        'C9ORF72', 'MT-TH', 'MARS1', 'RNU12', 'DYT17', 'WHCR', 'STING1',
        'YARS1', 'TRNV', 'KARS1', 'GINGF2', 'TRNF', 'DYT15', 'LARS1', 'ARSL',
        'SPG38', 'SPG19', 'DYT21', 'TRNP', 'SPG32', 'SPG27', 'QARS1', 'MT-TS1',
        'TRNL1', 'RMRP', 'C11ORF95', 'MT-TL2', 'HARS1', 'ADSS1', 'SPG24',
        'C15ORF41'
    }
    # Gene_Name
    # Gene_ORFName
    # Gene_OrderedLocusName
    # Gene_Synonym
    vocab = defaultdict(set)
    gene_seq = defaultdict(set)
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
    with f, tqdm(
        dynamic_ncols=True, ascii=True, desc=path.name, unit='lines'
    ) as t:
        rows = csv.reader(f, delimiter='\t')
        for row in rows:
            t.update()
            acc = row[0].strip().upper()
            id_type = row[1].strip()
            id = row[2].strip().upper()
            vocab[id_type].add(id)
            gene_seq[id].add(acc)
    print(f'Genes without a sequence ({len(unfound_genes)}): {unfound_genes}')
    print(
        f" - Gene_Name ({len(vocab['Gene_Name'] & unfound_genes)}): {vocab['Gene_Name'] & unfound_genes}"
    )
    print(
        f" - Gene_ORFName ({len(vocab['Gene_ORFName'] & unfound_genes)}): {vocab['Gene_ORFName'] & unfound_genes}"
    )
    print(
        f" - Gene_OrderedLocusName ({len(vocab['Gene_OrderedLocusName'] & unfound_genes)}): {vocab['Gene_OrderedLocusName'] & unfound_genes}"
    )
    print(
        f" - Gene_Synonym ({len(vocab['Gene_Synonym'] & unfound_genes)}): {vocab['Gene_Synonym'] & unfound_genes}"
    )
    print(
        f" - ALL ({len(set(gene_seq) & unfound_genes)}): {set(gene_seq) & unfound_genes}"
    )
    pprint(
        dict(
            [
                (s, len(gene_seq[s]))
                for s in sorted(unfound_genes) if s in gene_seq
            ]
        )
    )

    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print idmapping types.')
    parser.add_argument(
        '-d',
        '--dat',
        type=str,
        required=True,
        help="Path to idmapping_filtered.dat[.gz] file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    dat_path = verify_input_path(args.dat)
    print(f'Processing: {dat_path.name}')

    explore_unfound_genes(dat_path)

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# windows
# python -m util.idmapping.make_gene_symbol_mapping_from_idmapping --dat E:\idmapping\idmapping-20191105\idmapping_filtered.dat

# idmapping.dat: 2,442,456,179 lines [59:43, 681539 lines/s]
# ID Types (104):
# Allergome
# ArachnoServer
# Araport
# BioCyc
# BioGrid
# BioMuta
# CCDS
# CGD
# CPTAC
# CRC64
# ChEMBL
# ChiTaRS
# CollecTF
# ComplexPortal
# ConoServer
# DIP
# DMDM
# DNASU
# DisProt
# DrugBank
# EMBL
# EMBL-CDS
# ESTHER
# EchoBASE
# EcoGene
# Ensembl
# EnsemblGenome
# EnsemblGenome_PRO
# EnsemblGenome_TRS
# Ensembl_PRO
# Ensembl_TRS
# EuPathDB
# FlyBase
# GI
# GeneCards
# GeneDB
# GeneID
# GeneReviews
# GeneTree
# GeneWiki
# Gene_Name
# Gene_ORFName
# Gene_OrderedLocusName
# Gene_Synonym
# GenomeRNAi
# GlyConnect
# GuidetoPHARMACOLOGY
# HGNC
# HOGENOM
# HPA
# KEGG
# KO
# LegioList
# Leproma
# MEROPS
# MGI
# MIM
# MINT
# MaizeGDB
# NCBI_TaxID
# OMA
# Orphanet
# OrthoDB
# PATRIC
# PDB
# PeroxiBase
# PharmGKB
# PlantReactome
# PomBase
# ProteomicsDB
# PseudoCAP
# REBASE
# RGD
# Reactome
# RefSeq
# RefSeq_NT
# SGD
# STRING
# SwissLipids
# TAIR
# TCDB
# TreeFam
# TubercuList
# UCSC
# UniParc
# UniPathway
# UniProtKB-ID
# UniRef100
# UniRef50
# UniRef90
# VGNC
# VectorBase
# WBParaSite
# World-2DPAGE
# WormBase
# WormBase_PRO
# WormBase_TRS
# Xenbase
# ZFIN
# dictyBase
# eggNOG
# euHCVdb
# mycoCLAP
# neXtProt
# Run time: 3583.82 s
