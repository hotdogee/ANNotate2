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
from ..verify_paths import verify_input_path, verify_output_path, verify_indir_path, verify_outdir_path

# Download: ftp://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/idmapping/idmapping.dat.gz
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


def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def get_idmapping_types(path):
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
    id_types = set()
    with f, tqdm(
        dynamic_ncols=True, ascii=True, desc=path.name, unit='lines'
    ) as t:
        rows = csv.reader(f, delimiter='\t')
        for row in rows:
            t.update()
            id_types.add(row[1])
    return id_types


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Print idmapping types.')
    parser.add_argument(
        '-d',
        '--dat',
        type=str,
        required=True,
        help="Path to idmapping.dat[.gz] file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    dat_path = verify_input_path(args.dat)
    print(f'Processing: {dat_path.name}')

    id_types = get_idmapping_types(dat_path)
    # chr(10) = '\n'
    print(f'ID Types ({len(id_types)}):\n{chr(10).join(sorted(id_types))}')

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# windows
# python -m util.idmapping.list_idmapping_types --dat E:\idmapping\idmapping-20191105\idmapping.dat
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
