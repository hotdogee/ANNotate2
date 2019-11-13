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

# Download: http://purl.obolibrary.org/obo/hp.obo
# format-version: 1.2
# data-version: hp/releases/2019-11-08
# saved-by: Peter Robinson, Sebastian Koehler, Sandra Doelken, Chris Mungall, Melissa Haendel, Nicole Vasilevsky, Monarch Initiative, et al.
# subsetdef: hposlim_core "Core clinical terminology"
# subsetdef: secondary_consequence "Consequence of a disorder in another organ system."
# synonymtypedef: HP:0031859 "obsolete synonym"
# synonymtypedef: HP:0045076 "UK spelling"
# synonymtypedef: HP:0045077 "abbreviation"
# synonymtypedef: HP:0045078 "plural form"
# synonymtypedef: layperson "layperson term"
# default-namespace: human_phenotype
# remark: Please see license of HPO at http://www.human-phenotype-ontology.org
# ontology: hp.obo
# property_value: http://purl.org/dc/elements/1.1/creator "Human Phenotype Ontology Consortium" xsd:string
# property_value: http://purl.org/dc/elements/1.1/creator "Monarch Initiative" xsd:string
# property_value: http://purl.org/dc/elements/1.1/creator "Peter Robinson" xsd:string
# property_value: http://purl.org/dc/elements/1.1/creator "Sebastian Köhler" xsd:string
# property_value: http://purl.org/dc/elements/1.1/description "The Human Phenotype Ontology (HPO) provides a standardized vocabulary of phenotypic abnormalities and clinical features encountered in human disease." xsd:string
# property_value: http://purl.org/dc/elements/1.1/license https://hpo.jax.org/app/license xsd:string
# property_value: http://purl.org/dc/elements/1.1/rights "Peter Robinson, Sebastian Koehler, The Human Phenotype Ontology Consortium, and The Monarch Initiative" xsd:string
# property_value: http://purl.org/dc/elements/1.1/subject "Phenotypic abnormalities encountered in human disease" xsd:string
# property_value: http://purl.org/dc/elements/1.1/title "Human Phenotype Ontology" xsd:string
# property_value: http://purl.org/dc/terms/license https://hpo.jax.org/app/license xsd:string
# logical-definition-view-relation: has_part

# [Term]
# id: HP:0000001
# name: All
# comment: Root of all terms in the Human Phenotype Ontology.
# xref: UMLS:C0444868

# [Term]
# id: HP:0000002
# name: Abnormality of body height
# def: "Deviation from the norm of height with respect to that which is expected according to age and gender norms." [HPO:probinson]
# synonym: "Abnormality of body height" EXACT layperson []
# xref: UMLS:C4025901
# is_a: HP:0001507 ! Growth abnormality
# created_by: peter
# creation_date: 2008-02-27T02:20:00Z


def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]


def parse_obo(path):
    # initialize
    line_num = 0
    header = True
    current_term = ''
    # return
    term_is_a = defaultdict(set)
    term_info = defaultdict(dict)
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
        total=target,
        dynamic_ncols=True,
        ascii=True,
        smoothing=0.01,
        desc=path.name,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as t:
        for line in f:
            t.update(len(line))
            if t.n > t.total:
                t.total += 2**32
            line_num += 1
            # empty line
            line = line.strip()
            if not line:
                continue
            # check out of header section
            if line == '[Term]':
                if header:
                    header = False
                continue
            if header:
                continue
            tokens = line.split(': ')
            key = tokens[0].strip()
            value = tokens[1].strip()
            if key == 'id':
                current_term = value
                term_is_a[current_term]
            else:
                term_info[current_term][key] = value
                if key == 'is_a':
                    term_is_a[current_term].add(value.split(' ! ')[0].strip())
                elif key == 'replaced_by':
                    term_is_a[value.split()[0].strip()].add(current_term)

    def walk(p_set, p_list):
        if len(p_list) == 0:
            return
        else:
            for p in p_list:
                if p not in p_set:
                    p_set.add(p)
                    walk(p_set, term_is_a[p])

    term_parents = defaultdict(set)
    for t in term_is_a:
        p_set = term_parents[t]
        walk(p_set, term_is_a[t])
    return term_is_a, term_info, term_parents


def explore_obo(path):
    # initialize
    line_num = 0
    header = True
    term_ids = set()
    keys = set()
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
        total=target,
        dynamic_ncols=True,
        ascii=True,
        smoothing=0.01,
        desc=path.name,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as t:
        for line in f:
            t.update(len(line))
            if t.n > t.total:
                t.total += 2**32
            line_num += 1
            # empty line
            line = line.strip()
            if not line:
                continue
            # check out of header section
            if line == '[Term]':
                if header:
                    header = False
                continue
            if header:
                continue
            tokens = line.split(': ')
            key = tokens[0].strip()
            value = tokens[1].strip()
            keys.add(key)
            if key == 'id':
                term_ids.add(value)
    print(f'terms: {len(term_ids)}')
    print(f'keys: {keys}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse hp.obo into term_parents.'
    )
    parser.add_argument(
        '-o',
        '--obo',
        type=str,
        required=True,
        help="Path to hp.obo[.gz] file, required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    obo_path = verify_input_path(args.obo)
    print(f'Processing: {obo_path.name}')

    # explore_obo(obo_path)
    term_is_a, term_info, term_parents = parse_obo(obo_path)
    print(
        f"root terms: {[t for t in term_is_a if len(term_is_a[t]) == 0 and term_info[t].get('is_obsolete') != 'true']}"
    )
    print(f'term_is_a: {len(term_is_a)}')
    print(f'term_info: {len(term_info)}')
    print(f'term_parents: {len(term_parents)}')

    print(f'Run time: {time.time() - start_time:.2f} s\n')

# windows
# python -m util.hpo.parse_hp_obo --obo E:\hpo\hpo-20191011\ontology\hp.obo

# terms: 14832
# keys: {'creation_date', 'alt_id', 'replaced_by', 'comment', 'def', 'xref', 'is_a', 'created_by', 'subset', 'is_obsolete', 'name', 'consider', 'synonym', 'id', 'property_value'}
# Run time: 0.31 s
