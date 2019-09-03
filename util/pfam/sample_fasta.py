import os
import re
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

def _gzip_size(path):
    """Uncompressed size is stored in the last 4 bytes of the gzip file
    """
    with open(path, 'rb') as f:
        f.seek(-4, 2)
        return struct.unpack('I', f.read(4))[0]

def stats_pfam_regions_tsv(path):
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
    vocab = defaultdict(set)
    line_num = 0
    with f, tqdm(total=target, unit='bytes', dynamic_ncols=True, ascii=True) as t:
        for line in f:
            t.update(len(line))
            line_num += 1
            if line_num == 1: continue # skip header
            # line code
            tokens = line.strip().split()
            # vocab['UA'].add(f'{tokens[0]}')
            vocab['UA_SV'].add(f'{tokens[0]}.{tokens[1]}')
            vocab['PA'].add(f'{tokens[4]}')
            vocab['UA_SV_PA'].add((f'{tokens[0]}.{tokens[1]}',f'{tokens[4]}'))
            # vocab['UA_SV_PA_SE'].add((f'{tokens[0]}.{tokens[1]}',f'{tokens[4]}',f'{tokens[5]}.{tokens[6]}'))
    return vocab, line_num

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
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.EEXIST), path)
    # is dir
    if path.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
    # assert dirs
    path.parent.mkdir(parents=True, exist_ok=True) # pylint: disable=no-member
    return path

def print_set_stats(n1, s1, n2, s2, unit=''):
    print(f'''
{n1}: {len(s1)} {unit}
{n2}: {len(s2)} {unit}
{n1} & {n2}: {len(s1 & s2)} {unit}
{n1} | {n2}: {len(s1 | s2)} {unit}
{n1} - {n2}: {len(s1 - s2)} {unit}
{n2} - {n1}: {len(s2 - s1)} {unit}
''')

# uniprot_acc seq_version crc64            md5                              pfamA_acc seq_start seq_end
# A0A103YDS2  1           3C510D74FCA8479C 31e3d17a0f6766e2a545214e475d72a8 PF04863   52        107
# A0A087Y7L3  1           E79AF82467BBFA7D bb8465c4f3cd34f56f6c60817d43f551 PF09014   261       348
# A0A103YDS2  1           3C510D74FCA8479C 31e3d17a0f6766e2a545214e475d72a8 PF04863   433       488
# A0A103YDS2  1           3C510D74FCA8479C 31e3d17a0f6766e2a545214e475d72a8 PF04863   870       925
# A0A199V6R7  1           F37C9E162DD24DD7 0656490991e5645a3e2b402a861e450e PF04863   66        120
# A0A199V6R7  1           F37C9E162DD24DD7 0656490991e5645a3e2b402a861e450e PF04863   468       522
# A0A0D9V4Z5  1           25C8FDA89273F894 1ddc7bbbfec83e195473a9e51b763b16 PF04863   56        111
# W5N889      1           509B93CB8368603C 0e92ab8939197d40e00e0238de6b8a69 PF09014   264       351
# A0A0D9V4Z5  1           25C8FDA89273F894 1ddc7bbbfec83e195473a9e51b763b16 PF04863   552       607

# set stats:
# * UA: uniprot_acc
# * UA_SV: uniprot_acc + seq_version
# * PA: pfamA_acc
# * UA_SV_PA: uniprot_acc + seq_version + pfamA_acc
# * UA_SV_PA_SE: uniprot_acc + seq_version + pfamA_acc + seq_start + seq_end

# /Pfam31.0 88M
# $ wc -l Pfam-A.regions.uniprot.tsv
# 88761543 Pfam-A.regions.uniprot.tsv
# /Pfam32.0 138M
# $ wc -l Pfam-A.regions.uniprot.tsv
# 138165284 Pfam-A.regions.uniprot.tsv

# windows
# python .\util\pfam\extract_evaluation_data.py --tsv31 D:/pfam/Pfam31.0/Pfam-A.regions.uniprot.tsv.gz --tsv32 D:/pfam/Pfam32.0/Pfam-A.regions.uniprot.tsv.gz --fa32 D:/pfam/Pfam32.0/uniprot.gz --oldfa D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --oldtsv D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv.gz --oldadd D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv.gz --newfa D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.fa.gz --newtsv D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv.gz

# ubuntu 18.04
# $ cd Downloads
# $ tar zxf hmmer-3.1b2.tar.gz
# $ cd hmmer-3.1b2/
# $ ./configure --prefix /usr/local
# $ make
# $ sudo make install
# $ cd easel/
# $ sudo make install
# $ cd ../..
# $ sudo apt install -y libmoose-perl bioperl
# $ tar zxf PfamScan.tar.gz
# $ sudo mv ./PfamScan /opt
# $ cd ~
# $ mkdir pfam
# $ cd ~/pfam
# copy files over
# $ hmmpress Pfam-A.hmm
# $ export PERL5LIB=/opt/PfamScan:$PERL5LIB
# /opt/PfamScan/pfam_scan.pl -fasta p31_seqs_with_p32_regions_of_p31_domains.fa -dir ./ -outfile ./p31_seqs_with_p32_regions_of_p31_domains.p31result.tsv
# /opt/PfamScan/pfam_scan.pl -fasta p32_seqs_with_p32_regions_of_p31_domains.fa -dir ./ -outfile ./p32_seqs_with_p32_regions_of_p31_domains.p31result.tsv
# /usr/bin/time -v -o test.p31result.tsv.time /opt/PfamScan/pfam_scan.pl -fasta test.fa -dir ./ -outfile ./test.p31result.tsv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sample N sequences from a FASTA file.')
    parser.add_argument('-i', '--input', type=str, required=True,
        help="Path to the input FASTA file, required.")
    parser.add_argument('-o', '--output', type=str, required=True,
        help="Path to the output FASTA file, required.")
    parser.add_argument('-of', '--oldfa', type=str, required=True,
        help="Path to output old.fa.gz, required.")
    parser.add_argument('-ot', '--oldtsv', type=str, required=True,
        help="Path to output old.tsv.gz, required.")
    parser.add_argument('-oa', '--oldadd', type=str, required=True,
        help="Path to output old_addition_only.tsv.gz, required.")
    parser.add_argument('-nf', '--newfa', type=str, required=True,
        help="Path to output new.fa.gz, required.")
    parser.add_argument('-nt', '--newtsv', type=str, required=True,
        help="Path to output new.tsv.gz, required.")
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    tsv31_path = verify_input_path(args.tsv31)
    tsv32_path = verify_input_path(args.tsv32)
    fa32_path = verify_input_path(args.fa32)
    print(f"Comparing: {'/'.join(tsv31_path.parts[-2:])} and {'/'.join(tsv32_path.parts[-2:])}")

    oldfa_path = verify_output_path(args.oldfa) # p31_seqs_with_p32_regions_of_p31_domains.fa
    oldtsv_path = verify_output_path(args.oldtsv) # p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv
    oldadd_path = verify_output_path(args.oldadd) # p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv
    newfa_path = verify_output_path(args.newfa) # p32_seqs_with_p32_regions_of_p31_domains.fa
    newtsv_path = verify_output_path(args.newtsv) # p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv

    tsv31_vocab, tsv31_line_num = stats_pfam_regions_tsv(tsv31_path)
    print(f'Processed {tsv31_line_num} lines in {tsv31_path.name}')
    tsv32_vocab, tsv32_line_num = stats_pfam_regions_tsv(tsv32_path)
    print(f'Processed {tsv32_line_num} lines in {tsv32_path.name}')

    # Obtain new regions of old domains on old sequences that do not overlap with old regions
    # UA_SV_PA(Pfam32 - Pfam31), PA(Pfam31), UA_SV(Pfam31)
    # Output datasets:
    # old.fa
    # old.tsv
    # old_addition_only.tsv
    # new.fa
    # new.tsv
    new_regions = tsv32_vocab['UA_SV_PA'] - tsv31_vocab['UA_SV_PA']
    del tsv31_vocab['UA_SV_PA']
    del tsv32_vocab['UA_SV_PA']
    in_old = defaultdict(set)
    in_new = defaultdict(set)
    for r in new_regions:
        if r[0] in tsv31_vocab['UA_SV'] and r[1] in tsv31_vocab['PA']:
            in_old['UA_SV_PA'].add(r)
            in_old['UA_SV'].add(r[0])
        elif r[0] in tsv32_vocab['UA_SV'] and r[1] in tsv31_vocab['PA']:
            # in_new['UA_SV_PA'].add(r)
            in_new['UA_SV'].add(r[0])
    del tsv31_vocab['UA_SV']
    del tsv32_vocab['UA_SV']
    del tsv31_vocab['PA']
    del tsv32_vocab['PA']
    print(f'''
in_old['UA_SV_PA']: {len(in_old['UA_SV_PA'])}
in_old['UA_SV']: {len(in_old['UA_SV'])}
in_new['UA_SV']: {len(in_new['UA_SV'])}
''')

    # write tsv
    write_tsv = False
    if write_tsv:
        oldtsv_f = gzip.open(oldtsv_path, mode='wt', encoding='utf-8')
        oldadd_f = gzip.open(oldadd_path, mode='wt', encoding='utf-8')
        newtsv_f = gzip.open(newtsv_path, mode='wt', encoding='utf-8')
        tsv32_f = gzip.open(tsv32_path, mode='rt', encoding='utf-8')
        target = _gzip_size(tsv32_path)
        while target < os.path.getsize(tsv32_path):
            # the uncompressed size can't be smaller than the compressed size, so add 4GB
            target += 2**32
        # initialize
        line_num = 0
        with tsv32_f, oldtsv_f, oldadd_f, newtsv_f, tqdm(total=target, unit='bytes', dynamic_ncols=True, ascii=True) as t:
            for line in tsv32_f:
                # if target < tsv32_f.tell(): # OSError: telling position disabled by next() call
                #     target += 2**32
                #     t.total = target
                t.update(len(line))
                line_num += 1
                if line_num == 1: # header
                    oldtsv_f.write(line)
                    oldadd_f.write(line)
                    newtsv_f.write(line)
                    continue
                # parse
                tokens = line.strip().split()
                ua_sv = f'{tokens[0]}.{tokens[1]}'
                if (ua_sv,f'{tokens[4]}') in in_old['UA_SV_PA']:
                    oldadd_f.write(line)
                if ua_sv in in_old['UA_SV']:
                    oldtsv_f.write(line)
                if ua_sv in in_new['UA_SV']:
                    newtsv_f.write(line)

    # write fasta
    oldfa_f = gzip.open(oldfa_path, mode='wt', encoding='utf-8')
    newfa_f = gzip.open(newfa_path, mode='wt', encoding='utf-8')
    fa32_f = gzip.open(fa32_path, mode='rt', encoding='utf-8')
    target = _gzip_size(fa32_path)
    while target < os.path.getsize(fa32_path):
        # the uncompressed size can't be smaller than the compressed size, so add 4GB
        target += 2**32
    # initialize
    seq_id, seq_entry = '', ''
    line_num = 0
    with fa32_f, oldfa_f, newfa_f, tqdm(total=target, unit='bytes', dynamic_ncols=True, ascii=True) as t:
        for line in fa32_f:
            # if target < fa32_f.tell():
            #     target += 2**32
            #     t.total = target
            t.update(len(line))
            line_num += 1
            line_s = line.strip()
            if len(line_s) > 0 and line_s[0] == '>':
                if seq_entry:
                    if seq_id in in_old['UA_SV']:
                        oldfa_f.write(seq_entry)
                    if seq_id in in_new['UA_SV']:
                        newfa_f.write(seq_entry)
                    seq_id, seq_entry = '', ''
                # parse header
                seq_id = line_s.split()[0][1:]
            seq_entry += line

    print(f'Runtime: {time.time() - start_time:.2f} s\n')

# Processed 88761543 lines in Pfam-A.regions.uniprot.tsv.gz
# 10643100349bytes [13:33, 13078187.05bytes/s]
# Processed 138165284 lines in Pfam-A.regions.uniprot.tsv.gz

# in_old['UA_SV_PA']: 222373
# in_old['UA_SV']: 219236
# in_new['UA_SV']: 38168103

# 10643100349bytes [16:10, 10971842.48bytes/s]
# 49586165246bytes [28:55, 28575875.42bytes/s]
# Runtime: 4332.06 s (1950X)
