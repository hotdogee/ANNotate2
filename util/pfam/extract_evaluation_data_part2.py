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


def _open_input_path(input_path, mode='rt'):
    """Return a file handle and file size from path, supports gzip
    """
    target = os.path.getsize(input_path)
    if input_path.suffix == '.gz':
        in_f = gzip.open(input_path, mode=mode, encoding='utf-8')
        target = _gzip_size(input_path)
        while target < os.path.getsize(input_path):
            # the uncompressed size can't be smaller than the compressed size, so add 4GB
            target += 2**32
    else:
        in_f = input_path.open(mode=mode, encoding='utf-8')
    return in_f, target


def _open_output_path(output_path, mode='wt'):
    """Return a file handle and file size from path, supports gzip
    """
    if output_path.suffix == '.gz':
        f = gzip.open(output_path, mode=mode, encoding='utf-8')
    else:
        f = output_path.open(mode=mode, encoding='utf-8')
    return f


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
    with f, tqdm(
        total=target, unit='bytes', dynamic_ncols=True, ascii=True
    ) as t:
        for line in f:
            t.update(len(line))
            line_num += 1
            if line_num == 1:
                continue  # skip header
            # line code
            tokens = line.strip().split()
            # vocab['UA'].add(f'{tokens[0]}')
            vocab['UA_SV'].add(f'{tokens[0]}.{tokens[1]}')
            vocab['PA'].add(f'{tokens[4]}')
            vocab['UA_SV_PA'].add((f'{tokens[0]}.{tokens[1]}', f'{tokens[4]}'))
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
    path.parent.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
    return path


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
# initial run
# python .\util\pfam\extract_evaluation_data.py --tsv31 D:/pfam/Pfam31.0/Pfam-A.regions.uniprot.tsv.gz --tsv32 D:/pfam/Pfam32.0/Pfam-A.regions.uniprot.tsv.gz --fa32 D:/pfam/Pfam32.0/uniprot.gz --oldfa D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.fa.gz --oldtsv D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.all_regions.tsv.gz --oldadd D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains.p32_regions.tsv.gz --newfa D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.fa.gz --newtsv D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains.all_regions.tsv.gz

# fix last seq bug
# python .\util\pfam\extract_evaluation_data.py --tsv31 D:/pfam/Pfam31.0/Pfam-A.regions.uniprot.tsv.gz --tsv32 D:/pfam/Pfam32.0/Pfam-A.regions.uniprot.tsv.gz --fa32 D:/pfam/Pfam32.0/uniprot.gz --oldfa D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains_2.fa.gz --oldtsv D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains_2.all_regions.tsv.gz --oldadd D:/pfam/Pfam32.0/p31_seqs_with_p32_regions_of_p31_domains_2.p32_regions.tsv.gz --newfa D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.fa.gz --newtsv D:/pfam/Pfam32.0/p32_seqs_with_p32_regions_of_p31_domains_2.all_regions.tsv.gz
# p31_seqs_with_p32_regions_of_p31_domains_2.fa.gz
# No change
# p32_seqs_with_p32_regions_of_p31_domains_2.fa.gz
# >Z9K8E8.1 Z9K8E8_9GAMM Uncharacterized protein {ECO:0000313|EMBL:EWS99547.1} (Fragment)
# GAVFYALVRTFADLVSDNRDEVIAKHFKLIHRLLFWLDWLPARITSFGYLVIGNFNKGTSCWLHYVFDFSSPNHKVVTYTALAAEQVEERYYGCTFESLLV

if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description='Output non overlapping regions data tsv and fa files.'
    )
    parser.add_argument(
        '-31',
        '--tsv31',
        type=str,
        required=True,
        help="Path to the Pfam31.0/Pfam-A.regions.uniprot.tsv.gz file, required."
    )
    parser.add_argument(
        '-32',
        '--tsv32',
        type=str,
        required=True,
        help="Path to the Pfam32.0/Pfam-A.regions.uniprot.tsv.gz file, required."
    )
    parser.add_argument(
        '-f',
        '--fa32',
        type=str,
        required=True,
        help="Path to the Pfam32.0/uniprot.gz file, required."
    )
    parser.add_argument(
        '-nf',
        '--nofa',
        type=str,
        required=True,
        help=
        "Path to output p31_seqs_with_p32_regions_of_p31_domains_non_overlap.fa[.gz], required."
    )
    parser.add_argument(
        '-na',
        '--noalltsv',
        type=str,
        required=True,
        help=
        "Path to output p31_seqs_with_p32_regions_of_p31_domains_non_overlap.all_regions.tsv[.gz], required."
    )
    parser.add_argument(
        '-nn',
        '--nonewtsv',
        type=str,
        required=True,
        help=
        "Path to output p31_seqs_with_p32_regions_of_p31_domains_non_overlap.new_regions.tsv[.gz], required."
    )
    args, unparsed = parser.parse_known_args()
    start_time = time.time()

    tsv31_path = verify_input_path(args.tsv31)
    tsv32_path = verify_input_path(args.tsv32)
    fa32_path = verify_input_path(args.fa32)

    # p31_seqs_with_p32_regions_of_p31_domains_non_overlap.fa.gz
    nofa_path = verify_output_path(args.nofa)
    # p31_seqs_with_p32_regions_of_p31_domains_non_overlap.all_regions.tsv.gz
    noalltsv_path = verify_output_path(args.noalltsv)
    # p31_seqs_with_p32_regions_of_p31_domains_non_overlap.new_regions.tsv.gz
    nonewtsv_path = verify_output_path(args.nonewtsv)

    print(
        f"Comparing: {'/'.join(tsv31_path.parts[-2:])} and {'/'.join(tsv32_path.parts[-2:])}"
    )
    # Index Pfam31.0 regions into p31[ua_sv][(start, end),]
    # A0A103YDS2  1  3C510D74FCA8479C 31e3d17a0f6766e2a545214e475d72a8 PF04863   52   107
    p31 = defaultdict(list)
    p31_domains = set()
    line_num = 0
    tsv31_f, target = _open_input_path(tsv31_path)
    with tsv31_f as f, tqdm(
        total=target,
        unit='bytes',
        dynamic_ncols=True,
        ascii=True,
        desc='Pfam31 Indexing'
    ) as t:
        for line in f:
            t.update(len(line))
            line_num += 1
            if line_num == 1:
                continue  # skip header
            # line code
            tokens = line.strip().split()
            ua_sv = f'{tokens[0]}.{tokens[1]}'
            start = int(tokens[5])
            end = int(tokens[6])
            p31[ua_sv].append((start, end))
            pa = tokens[4]
            p31_domains.add(pa)
    print(f'Indexed {line_num} lines in {tsv31_path.name}')

    # Parse Pfam32.0 regions line by line
    # output p31_seqs_with_p32_regions_of_p31_domains_non_overlap.new_regions.tsv.gz
    seqs_with_non_overlapping_regions = set()
    tsv32_f, target = _open_input_path(tsv32_path)
    nonewtsv_f = _open_output_path(nonewtsv_path)

    with tsv32_f as f, nonewtsv_f, tqdm(
        total=target,
        unit='bytes',
        dynamic_ncols=True,
        ascii=True,
        desc='Pfam32 Pass 1'
    ) as t:
        for line in f:
            t.update(len(line))
            line_num += 1
            if line_num == 1:
                nonewtsv_f.write(line)
                continue  # skip header
            # line code
            tokens = line.strip().split()
            ua_sv = f'{tokens[0]}.{tokens[1]}'
            pa = tokens[4]
            # if sequence is an old sequence
            if ua_sv in p31 and pa in p31_domains:
                overlap = False
                # get start and end of this new region
                start = int(tokens[5])
                end = int(tokens[6])
                # loop over start and end of all old regions
                for s, e in p31[ua_sv]:
                    # (2,4) (5,7)
                    # (2,4) (4,7)
                    # (2,4) (3,7)
                    # (2,4) (1,2)
                    # (2,4) (1,5)
                    # compute overlap
                    if not ((start < s and end < s) or (start > e and end > e)):
                        # not no overlap
                        overlap = True
                        break
                if not overlap:
                    # add seq and write region to nonewtsv_f
                    seqs_with_non_overlapping_regions.add(ua_sv)
                    nonewtsv_f.write(line)

    # output p31_seqs_with_p32_regions_of_p31_domains_non_overlap.all_regions.tsv.gz
    tsv32_f, target = _open_input_path(tsv32_path)
    noalltsv_f = _open_output_path(noalltsv_path)

    with tsv32_f as f, noalltsv_f, tqdm(
        total=target,
        unit='bytes',
        dynamic_ncols=True,
        ascii=True,
        desc='Pfam32 Pass 2'
    ) as t:
        for line in f:
            t.update(len(line))
            line_num += 1
            if line_num == 1:
                noalltsv_f.write(line)
                continue  # skip header
            # line code
            tokens = line.strip().split()
            ua_sv = f'{tokens[0]}.{tokens[1]}'
            pa = tokens[4]
            if ua_sv in seqs_with_non_overlapping_regions:
                noalltsv_f.write(line)

    # output p31_seqs_with_p32_regions_of_p31_domains_non_overlap.fa.gz
    fa32_f, target = _open_input_path(fa32_path)
    nofa_f = _open_output_path(nofa_path)

    with fa32_f as f, nofa_f, tqdm(
        total=target,
        unit='bytes',
        dynamic_ncols=True,
        ascii=True,
        desc='Pfam32 fasta'
    ) as t:
        seq_id, seq_entry = '', ''
        line_num = 0
        for line in fa32_f:
            # if target < fa32_f.tell():
            #     target += 2**32
            #     t.total = target
            t.update(len(line))
            line_num += 1
            line_s = line.strip()
            if len(line_s) > 0 and line_s[0] == '>':
                if seq_entry:
                    if seq_id in seqs_with_non_overlapping_regions:
                        nofa_f.write(seq_entry)
                    seq_id, seq_entry = '', ''
                # parse header
                seq_id = line_s.split()[0][1:]
            seq_entry += line
        if seq_entry:  # handle last seq
            if seq_id in seqs_with_non_overlapping_regions:
                nofa_f.write(seq_entry)

    print(f'Runtime: {time.time() - start_time:.2f} s\n')

# Mon Oct 07 23:00:53-hotdogee@W2125$ python ./util/pfam/extract_evaluation_data_part2.py --tsv31 /home/hotdogee/datasets3/Pfam31.0/Pfam-A.regions.uniprot.tsv --tsv32 /home/hotdogee/datasets3/Pfam32.0/Pfam-A.regions.uniprot.tsv.gz --fa32 /home/hotdogee/datasets3/Pfam32.0/uniprot.gz --nofa /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_non_overlap.fa --noalltsv /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_non_overlap.all_regions.tsv --nonewtsv /home/hotdogee/pfam/p31_seqs_with_p32_regions_of_p31_domains_non_overlap.new_regions.tsv
