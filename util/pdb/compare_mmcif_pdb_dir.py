import os
import re
import time
import argparse
from glob import glob
from pathlib import Path

# windows
# python .\util\pdb\compare_mmcif_pdb_dir.py --mmcif D:/pdb/mmCIF-20190208 --pdb D:/pdb/pdb-20190209

if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description='Parse a single mmCIF file.')
    parser.add_argument('-m', '--mmcif', type=str, default=r'~/pdb/mmCIF',
        help="Path to mmCIF directory, default is ~/pdb/mmCIF.")
    parser.add_argument('-p', '--pdb', type=str, default=r'~/pdb/pdb',
        help="Path to mmCIF directory, default is ~/pdb/pdb.")

    args, unparsed = parser.parse_known_args()
    # print(f'mmCIF: {args.mmcif}\npdb: {args.pdb}\n')

    start_time = time.time()

    # get absolute path to dataset directory
    mmcif_path = Path(os.path.abspath(os.path.expanduser(args.mmcif)))
    pdb_path = Path(os.path.abspath(os.path.expanduser(args.pdb)))
    # print(f'mmCIF: {mmcif_path}\npdb: {pdb_path}\n')

    mmcif_files = list(mmcif_path.glob('**/*.gz'))
    pdb_files = list(pdb_path.glob('**/*.gz'))

    mmcif_count = len(mmcif_files)
    pdb_count = len(pdb_files)

    print(f'mmCIF: {mmcif_count} files\npdb: {pdb_count} files\n')
    print(f'Run time: {time.time() - start_time:.2f} s\n')

    mmcif_re = re.compile(r'(\w+).cif.gz')
    pdb_re = re.compile(r'pdb(\w+).ent.gz')

    mmcif_set = set([mmcif_re.match(f.name).group(1) for f in mmcif_files])
    pdb_set = set([pdb_re.match(f.name).group(1) for f in pdb_files])
    print(f'mmCIF - pdb: {mmcif_set - pdb_set}')
    print(f'''
mmCIF: {len(mmcif_set)} structures
pdb: {len(pdb_set)} structures
mmCIF & pdb: {len(mmcif_set & pdb_set)} structures
mmCIF | pdb: {len(mmcif_set | pdb_set)} structures
mmCIF - pdb: {len(mmcif_set - pdb_set)} structures
pdb - mmCIF: {len(pdb_set - mmcif_set)} structures
''')
    print(f'Run time: {time.time() - start_time:.2f} s\n')
    '''
    set([mmcif_re.match(f.name).group(1) for f in mmcif_path.glob('**/*.gz')])
    Run time is 3.90 secs
    set([mmcif_re.match(f.name).group(1) for f in list(mmcif_path.glob('**/*.gz'))])
    Run time is 3.34 secs
    '''

