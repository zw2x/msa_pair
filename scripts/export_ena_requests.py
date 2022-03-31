import os
import sys
import json
import logging
import argparse

from tqdm import tqdm

from msa_pair.data.ena.ena_pairing import EnaPairing
from msa_pair.data.ena import ena_downloader

logger = logging.getLogger(__file__)

IDMAPPINGS = 'idmappings'
ENA_REPO = 'ena_repo'

def read_include_file(include_file):
    names = set()
    with open(include_file) as fh:
        for line in fh:
            name = line.strip().split()[0]
            names.add(name)

    return names

def export_all_requests(input_root, include_file=None):
    ena_pairing = EnaPairing([IDMAPPINGS], ENA_REPO)
    if include_file is not None:
        names = read_include_file(include_file)
    else:
        names = None
    for sub_dir in sorted(os.listdir(input_root)):
        if names is not None and sub_dir not in names:
            continue
        logger.warning(f'Start processing {sub_dir}')
        input_dir = os.path.join(input_root, sub_dir)
        output_path = os.path.join(input_dir, 'ena_request.json')
        result = ena_pairing.export_paired_accessions(input_dir)
        with open(output_path, 'wt') as fh:
            json.dump(result, fh)

def download_all_requests(input_root, include_file=None):
    if include_file is not None:
        names = read_include_file(include_file)
    else:
        names = None
    for sub_dir in tqdm(sorted(os.listdir(input_root))):
        if names is not None and sub_dir not in names:
            continue
        src_file = os.path.join(input_root, sub_dir, 'ena_request.json')
        if os.path.exists(src_file):
            ena_downloader.download_requests(src_file, ENA_REPO, max_ncpu=50)

def get_options():
    parser = argparse.ArgumentParser('Download ENA data')
    parser.add_argument('input-root', type=str)
    parser.add_argument('--idmappings', type=str, help='Path to UniProt IDs')
    parser.add_argument(
        '--ena_repo', type=str, help='Repository to all ENA data'
    )

    args = parse.parse_args()
    return args

def main():
    args = get_options()

    IDMAPPINGS = args.idmappings or IDMAPPINGS
    ENA_REPO = args.ena_repo or ENA_REPO

    export_all_requests(args.input_root)
    download_all_requests(args.input_root)

if __name__ == '__main__':
    main()
