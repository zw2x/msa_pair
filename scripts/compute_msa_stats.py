import os
import json
import tempfile
import subprocess

from tqdm import tqdm

import numpy as np
from alphafold.data import parsers, msa_identifiers

from msa_pair.data import species_processing
from msa_pair.common import sequence_utils

def calculate_log_combinations(n, m):
    return np.sum(np.log2(np.arange(n-m+1,n+1)))

def parse_msas(input_dir):
    stats = {}
    species_dict, msas_dict, _ = species_processing.parse(
        input_dir, ['uniprot.a3m'], pair_species=True
    )
    entropy = 0
    for species_key in species_dict.keys():
        rows = [len(v.msa_row) for v in species_dict[species_key].values()]
        entropy_ = calculate_log_combinations(max(rows), min(rows))
        entropy += entropy_
    num_species = len(species_dict)
    return {
        'entropy': entropy,
        'num_species': num_species,
    }

def parse_meff(input_path, meff_binary_path):
    msa = dict(np.load(input_path))['msa']
    a3m_string = '\n'.join(
        [
            f'>{i}\n{sequence_utils.aatype_to_sequence(aatype)}' for i, aatype 
            in enumerate(msa)
        ]
    )
    num_seq = len(msa)
    with tempfile.NamedTemporaryFile('w+t') as fh:
        fh.write(a3m_string)
        fh.seek(0)
        cmd = [meff_binary_path, '-i', fh.name]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, stderr = process.communicate()
        retcode = process.wait()
        if retcode:
            raise RuntimeError(stderr)
        meff = float(output.decode().strip())
    return {
        'meff': meff,
        'num_seq': num_seq,
    }

if __name__ == '__main__':
    import sys
    input_root = sys.argv[1]
    meff_binary_path = sys.argv[2]
    all_stats = {}
    for name in tqdm(os.listdir(input_root)):
        input_dir = os.path.join(input_root, name)
        species_stats = parse_msas(input_dir)
        alphafold_depths_stats = parse_meff(
            os.path.join(input_dir, 'multimer_alphafold.npz'), meff_binary_path
        )
        esm_depths_stats = parse_meff(
            os.path.join(input_dir, 'multimer_esm.npz'), meff_binary_path
        )
        all_stats[name] = {
            'species': species_stats,
            'alphafold': alphafold_depths_stats,
            'esm': esm_depths_stats
        }

    with open(sys.argv[3], 'wt') as fh:
        json.dump(all_stats, fh, indent=2, sort_keys=True)
