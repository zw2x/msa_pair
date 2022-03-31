import os
import logging

from tqdm import tqdm

import numpy as np
from alphafold.data import parsers

from msa_pair.data.alphafold_pipeline import AlphaFoldPipeline

logger = logging.getLogger(__file__)

def write_multimer_fasta(input_dir, chain_ids=['A', 'B']):
    output_path = os.path.join(input_dir, 'multimer.fasta')
    if not os.path.exists(output_path):
        fasta_str = ''
        for chain_id in chain_ids:
            input_a3m_path = os.path.join(input_dir, chain_id, 'uniclust30.a3m')
            with open(input_a3m_path) as fh:
                a3m_str = fh.read()
                msa = parsers.parse_a3m(a3m_str)
                chain_sequence = msa.sequences[0]
                fasta_str += f'>{chain_id}\n{chain_sequence}\n'
        with open(output_path, 'wt') as fh:
            fh.write(fasta_str)

"""Build AlphaFold-Multimer features using the default pairing method
"""
def process_batch(root_dir):
    all_names = os.listdir(root_dir)
    for name in tqdm(all_names):
        input_dir = os.path.join(root_dir, name)
        process(input_dir)

def get_dir_size(input_dir):
    total = 0
    with os.scandir(input_dir) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def process(input_dir):
    output_path = os.path.join(input_dir, 'multimer_alphafold.npz')
    if os.path.exists(output_path):
        return
    logger.info(f'Processing {input_dir}')
    write_multimer_fasta(input_dir)
    # dir_size = get_dir_size(input_dir)
    # if dir_size / 2**20 > 1000:
    #     logger.info(f'Ignore {input_dir}, {dir_size / 2**20:.4f} MB')
    #     return
    pipeline = AlphaFoldPipeline()
    np_example = pipeline.process(input_dir)
    np.savez(output_path, **np_example)

if __name__ == '__main__':
    import sys
    root_dir = sys.argv[1]
    logging.basicConfig(level=logging.INFO)
    process_batch(root_dir)
