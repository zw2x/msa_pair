import os
import json
import multiprocessing as mp

from tqdm import tqdm

import numpy as np

from msa_pair.data import pairing_pipeline
from msa_pair.data.ena.ena_pairing import batch_process

def pair_all_rows(input_root):
    names = sorted(os.listdir(input_root))
    all_names = []
    for name in names:
        sub_dir = os.path.join(input_root, name)
        pr_path = os.path.join(sub_dir, 'ena_pr.json')
        if os.path.exists(pr_path):
            continue 
        else:
            all_names.append(name)

    names = all_names
    ncpu = min(25, len(names))
    with mp.Pool(ncpu) as pool:
        results = []
        for i in range(ncpu):
            names_ = [names[j] for j in range(i, len(names), ncpu)]
            result = pool.apply_async(batch_process, (names_, input_root, i==0))
            results.append(result)
        [ result.get() for result in results ]

def process(input_dir, src_pr_path, dst_path):
    pipeline = pairing_pipeline.PairingPipeline()

    with open(src_pr_path) as fh:
        paired_rows_dict = json.load(fh)
        for chain_id, rows in list(paired_rows_dict.items()):
            if 0 not in rows:
                paired_rows_dict[chain_id] = [0] + rows
    try:
        np_example = pipeline.process(input_dir, paired_rows_dict)
    except IOError as e:
        print(e)
        return

    np.savez(dst_path, **np_example)

if __name__ == '__main__':
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)
    
    input_root = sys.argv[1]

    pair_all_rows(input_root) 

    for name in tqdm(os.listdir(input_root)):
        input_dir = os.path.join(input_root, name)
        pr_path = os.path.join(input_dir, 'ena_pr.json')
        dst_path = os.path.join(input_dir, 'multimer_ena.npz')
        process(input_dir, pr_path, dst_path)
