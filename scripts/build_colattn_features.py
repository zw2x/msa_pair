import os
import json

from tqdm import tqdm
import numpy as np

from msa_pair.data import (
    species_processing, row_processing, pairing_pipeline,
)

def compute_scores(input_dir, dst_path):
    from msa_pair.data import esm_scoring

    species_dict, msas_dict, _, _ = species_processing.pair_species(input_dir)

    esm_scorer = esm_scoring.EsmScoring()
    sequences_scores = esm_scorer.score_sequences(species_dict, msas_dict)
    with open(dst_path, 'wt') as fh:
        json.dump(sequences_scores, fh, indent=2, sort_keys=True)


def pair_rows(input_dir, src_score_path, dst_pr_path, overwrite=False):
    if overwrite or not os.path.exists(dst_pr_path):
        species_dict, msas_dict, _, _ = species_processing.pair_species(
            input_dir
        )

        with open(src_score_path) as fh:
            sequences_scores = json.load(fh)

        paired_rows_dict = row_processing.create_paired_rows_dict(
            species_dict, msas_dict, sequences_scores
        )

        with open(dst_pr_path, 'wt') as fh:
            json.dump(paired_rows_dict, fh, indent=2)


def process(input_dir, src_pr_path, dst_path, overwrite=False):
    if not overwrite and os.path.exists(dst_path):
        return

    pipeline = pairing_pipeline.PairingPipeline()

    with open(src_pr_path) as fh:
        paired_rows_dict = json.load(fh)

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
    for name in tqdm(os.listdir(input_root)):
        input_dir = os.path.join(input_root, name)
        score_path = os.path.join(input_dir, 'esm_scores.json')
        compute_scores(input_dir, score_path)

        pr_path = os.path.join(input_dir, 'esm_pr.json')
        pair_rows(input_dir, score_path, pr_path)

        dst_path = os.path.join(input_dir, 'multimer.npz')
        process(input_dir, pr_path, dst_path)
