"""AlphaFold protein structure prediction."""

import os
import sys
import time
import random
import logging
import argparse
from tqdm import tqdm

import torch
import numpy as np
from alphafold.common import residue_constants
from alphafold.model import model as alphafold_model

from msa_pair.runner import utils, model_preset_runner

logger = logging.getLogger(__file__)


def run_alphafold(
    args,
    model_runner,
    random_seed: int,
    max_num_res: int = -1,
):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_feature_dict = dict(np.load(args.input_npz, allow_pickle=True))
    num_res = len(processed_feature_dict['aatype'])

    if max_num_res >= 0 and num_res > max_num_res:
        logger.warning(f'Too many residues {num_res}\n{args}\n')
        return

    all_outputs = model_runner.predict(
        processed_feature_dict,
        rng_seed=random_seed,
        num_predictions_per_model=args.num_predictions_per_model,
    )
    for output_name, output in all_outputs.items():
        write_output(
            args,
            processed_feature_dict,
            output,
            output_pdb=os.path.join(
                args.output_dir, f'{output_name}.pdb'
            ),
            output_ranking=os.path.join(
                args.output_dir, f'{output_name}_ranking.json'
            ),
            is_multimer = True
        )

def write_output(
    args,
    processed_feature_dict,
    output,
    output_pdb,
    output_ranking,
    is_multimer: bool = True,
):
    result = output['result']

    plddt_b_factors = np.repeat(
        result['plddt'][:, None], residue_constants.atom_type_num, axis=-1
    )

    utils.write_pdb(
        output_pdb,
        processed_feature_dict,
        result,
        plddt=plddt_b_factors,
        # mono_multimer needs to remove leading dimension
        is_multimer=is_multimer,
    )

    utils.write_ranking(
        output_ranking, result, output['timing'], random_seed=output['seed']
    )



def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-npz', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)

    parser.add_argument('--num-predictions-per-model', type=int, default=1)
    parser.add_argument('--ignore-unpaired-sequences', action='store_true')

    parser.add_argument('--verbose', action='store_true')

    parser.add_argument(
        '--data-dir', type=str,
        default=os.environ.get('ALPHAFOLD_DATABASE', None)
    )
    parser.add_argument('--device-id', type=int, default=-1)

    args = parser.parse_args()

    return args


def main():
    args = get_options()

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    if args.device_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)


    model_runner = model_preset_runner.ModelPresetRunner(
        args.data_dir,
        ignore_unpaired_sequences=args.ignore_unpaired_sequences,
    )

    seed = random.randrange(1, 2**24)
    run_alphafold(args, model_runner, seed)


if __name__ == '__main__':
    main()
