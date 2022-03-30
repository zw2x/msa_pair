import os
import sys
import copy
import json
import logging
import argparse
import multiprocessing as mp
from tqdm import tqdm

from msa_pair.assess import multimer_assess, monomer_assess

logger = logging.getLogger(__file__)

class AssessModel:
    def __init__(self, dockq_exec_path, tm_align_binary_path):
        self.assess = multimer_assess.MultimerAssess(dockq_exec_path)
        self.mono_assess = monomer_assess.MonomerAssess(tm_align_binary_path)

    def assess_monomer(self, query_pdb, truth_pdb, dst_dir):
        output_pdb_json = os.path.join(dst_dir, f'assess_mono.json')

        results = self.mono_assess.assess(query_pdb, truth_pdb)

        if len(results) > 0:
            with open(output_pdb_json, 'wt') as fh:
                json.dump(results, fh, indent=2)
        else:
            logger.warning(f'Empty results in name {query_pdb} {truth_pdb}')

    def assess_multimer(self, query_pdb, truth_pdb, dst_dir):

        output_pdb_json = os.path.join(dst_dir, f'assess.json')
        if os.path.exists(output_pdb_json):
            return

        output_pdb_dir = os.path.join(dst_dir, 'output')
        if not os.path.exists(output_pdb_dir):
            os.makedirs(output_pdb_dir)

        gt_prefix = os.path.join(output_pdb_dir, 'gt')
        # Get all ground truth models
        gt_paths = self.assess.build_models(truth_pdb, gt_prefix)

        final_results = {}
        for model_id, gt_path in gt_paths.items():
            # Align residue index of query pdb to a ground truth model
            results = self.assess.align_residue_index(
                query_pdb, gt_path, output_pdb_dir, f'gt_{model_id}'
            )
            if results is None:
                continue
            best_score_ = 0
            best_result_ = None
            # Compute the DockQ scores for all possible alignments
            for chain, output_pdb in results:
                dockq_result = self.assess.assess(
                    output_pdb, gt_path, chain
                )
                if dockq_result is None:
                    continue
                dockq_result['receptor_chain'] = chain
                dockq_result['query_pdb'] = os.path.basename(output_pdb)
                if dockq_result['dockq'] > best_score_:
                    best_result_ = dockq_result
                    best_score_ = dockq_result['dockq']

            if best_result_ is not None:
                final_results[model_id] = best_result_

        if len(final_results) > 0:
            best_score = 0
            best_model_id = None
            for model_id, result in final_results.items():
                if result['dockq'] > best_score:
                    best_model_id = model_id
                    best_score = result['dockq']
            # Find the model with the best score
            final_results = {
                'best': {
                    'score': best_score,
                    'model_id': best_model_id,
                },
                'results': final_results
            }
            with open(output_pdb_json, 'wt') as fh:
                json.dump(final_results, fh, indent=2)
        else:
            logger.warning(f'No results for {query_pdb}')

def process(args):
    assess_model = AssessModel(args.dockq_path, args.tmalign_path)
    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)

    def _assess_multimer():
        assess_model.assess_multimer(
            args.query, args.ground_truth, args.dst_dir,
        )

    def _assess_monomer():
        input_pdb_json = os.path.join(args.dst_dir, f'assess.json')
        with open(input_pdb_json) as fh:
            assess_data = json.load(fh)
        # Find the best model
        model_id = assess_data['best']['model_id']
        query_pdb = assess_data['results'][str(model_id)]['query_pdb']
        truth_pdb = query_pdb.split('to')[-1].strip('_')
        assess_model.assess_monomer(
            os.path.join(args.dst_dir, 'output', query_pdb),
            os.path.join(args.dst_dir, 'output', truth_pdb),
            args.dst_dir
        )


    _assess_multimer()
    _assess_monomer()


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth', type=str, help='Ground truth PDB')
    parser.add_argument('--query', type=str, help='Query PDB')
    parser.add_argument('--dst-dir', type=str, help='destination directory')
    parser.add_argument(
        '--tmalign-path', type=str, help='The path to the TM-align binary'
    )
    parser.add_argument(
        '--dockq-path', type=str, help='The path to DockQ.py'
    )

    args = parser.parse_args()

    return args


def main():
    args = get_options()
    process(args)

if __name__ == '__main__':
    main()
