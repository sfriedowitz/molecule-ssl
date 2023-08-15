from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer


def largest_ring_size(molecule):
    """Calculates the largest ring size in the molecule."""
    cycle_list = molecule.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def penalized_logp(molecule):
    """Calculates the penalized logP of a molecule.

    Source: https://github.com/google-research/google-research/blob/master/mol_dqn/chemgraph/dqn/py/molecules.py
    """
    log_p = Descriptors.MolLogP(molecule)
    sas_score = sascorer.calculateScore(molecule)
    largest_ring = largest_ring_size(molecule)
    cycle_score = max(largest_ring - 6, 0)
    return log_p - sas_score - cycle_score
