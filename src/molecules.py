import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer
from deepchem.feat.molecule_featurizers import RDKitDescriptors


def get_rdkit_descriptors(smiles, *, names=None):
    featurizer = RDKitDescriptors()
    if names is not None:
        filtered_list = [(name, fx) for name, fx in Descriptors.descList if name in names]
        featurizer.descList = filtered_list
        featurizer.descriptors = names
    return featurizer.featurize(smiles)


def largest_ring_size(mol):
    """Calculates the largest ring size in the molecule."""
    cycle_list = mol.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def penalized_logp(mol):
    """Calculates the penalized logP of a molecule.

    Source: https://github.com/google-research/google-research/blob/master/mol_dqn/chemgraph/dqn/py/molecules.py
    """
    logp = Descriptors.MolLogP(mol)
    sas = sascorer.calculateScore(mol)
    largest_ring = largest_ring_size(mol)
    cycle_score = max(largest_ring - 6, 0)
    return logp - sas - cycle_score


def penalized_qed(mol):
    """Calculates the QED of a molecule, penalized by its SAS.

    Source: https://pubs.acs.org/doi/10.1021/acscentsci.7b00572
    """
    qed = Descriptors.qed(mol)
    sas = sascorer.calculateScore(mol)
    return 5.0 * qed - sas
