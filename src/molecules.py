from typing import Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from deepchem.feat.molecule_featurizers import RDKitDescriptors
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer


def compressed_rdkit_descriptors(
    mols,
    *,
    n_components: Optional[int] = None,
    use_fragment: bool = False,
    scale: bool = True,
):
    calc = RDKitDescriptors(use_fragment=use_fragment)
    descriptors = calc.featurize(mols)
    if n_components:
        descriptors = PCA(n_components=n_components).fit_transform(descriptors)
    if scale:
        descriptors = StandardScaler().fit_transform(descriptors)
    return descriptors


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
    log_p = Descriptors.MolLogP(mol)
    sas_score = sascorer.calculateScore(mol)
    largest_ring = largest_ring_size(mol)
    cycle_score = max(largest_ring - 6, 0)
    return log_p - sas_score - cycle_score
