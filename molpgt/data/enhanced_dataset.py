from functools import lru_cache

import numpy as np
import torch
import pickle
from typing import List
from rdkit import Chem
from unicore.data import Dictionary
from functools import lru_cache
from unicore.data import BaseWrapperDataset

ele2emb = pickle.load(open('kgembedding/ele2emb.pkl','rb'))
def ele_features(ele):
    fele = ele2emb[ele]
    return fele.tolist()

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

class KnowledgeTokenizeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dictionary: Dictionary,
    ):
        self.dataset = dataset
        self.dictionary = dictionary

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        atomic_nums = []
        raw_data = self.dataset[index]["atoms"]
        smiles = self.dataset[index]["smi"]
        mol = Chem.MolFromSmiles(smiles)
        atoms = mol.GetAtoms()
        type_to_num = {}
        for atom in atoms:
            atomsb = atom.GetSymbol()
            type_to_num[atomsb] = atom.GetAtomicNum()
        f_eles = []
        for i, atom in enumerate(raw_data):
            atomicnum = type_to_num[atom]
            atomic_nums.append(atomicnum)
            f_eles.append(ele_features(atomicnum))
        dict_atoms = self.dictionary.vec_index(raw_data)

        f_atoms = []
        for i in dict_atoms:
            f_atoms.append(onek_encoding_unk(i, list(range(self.dictionary.__len__()))))
        assert len(f_atoms) == len(f_eles)
        f_cl = []
        for i in range(len(raw_data)):
            f_cl.append(f_atoms[i] + f_eles[i]) #31+133
            assert len(f_atoms[i] + f_eles[i]) == 164
        f_cl = np.array(f_cl).astype(np.float32)
        f_cl = torch.from_numpy(f_cl)
        self.dataset[index]["atoms"] = f_cl
        return self.dataset[index]