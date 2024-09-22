'''
Author: QHGG
Date: 2021-11-03 22:40:24
LastEditTime: 2022-08-22 18:21:54
LastEditors: QHGG
Description: dataloader with coords
FilePath: /AlphaDrug/utils/baseline.py
'''

import json
import re
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from scipy import stats
from easydict import EasyDict
import selfies as sf
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    
    def __init__(self, data, embedding_path):
        self.drug_embeddings = np.load('drug_embeddings_96dim.npy')

        proIndices, selIndices, labelIndices, proMask, selMask = data
        self._len = len(proIndices)
        self.x = proIndices
        self.y = selIndices
        self.label = labelIndices
        self.proMask = proMask
        self.selMask = selMask
    
    def __getitem__(self, idx):
        drug_embedding = self.drug_embeddings[self.y[idx]]
        proMask = [1.0] * self.proMask[idx] + [0.0] * (len(self.x[idx]) - self.proMask[idx])
        selMask = [1.0] * self.selMask[idx] + [0.0] * (len(self.label[idx]) - self.selMask[idx])
        
        return self.x[idx], self.y[idx], self.label[idx], np.array(proMask).astype(int), \
        np.array(selMask).astype(int)

    def __len__(self):
        return self._len

def prepareDataset(config, embedding_path):
    train, valid = prepareData(config)

    trainLoader = DataLoader(MyDataset(train, embedding_path=embedding_path), shuffle=True, batch_size=config.batchSize, drop_last=False)
    validLoader = DataLoader(MyDataset(valid, embedding_path=embedding_path), shuffle=False, batch_size=config.batchSize, drop_last=False)

    return trainLoader, validLoader

def padPocCoords(Coords, MaxLen):
    return [[0.0, 0.0, 0.0]] + Coords + (MaxLen - 1 - len(Coords)) * [[0.0,0.0,0.0]]

def padLabelPocCoords(Coords, MaxLen):
    return Coords + (MaxLen - len(Coords)) * [[0.0,0.0,0.0]]

def selfiesCoordsMask(mask, MaxLen):
    return mask + (MaxLen - len(mask)) * [0]

# def readBindingDB(PATH):
#     i = 0
#     n = 0
#     pdbidArr = []
#     pocSeqArr = []
#     selArr = []
#     affinityArr = []
#     with open(PATH, 'r') as f:
#         for lines in tqdm(f.readlines()):
#             n+=1
#             arr = lines.split(' ')
#             pocSeq = arr[0]
#             print(lines.split('\t'))
#             sel = arr[1][:-1]
#             try:
#                 mol = Chem.MolFromSmiles(sel)
#                 mol = Chem.RemoveHs(mol, sanitize=False)
#                 sel = Chem.MolToSmiles(mol)
#                 if '%' in sel:
#                     continue
#                 if '.' in sel:
#                     continue
#             except:
#                 continue
#
#             pdbidArr.append('xxxx')
#             pocSeqArr.append(pocSeq)
#             selArr.append(sel)
#             affinityArr.append(0.0)
#
#             i+=1
#
#
#     print(i, n, i / n)
#
#     data = pd.DataFrame({
#         'pdbid': pdbidArr,
#         'protein': pocSeqArr,
#         'smile': selArr,
#         'affinity': affinityArr,
#     })
#     return data

def prepareData(config):

    data = pd.read_csv('./data/small_with_selfies.tsv', sep = '\t')
    # 小样本测试
    # print(slices)
    # slices = slices[:10]

    # Randomly split the data into train and validation sets
    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)

    # Process training data
    logger.info('Processing training data')
    selArr_train = train_data['selfies'].apply(splitSelfies).tolist()
    proArr_train = train_data['protein'].apply(list).tolist()

    selIndices_train, labelIndices_train, selMask_train = fetchIndices(selArr_train, config.selVoc, config.selMaxLen)
    proIndices_train, _, proMask_train = fetchIndices(proArr_train, config.proVoc, config.proMaxLen)

    # Process validation data
    logger.info('Processing validation data')
    selArr_val = val_data['selfies'].apply(splitSelfies).tolist()
    proArr_val = val_data['protein'].apply(list).tolist()

    selIndices_val, labelIndices_val, selMask_val = fetchIndices(selArr_val, config.selVoc, config.selMaxLen)
    proIndices_val, _, proMask_val = fetchIndices(proArr_val, config.proVoc, config.proMaxLen)

    selIndices_train = torch.tensor(selIndices_train, dtype=torch.long)
    labelIndices_train = torch.tensor(labelIndices_train, dtype=torch.long)
    selMask_train = torch.tensor(selMask_train, dtype=torch.long)
    proIndices_train = torch.tensor(proIndices_train, dtype=torch.long)
    proMask_train = torch.tensor(proMask_train, dtype=torch.long)
    labelIndices_val = torch.tensor(labelIndices_val, dtype=torch.long)
    # Return both training and validation data
    train = (proIndices_train, selIndices_train, labelIndices_train, proMask_train, selMask_train)
    valid = (proIndices_val, selIndices_val, labelIndices_val, proMask_val, selMask_val)

    return train, valid

def loadConfig(args):
    logger.info('prepare data config...')
    data = pd.read_csv('./data/small_with_selfies.tsv', sep = '\t')
    
    proMaxLen = max(list(data['protein'].apply(len))) + 2
    selMaxLen = max(list(data['selfies'].apply(splitSelfies).apply(len))) + 2

    pros_split = data['protein'].apply(list)
    proVoc = sorted(list(set([i for j in pros_split for i in j])) + ['&', '$', '^'])

    selfies_split = data['selfies'].apply(splitSelfies)
    selVoc = sorted(list(set([i for j in selfies_split for i in j])) + ['&', '$', '^'])

    return EasyDict({
        'proMaxLen': proMaxLen,
        'selMaxLen': selMaxLen,
        'proVoc': proVoc,
        'selVoc': selVoc,
        'args': args
    })

def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''
    x = np.array(x)
    y = np.array(y)
    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi
    
def splitSmi(smi):
    '''
    description: 将smiles拆解为最小单元
    param {*} smi
    return {*}
    '''
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens


def splitSelfies(selfie):
    '''
    description: Decomposes a SELFIES string into its smallest components (tokens).
    param {*} selfie: A string in SELFIES format.
    return {*}: A list of tokens extracted from the SELFIES string.
    '''
    # Check if the input is a valid SELFIES string


    # Use regex to split SELFIES string based on bracketed groups
    pattern = r'(\[[^\]]+\])'  # Pattern to match each [token]
    tokens = re.findall(pattern, selfie)
    tokens_without_brackets = [token.strip('[]') for token in tokens]
    # Reconstruct to check the correctness of the tokenization
    reconstructed_selfie = ''.join(tokens)
    assert selfie == reconstructed_selfie, "The tokenized SELFIES does not match the original."

    return tokens_without_brackets


def fetchIndices(selArr, selVoc, selMaxLen):
    selIndices = []
    labelIndices = []
    mask = []
    # padding symbol: ^ ; end symbol: $ ; start symbol: &
    for sel in tqdm(selArr):
        selSplit = sel[:]
        selSplit.insert(0, '&')
        selSplit.append('$')

        labelSel = selSplit[1:]
        mask.append(len(selSplit))

        selSplit.extend(['^'] * (selMaxLen - len(selSplit)))
        selIndices.append([selVoc.index(sel) for sel in selSplit])

        labelSel.extend(['^'] * (selMaxLen - len(labelSel)))
        labelIndices.append([selVoc.index(sel) for sel in labelSel])

    return np.array(selIndices), np.array(labelIndices), np.array(mask)



    
if __name__ == '__main__':
    pass

    
