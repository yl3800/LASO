import os
import pandas as pd
import pickle
import math
import random
import numpy as np
from torch.utils.data import Dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


class AffordQ(Dataset):


    def __init__(self,
                 split='train',
                 **kwargs
                 ):
        data_root='../../3d_affordance/LASO_dataset'

        self.split = split
        #
        classes = ["Bag", "Bed", "Bowl","Clock", "Dishwasher", "Display", "Door", "Earphone", "Faucet",
            "Hat", "StorageFurniture", "Keyboard", "Knife", "Laptop", "Microwave", "Mug",
            "Refrigerator", "Chair", "Scissors", "Table", "TrashCan", "Vase", "Bottle"]
        
        afford_cl = ['lay','sit','support','grasp','lift','contain','open','wrap_grasp','pour', 
                     'move','display','push','pull','listen','wear','press','cut','stab']
        
        self.cls2idx = {cls.lower():np.array(i).astype(np.int64) for i, cls in enumerate(classes)}
        self.aff2idx = {cls:np.array(i).astype(np.int64) for i, cls in enumerate(afford_cl)}

        with open(os.path.join(data_root, f'anno_{split}.pkl'), 'rb') as f:
            self.anno = pickle.load(f)
        
        with open(os.path.join(data_root, f'objects_{split}.pkl'), 'rb') as f:
            self.objects = pickle.load(f)

        # Load the CSV file, and use lower case
        self.question_df = pd.read_csv(os.path.join(data_root, 'Affordance-Question.csv'))
    
        self.len = len(self.anno)
       
        print(f"load {split} set successfully, lenth {len(self.anno)}")

        # sort anno by object class and affordance type
        self.sort_anno ={}
        for item in sorted(self.anno, key=lambda x: x['class']):
            key = item['class']
            value = {'shape_id': item['shape_id'], 'mask': item['mask'], 'affordance': item['affordance']}
            
            if key not in self.sort_anno:
                self.sort_anno[key] = [value]
            else:
                self.sort_anno[key].append(value)


    def find_rephrase(self, df, object_name, affordance):
        qid = str(np.random.randint(1, 15)) if self.split == 'train' else '0'
        qid = 'Question'+qid
        result = df.loc[(df['Object'] == object_name) & (df['Affordance'] == affordance), [qid]]
        if not result.empty:
            # return result.index[0], result.iloc[0]['Rephrase']
            return result.iloc[0][qid]
        else:
            raise NotImplementedError
            
         
    def __getitem__(self, index):
        data = self.anno[index]            
        shape_id = data['shape_id']
        cls = data['class']
        affordance = data['affordance']
        gt_mask = data['mask']
        point_set = self.objects[str(shape_id)]
        point_set,_,_ = pc_normalize(point_set)
        point_set = point_set.transpose()
            
        question = self.find_rephrase(self.question_df, cls, affordance)
        affordance = self.aff2idx[affordance]

        return point_set, self.cls2idx[cls], gt_mask, question, affordance

    def __len__(self):
        return len(self.anno)


if __name__ == '__main__':
    train = AffordQ('train')
    print(len(train))
    
    for p,cls,mask,q,aff in train:
        print(q)
