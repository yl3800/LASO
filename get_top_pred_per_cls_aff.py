import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from LASO.model.PointRefer import get_IAGNet
from utils.util import seed_torch, read_yaml
from data_utils.shapenetpart import AffordQ
from tqdm import tqdm
import pandas as pd
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def get_top_by_obj_aff(data_array, top_n=10):

    # Create a DataFrame
    df = pd.DataFrame(data_array, columns=['idx', 'aff', 'cls', 'iou'])

    # Convert specific columns to integers
    df[['idx', 'aff', 'cls']] = df[['idx', 'aff', 'cls']].astype(int)

    # Define the mapping lists
    classes = ["Bag", "Bed", "Bowl", "Clock", "Dishwasher", "Display", "Door", "Earphone", "Faucet",
               "Hat", "StorageFurniture", "Keyboard", "Knife", "Laptop", "Microwave", "Mug",
               "Refrigerator", "Chair", "Scissors", "Table", "TrashCan", "Vase", "Bottle"]

    affordance = ['lay', 'sit', 'support', 'grasp', 'lift', 'contain', 'open', 'wrap_grasp', 'pour', 
                  'move', 'display', 'push', 'pull', 'listen', 'wear', 'press', 'cut', 'stab']

    # Replace indices with actual values
    df['aff'] = df['aff'].apply(lambda x: affordance[x] if x < len(affordance) else 'Unknown')
    df['cls'] = df['cls'].apply(lambda x: classes[x] if x < len(classes) else 'Unknown')

    # Function to select the top entries based on 'iou'
    def select_top_entries_or_all(group):
        if len(group) < top_n:
            return group.nlargest(len(group), 'iou')
        else:
            return group.nlargest(top_n, 'iou')

    # Group the DataFrame and apply the selection function
    grouped = df.groupby(['aff', 'cls'], group_keys=False).apply(select_top_entries_or_all)

    # Select relevant columns to display
    result = grouped[['aff', 'cls', 'idx', 'iou']]

    return result




def calculate_batch_iou(results, targets):
    iou = np.zeros(results.shape[0])

    IOU_thres = np.linspace(0, 1, 20)
    targets = targets >= 0.5
    targets = targets.astype(int)
    for i in range(results.shape[0]):
        t_true = targets[i]
        p_score = results[i]

        if np.sum(t_true) == 0:
            iou_ = np.nan
        else:

            p_mask = (p_score > 0.5).astype(int)
            temp_iou = []
            for thre in IOU_thres:
                p_mask = (p_score >= thre).astype(int)
                intersect = np.sum(p_mask & t_true)
                union = np.sum(p_mask | t_true)
                temp_iou.append(1.*intersect/union)
            temp_iou = np.array(temp_iou)
            iou_ = np.mean(temp_iou)
        iou[i]=iou_
    return iou


def evaluate(model, test_dataset, device, num_votes=3, top_n=10):
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=8, shuffle=False)

    # Initialize the structures for holding the predictions and targets
    result = []
    pred = []
    
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        # Iterate over the test data loader to gather predictions
        for i, (point, cls, label, question, aff_label) in tqdm(enumerate(test_loader), total=len(test_loader), ascii=True):
            point, label = point.float(), label.float()
            point = point.to(device)  # Ensure the data is on the right device
            label = label.to(device)

            _3d = torch.zeros_like(label)  # Initialize the structure for aggregated predictions

            # Aggregate predictions with voting
            for v in range(num_votes):
                seed_torch(v)  # Set the seed for reproducibility
                _3d += model(question, point)[0]  # Add the predictions to the aggregate
                
            _3d /= num_votes  # Average the predictions over all votes

            """ 
            ########### Evaluation ##########
            """
            _3d = _3d.cpu().numpy()
            label = label.cpu().numpy()
            cls = cls.numpy()
            aff_label = aff_label.numpy()

            iou = calculate_batch_iou(_3d, label)  # Calculating IoU for this instance
            
            result.append(np.vstack([aff_label,cls,iou]))
            pred.append(_3d)
            
    result=np.transpose(np.hstack(result)) #  nsample,3
    result = np.hstack([np.expand_dims(np.arange(len(result)), axis=1),result]) # nsample,4
    pred = np.vstack(pred)
    
    
    top_result = get_top_by_obj_aff(result, top_n=10)
    top_result=top_result.to_dict('records')
    print(top_result)
    anno, objects = test_dataset.anno, test_dataset.objects

    for dict in top_result:
        dict['pred']=pred[dict['idx']]
        dict['gt_mask']=anno[dict['idx']]['mask']
        dict['shape_id']=str(anno[dict['idx']]['shape_id'])
        dict['shape']=objects[str(anno[dict['idx']]['shape_id'])]
    
    output_file_path = '/storage_fast/ycli/3d_affordance/visualizer/data/top{}-objaff.pkl'.format(top_n) 
    with open(output_file_path, 'wb') as file:
        pickle.dump(top_result, file)


if __name__=='__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = AffordQ('test')
    
    dict = read_yaml('config/config_seen.yaml')
    model = get_IAGNet(emb_dim=dict['emb_dim'],
                       proj_dim=dict['proj_dim'], num_heads=dict['num_heads'], N_raw=dict['N_raw'],
                       num_affordance = dict['num_affordance'], n_groups=20)
    model_checkpoint = torch.load('/storage_fast/ycli/3d_affordance/10270_3GP_Q_Q_change_group/runs/train/IAG/best_model-20groups-1_at_10.28_1.20.54.pt', map_location='cuda:0')
    model.load_state_dict(model_checkpoint['model'])
    model.to(device)
   
    evaluate(model, test_dataset,device, num_votes=3, top_n=5)
