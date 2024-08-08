import argparse
import torch
import numpy as np
import os
import logging
from utils.util import seed_torch, set_gpu_devices, read_yaml
from utils.logger import logger as loggger

parser = argparse.ArgumentParser()

parser.add_argument("-v", type=str, required=True, help="version", default="try")
parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=16)
parser.add_argument("-lr", type=float, action="store", help="learning rate", default=1e-4)
parser.add_argument("-tlr", type=float, action="store", help="learning rate for text encoder", default=5e-6)
parser.add_argument("-epoch", type=int, action="store", help="epoch for train", default=50)

# parser.add_argument('-radius', type=float, default=0.1, help='radius for ball query in last decoder')
parser.add_argument("-n_groups", type=int, action="store", help="num of queres", default=40)
parser.add_argument("-gpu", type=int, help="set gpu id", default=1) 
parser.add_argument('--decay_rate', type=float, default=1e-3, help='weight decay [default: 1e-3]')
parser.add_argument('--use_gpu', type=str, default=True, help='whether or not use gpus')
parser.add_argument('--save_dir', type=str, default='runs/train/', help='path to save .pt model while training')
parser.add_argument('--name', type=str, default='PointRefer', help='training name to classify each training process')
parser.add_argument('--resume', type=str, default=False, help='start training from previous epoch')
parser.add_argument('--checkpoint_path', type=str, default='runs/train/PointRefer/best.pt', help='checkpoint path')
parser.add_argument('--log_name', type=str, default='train.log', help='the name of current training')
parser.add_argument('--loss_cls', type=float, default=0.3, help='cls loss scale')
parser.add_argument('--storage', type=bool, default=False, help='whether to storage the model during training')
parser.add_argument('--yaml', type=str, default='config/default.yaml', help='yaml path')

opt = parser.parse_args()

seed_torch(seed=42)
set_gpu_devices(opt.gpu)

import torch.nn as nn
from torch.utils.data import DataLoader
from model.PointRefer import get_PointRefer
from utils.loss import HM_Loss, kl_div
from utils.eval import evaluating, SIM
from eval_lyc import evaluate, print_metrics_in_table 
from data_utils.shapenetpart import AffordQ
from sklearn.metrics import roc_auc_score


def main(opt, dict):
    logger, sign = loggger(opt)

    if opt.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    save_path = opt.save_dir + opt.name
    foler = os.path.exists(save_path)
    if not foler:
        os.makedirs(save_path)

    batch_size = dict['bs']

    logger.debug('Start loading train data---')
    train_dataset = AffordQ('train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8 ,shuffle=True, drop_last=True)
    logger.debug(f'train data loading finish, loading data files:{len(train_dataset)}')

    logger.debug('Start loading val data---')
    val_dataset = AffordQ('val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    logger.debug(f'val data loading finish, loading data files:{len(val_dataset)}')

    logger.debug('Start loading test data---')
    test_dataset = AffordQ('test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    logger.debug(f'test data loading finish, loading data files:{len(test_dataset)}')

    model = get_PointRefer(emb_dim=dict['emb_dim'],
                       proj_dim=dict['proj_dim'], num_heads=dict['num_heads'], N_raw=dict['N_raw'],
                       num_affordance = dict['num_affordance'], n_groups=opt.n_groups)

    criterion_hm = HM_Loss()
    criterion_ce = nn.CrossEntropyLoss()
    param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad]},
    {"params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad], "lr": opt.tlr}]
    optimizer = torch.optim.Adam(params = param_dicts, lr=dict['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.decay_rate)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=dict['Epoch'], eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if opt.resume:
        model_checkpoint = torch.load(opt.checkpoint_path, map_location='cuda:0')
        model.load_state_dict(model_checkpoint['model'])
        optimizer.load_state_dict(model_checkpoint['optimizer'])
        start_epoch = model_checkpoint['Epoch']
    else:
        start_epoch = -1

    model = model.to(device)
    criterion_hm = criterion_hm.to(device)
    criterion_ce = criterion_ce.to(device)

    best_IOU = 0
    '''
    Training
    '''
    for epoch in range(start_epoch+1, dict['Epoch']):
        logger.debug(f'Epoch:{epoch} strat-------')
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        logger.debug(f'lr_rate:{learning_rate}')

        num_batches = len(train_loader)
        loss_sum = 0
        total_point = 0
        model = model.train()
        # print(f'cuda memory:{torch.cuda.memory_allocated(opt.gpu) / (1024*1024)}')
        for i,(point, cls, gt_mask, question, aff_label) in enumerate(train_loader):
            
            optimizer.zero_grad()      

            if(opt.use_gpu):
                point = point.to(device)
                gt_mask = gt_mask.to(device)
                aff_label = aff_label.to(device)
                cls = cls.to(device)

            _3d = model(question, point)
            loss_hm = criterion_hm(_3d, gt_mask)
            # loss_ce = criterion_ce(logits, cls)

            temp_loss = loss_hm # + opt.loss_cls*loss_ce

            print(f'Epoch:{epoch}| iteration:{i}|{len(train_loader)} | loss:{temp_loss.item()}')
            temp_loss.backward()
            optimizer.step()
            loss_sum += temp_loss.item()

        # print(f'cuda memorry:{torch.cuda.memory_allocated(opt.gpu) / (1024*1024)}')
        mean_loss = loss_sum / (num_batches*dict['pairing_num'])
        logger.debug(f'Epoch:{epoch} | mean_loss:{mean_loss}')

        if(opt.storage == True):
            if((epoch+1) % 1==0):
                model_path = save_path + '/Epoch_' + str(epoch+1) + '.pt'
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'Epoch': epoch
                }
                torch.save(checkpoint, model_path)
                logger.debug(f'model saved at {model_path}')
        
        results = torch.zeros((len(val_dataset), 2048, 1))
        targets = torch.zeros((len(val_dataset), 2048, 1))
        '''
        Evalization
        '''
        if((epoch+1)%1 == 0):
            num = 0
            with torch.no_grad():
                logger.debug(f'EVALUATION strat-------')
                num_batches = len(val_loader)
                val_loss_sum = 0
                total_MAE = 0
                total_point = 0
                model = model.eval()
                for i,(point, _, label, question,aff_label) in enumerate(val_loader):
                    print(f'iteration: {i}|{len(val_loader)} start----')
                    point, label = point.float(), label.float()
                    if(opt.use_gpu):
                        point = point.to(device)
                        label = label.to(device)
                    
                    _3d = model(question, point)

                    # val_loss_hm = criterion_hm(_3d, label)
                    # val_loss_ce = criterion_ce(logits, aff_label)
                    # val_loss = val_loss_hm + opt.loss_ce*val_loss_ce

                    mae, point_nums = evaluating(_3d, label)
                    total_point += point_nums
                    # val_loss_sum += val_loss.item()
                    total_MAE += mae.item()
                    pred_num = _3d.shape[0]
                    # print(f'---val_loss | {val_loss.item()}')
                    results[num : num+pred_num, :, :] = _3d.unsqueeze(-1)
                    targets[num : num+pred_num, :, :] = label.unsqueeze(-1)
                    num += pred_num

                # val_mean_loss = val_loss_sum / num_batches
                # logger.debug(f'Epoch_{epoch} | val_loss | {val_mean_loss}')
                mean_mae = total_MAE / total_point
                results = results.detach().numpy()
                targets = targets.detach().numpy()
                # print(f'cuda memorry:{torch.cuda.memory_allocated(opt.gpu)/ (1024*1024)}')
                SIM_matrix = np.zeros(targets.shape[0])
                for i in range(targets.shape[0]):
                    SIM_matrix[i] = SIM(results[i], targets[i])
                
                sim = np.mean(SIM_matrix)
                AUC = np.zeros((targets.shape[0], targets.shape[2]))
                IOU = np.zeros((targets.shape[0], targets.shape[2]))
                IOU_thres = np.linspace(0, 1, 20)
                targets = targets >= 0.5
                targets = targets.astype(int)
                for i in range(AUC.shape[0]):
                    t_true = targets[i]
                    p_score = results[i]

                    if np.sum(t_true) == 0:
                        AUC[i] = np.nan
                        IOU[i] = np.nan
                    else:
                        auc = roc_auc_score(t_true, p_score)
                        AUC[i] = auc

                        p_mask = (p_score > 0.5).astype(int)
                        temp_iou = []
                        for thre in IOU_thres:
                            p_mask = (p_score >= thre).astype(int)
                            intersect = np.sum(p_mask & t_true)
                            union = np.sum(p_mask | t_true)
                            temp_iou.append(1.*intersect/union)
                        temp_iou = np.array(temp_iou)
                        aiou = np.mean(temp_iou)
                        IOU[i] = aiou
                
                AUC = np.nanmean(AUC)
                IOU = np.nanmean(IOU)

                logger.debug(f'AUC:{AUC} | IOU:{IOU} | SIM:{sim} | MAE:{mean_mae}')

                current_IOU = IOU
                if(current_IOU > best_IOU):
                    best_IOU = current_IOU
                    best_model_path = save_path + '/best_model-{}.pt'.format(sign)
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'Epoch': epoch
                    }
                    torch.save(checkpoint, best_model_path)
                    logger.debug(f'best model saved at {best_model_path}')
        scheduler.step()
    logger.debug(f'Best Val IOU:{best_IOU}')

    category_metrics, affordance_metrics, overall_metrics = evaluate(model, test_loader, device, 3)
    print_metrics_in_table(category_metrics, affordance_metrics, overall_metrics, logger)


if __name__=='__main__':
    dict = read_yaml(opt.yaml)
    dict['bs'] = opt.bs
    dict['lr'] = opt.lr
    dict['Epoch'] =opt.epoch
    main(opt, dict)
    # torch.autograd.set_detect_anomaly(True)