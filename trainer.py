import os
import sys
sys.path.append("..")
from loss import *
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    average_precision_score, accuracy_score, precision_score,f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from model import * 
from dataloader import * 
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from graph_learning import *
from torch_geometric.data import DataLoader
import os
import sys
from torch_geometric.nn import GCNConv, global_mean_pool
from loss import *
from sklearn.neighbors import KNeighborsClassifier
from model import *
import torch
import numpy as np
import torch.nn as nn
import os
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from graph_learning import *
from augmentations import *
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
from load_data import *
from torch_geometric.data import DataLoader, Batch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from fairlearn.metrics import demographic_parity_difference
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import DataLoader, Batch
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from loss import *
from model import *
from dataloader import *
from graph_learning import *
from augmentations import *
from load_data import *
from fairlearn.metrics import demographic_parity_ratio
from geomloss import SamplesLoss

#####################################################################
dataset_name = 'ABIDE'
atlas_name = 'AAL'

def check_gradients(model):
    """Check if model gradients are updating."""
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"WARNING: No gradient for {name}")

def check_embeddings_change(prev_emb, new_emb, epoch):
    """Check if embeddings are changing across epochs."""
    if prev_emb is not None:
        change = np.linalg.norm(new_emb - prev_emb)
        print(f"Epoch {epoch}: Embedding Change: {change:.6f}")
    return new_emb

#####################################################################
# Data Collation Function
#####################################################################

def collate(data_list):
    return Batch.from_data_list(data_list)

#####################################################################
# One-Hot Encoding
#####################################################################

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    return np.eye(n_values)[X]

#####################################################################
# Trainer Function
####################################################################

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import StratifiedKFold

#####################################################################

def Trainer(model, model_optimizer, classifier, classifier_optimizer, sourcedata_path, device,
            config, experiment_log_dir, training_mode):
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    if training_mode == 'pre_train':
        print(' Pretraining on source dataset...')
        for epoch in range(1, config.num_epoch + 1):
            print(f'\n Epoch {epoch}/{config.num_epoch}')
            clean_dataset_time, pert_dataset_time, clean_dataset_freq, pert_dataset_freq, clean_adj_time, clean_adj_freq = train_get_generator(
                sourcedata_path, dataset_name, atlas_name, training_mode)
            train_loss = model_pretrain(
                model, model_optimizer, criterion, clean_dataset_time, pert_dataset_time,
                clean_dataset_freq, pert_dataset_freq, clean_adj_time, clean_adj_freq,
                config, device, training_mode
            )
        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, os.path.join(experiment_log_dir, "saved_models", 'ckp_last.pt'))
        print(f'Pretrained model saved at: {experiment_log_dir}/saved_models/ckp_last.pt')

    else:
        print('Fine-tuning on fine-tuning set...')
        pretrained_model_path = os.path.join(experiment_log_dir, "saved_models", 'ckp_last.pt')
        if os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])


        
        # Load data for fine-tuning, validation, and testing
        finetune_dataset_time, finetune_dataset_freq, clean_adj_time, clean_adj_freq = finetune_test_get_generator(
            sourcedata_path, dataset_name, atlas_name, training_mode)
        labels = [data.y for data in finetune_dataset_time] 

        # Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=89)

        best_val_test_accuracies = []
        best_val_test_results = []

        for fold, (trainval_idx, test_idx) in enumerate(skf.split(finetune_dataset_time, labels)):
            print('\ncurrent Fold', fold)
            trainval_time = [finetune_dataset_time[i] for i in trainval_idx]
            trainval_freq = [finetune_dataset_freq[i] for i in trainval_idx]
            trainval_labels = [labels[i] for i in trainval_idx]
        
            test_time = [finetune_dataset_time[i] for i in test_idx]
            test_freq = [finetune_dataset_freq[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]
        
            # Split using train_test_split to generate relative indices
            rel_finetune_idx, rel_val_idx = train_test_split(range(len(trainval_time)), test_size=0.1, shuffle=True, stratify=trainval_labels, random_state=89)
            
            # Now use these relative indices to access trainval_time and trainval_freq
            finetune_time = [trainval_time[i] for i in rel_finetune_idx]
            finetune_freq = [trainval_freq[i] for i in rel_finetune_idx]
            validation_time = [trainval_time[i] for i in rel_val_idx]
            validation_freq = [trainval_freq[i] for i in rel_val_idx]

            best_val_test_accuracy = -1
            best_val_test_result = None

            for epoch in range(1,config.num_epoch):  # Run for 100 epochs
                print(f"\n  Epoch {epoch}")

                # Fine-tuning the model
                print('Finetune........................')
                finetune_loss, _, _, _ = model_finetune(
                    model, model_optimizer, finetune_time, finetune_freq, clean_adj_time, clean_adj_freq,
                    config, device, training_mode, classifier, classifier_optimizer
                )

                # Validating the model
                print('Validation........................')
                val_loss, val_accuracy, val_auc, val_prc, _, val_metrics = model_test(
                    model, validation_time, validation_freq, clean_adj_time, clean_adj_freq,
                    config, device, training_mode, classifier, classifier_optimizer
                )

                # Testing the model
                print('test........................')
                test_loss, test_accuracy, test_auc, test_prc, _, test_metrics = model_test(
                    model, test_time, test_freq, clean_adj_time, clean_adj_freq,
                    config, device, training_mode, classifier, classifier_optimizer
                )


                if val_accuracy > best_val_test_accuracy:
                    best_val_test_accuracy = test_accuracy
                    best_val_test_result = test_metrics

            best_val_test_accuracies.append(best_val_test_accuracy)
            best_val_test_results.append(best_val_test_result)



        print("\n FINAL RESULTS:")
        for fold in range(5):
            print(f"Fold {fold + 1}: Metrics: {best_val_test_results[fold]}")
        
        # Compute averages and standard deviations of test accuracy, F1, AUC, SER, DPR, and DPD across folds
        average_test_accuracy = np.mean([result[0] for result in best_test_results])
        std_test_accuracy = np.std([result[0] for result in best_test_results])
        
        average_test_auc = np.mean([result[4] for result in best_test_results])
        std_test_auc = np.std([result[4] for result in best_test_results])
        
        demographic_diff = np.mean([result[5] for result in best_test_results])
        std_demographic_diff = np.std([result[5] for result in best_test_results])
        

        
        # Compute averages and standard deviations for validation results similarly
        average_val_test_accuracy = np.mean([result[0] for result in best_val_test_results])
        std_val_test_accuracy = np.std([result[0] for result in best_val_test_results])

        average_val_test_auc = np.mean([result[4] for result in best_val_test_results])
        std_val_test_auc = np.std([result[4] for result in best_val_test_results])
        
        val_test_demographic_diff = np.mean([result[5] for result in best_val_test_results])
        std_val_test_demographic_diff = np.std([result[5] for result in best_val_test_results])
    
        print(f"Accuracy: {average_val_test_accuracy:.4f} (±{std_val_test_accuracy:.4f})")
        print(f"AUC: {average_val_test_auc:.4f} (±{std_val_test_auc:.4f})")
        print(f"Demographic Parity Difference: {val_test_demographic_diff:.4f} (±{std_val_test_demographic_diff:.4f})")


#####################################################################    

def reconstruction_loss(x_t, adj_freq):
    batch_size, d, _ = adj_freq.shape  # Extract dimensions
    loss = 0.0
    
    expected_rows = batch_size * d
    if x_t.shape[0] != expected_rows:
        raise ValueError(f"x_t should have {expected_rows} rows, but has {x_t.shape[0]} rows")

    # Process each batch sample
    for i in range(batch_size):
        start_idx = i * d
        end_idx = start_idx + d
        x_d = x_t[start_idx:end_idx, :] 
        x_x_t = torch.matmul(x_d, x_d.T)  
        x_x_t = torch.sigmoid(x_x_t)
        frob_norm = torch.norm(x_x_t - adj_freq[i], p='fro')**2 
        loss += frob_norm

    return loss



#####################################################################


def model_pretrain(model, model_optimizer, criterion, clean_dataset_time, pert_dataset_time, clean_dataset_freq, pert_dataset_freq, adj_freq, adj_time, config, device, training_mode):
    total_loss = []
    model.train()

    dataset_size = len(clean_dataset_time)
    
    # Create a shared shuffled index list
    indices = torch.randperm(dataset_size).tolist() 
  
    clean_dataset_time_subset = Subset(clean_dataset_time, indices)
    pert_dataset_time_subset = Subset(pert_dataset_time, indices)
    clean_dataset_freq_subset = Subset(clean_dataset_freq, indices)
    pert_dataset_freq_subset = Subset(pert_dataset_freq, indices)
    
    # Create DataLoaders with subsets to ensure identical sample indices across datasets
    data_loader_time = DataLoader(clean_dataset_time_subset, batch_size=config.batch_size, num_workers=0, collate_fn=collate, shuffle=False)
    data_loader_time_perb = DataLoader(pert_dataset_time_subset, batch_size=config.batch_size, num_workers=0, collate_fn=collate, shuffle=False)
    data_loader_freq = DataLoader(clean_dataset_freq_subset, batch_size=config.batch_size, num_workers=0, collate_fn=collate, shuffle=False)
    data_loader_freq_perb = DataLoader(pert_dataset_freq_subset, batch_size=config.batch_size, num_workers=0, collate_fn=collate, shuffle=False)
    

    for batch_index, (data, data_f, aug1, aug1_f) in enumerate(zip(data_loader_time, data_loader_freq, data_loader_time_perb, data_loader_freq_perb)):
        data = data.to(device)
        data_f = data_f.to(device)
        aug1 = aug1.to(device)
        aug1_f = aug1_f.to(device)
     
        sex = data.sex 
        if isinstance(sex, list):
            sex_tensor = torch.cat([torch.tensor(s, dtype=torch.long, device=device) for s in sex])
        else:
            sex_tensor = torch.tensor(sex, dtype=torch.long, device=device).to(device)
                
        batch_indices = data.idx  
        current_adj_freq = [adj_freq[i] for i in batch_indices] 
        current_adj_time = [adj_time[i] for i in batch_indices]
    
        # Convert to tensors
        adj_freq_tensor = torch.stack([torch.tensor(item, dtype=torch.float32, device=device) for item in current_adj_freq])
        adj_time_tensor = torch.stack([torch.tensor(item, dtype=torch.float32, device=device) for item in current_adj_time])


        # Reset gradients
        model_optimizer.zero_grad()

        # Produce embeddings
        h_t, z_t, h_f, z_f, x_t, x_f = model(data, data_f) 
        h_t_aug, z_t_aug, h_f_aug, z_f_aug, x_t_aug, x_f_aug = model(aug1, aug1_f)

        # Compute pre-training loss
        nt_xent_criterion = NTXentLoss_poly(device, config.batch_size, config.Context_Cont.temperature, config.Context_Cont.use_cosine_similarity)
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)

        fused_x = torch.cat((x_t , x_f), dim = 1)
        d = 116 
        pooled_results = []

        for i in range(0, fused_x.shape[0], d):
            slice_ = fused_x[i:i + d]
            mean_pooled = torch.mean(slice_, dim=0, keepdim=True)
            pooled_results.append(mean_pooled)
            
        pool_matrix = torch.cat(pooled_results, dim=0)
        male_pool = pool_matrix[sex_tensor == 1]
        female_pool = pool_matrix[sex_tensor == 2]


        loss_recons1 = reconstruction_loss(fused_x, adj_freq_tensor)
        loss_recons2 = reconstruction_loss(fused_x, adj_time_tensor)
        loss_recons = loss_recons1 + loss_recons2

        sinkhorn_loss = SamplesLoss(loss ="sinkhorn", p=2, blur=0.01)
        divergence = sinkhorn_loss(male_pool,female_pool)
        

        # Combine losses
        lam = 0.5
        alpha = 0.8
        beta  = 1
        loss = lam * (loss_t + loss_f) + alpha * loss_recons + beta*divergence
        
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

    print(f'Pretraining: overall loss: {total_loss[-1]}, loss_t: {loss_t.item()}, loss_f: {loss_f.item()}, loss_reconstruction: {loss_recons.item()}, divergence:          {divergence.item()}')

    ave_loss = torch.tensor(total_loss).mean()
    
    return ave_loss

#####################################################################
def model_finetune(model, model_optimizer, clean_dataset_time, clean_dataset_freq, clean_adj_time, clean_adj_freq,
                   config, device, training_mode, classifier=None, classifier_optimizer=None, global_epoch=None):
    global labels, pred_numpy, fea_concat_flat

    total_loss = []
    all_labels, all_preds, all_probs = [], [], []  # Storing labels for final metric computation
    
    criterion = nn.CrossEntropyLoss()

    model.train()
    classifier.train()

    dataset_size = len(clean_dataset_time)
    indices = torch.randperm(dataset_size).tolist()
    clean_dataset_time_subset = Subset(clean_dataset_time, indices)
    clean_dataset_freq_subset = Subset(clean_dataset_freq, indices)

    data_loader_time = DataLoader(clean_dataset_time_subset, batch_size=config.batch_size, num_workers=0, collate_fn=collate, shuffle=True)
    data_loader_freq = DataLoader(clean_dataset_freq_subset, batch_size=config.batch_size, num_workers=0, collate_fn=collate, shuffle=True)

    prev_emb = None  # Used to check if embeddings are changing

    for epoch in range(1, 1 + 1):

        for batch_index, (data, data_f) in enumerate(zip(data_loader_time, data_loader_freq)):
            data, data_f = data.to(device), data_f.to(device)
            labels = data.y.to(device)
            genders = torch.tensor(np.concatenate(data.sex), dtype=torch.long)  # Extract gender info

            model_optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            h_t, z_t, h_f, z_f, x_t, x_f = model(data, data_f)
            fea_concat = torch.cat((x_t , x_f), dim=1)
            d,d = clean_adj_freq[0].shape  # Extract dimensions
            
            pooled_results = []

            for i in range(0, fea_concat.shape[0], d):
                slice_ = fea_concat[i:i + d]
                mean_pooled = torch.mean(slice_, dim=0, keepdim=True)
                pooled_results.append(mean_pooled)
           
            pool_matrix = torch.cat(pooled_results, dim=0)

        
            male_pool = pool_matrix[genders == 1]
            female_pool = pool_matrix[genders == 2]

        
            predictions = classifier(pool_matrix)
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)

            loss = criterion(predictions, labels)

            loss.backward()
            model_optimizer.step()
            classifier_optimizer.step()
            pred_numpy = predictions.detach().cpu().numpy()
            pred_labels = np.argmax(pred_numpy, axis=1)
            true_labels = labels.cpu().numpy()
            pred_probs = torch.softmax(predictions, dim=1).detach().cpu().numpy()[:, 1]  # Probability of positive class

            all_labels.extend(true_labels)
            all_preds.extend(pred_labels)
            all_probs.extend(pred_probs)

            total_loss.append(loss.item())

            prev_emb = fea_concat_flat.detach().cpu().numpy()

    ave_loss = np.mean(total_loss)
    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds, average='macro')
    overall_auc = roc_auc_score(all_labels, np.array(all_probs), average='macro')

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro')
    ave_prc = average_precision_score(all_labels, np.array(all_probs), average='macro')
    
    print(' Fine-tune: Loss = %.4f | Acc = %.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC = %.4f | AUPRC = %.4f'
          % (ave_loss, overall_accuracy * 100, precision * 100, recall * 100, overall_f1 * 100, overall_auc * 100, ave_prc * 100))

    return ave_loss, prev_emb, all_labels, overall_f1


#####################################################################

def model_test(model, clean_dataset_time, clean_dataset_freq, clean_adj_time, clean_adj_freq,
               config, device, training_mode, classifier=None, classifier_optimizer=None):
   
    
    dataset_size = len(clean_dataset_time)
    indices = torch.randperm(dataset_size).tolist()  # Shuffle dataset indices
    clean_dataset_time_subset = Subset(clean_dataset_time, indices)
    clean_dataset_freq_subset = Subset(clean_dataset_freq, indices)
    
    data_loader_time = DataLoader(clean_dataset_time_subset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    data_loader_freq = DataLoader(clean_dataset_freq_subset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    
    model.eval()
    classifier.eval()

    total_loss = []
    all_labels, all_preds, all_probs = [], [], []  # For final metric computation
    all_male_labels, all_male_preds, all_male_probs = [], [], []  # Male-specific metrics
    all_female_labels, all_female_preds, all_female_probs = [], [], []  # Female-specific metrics
    sf_data = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_index, (data, data_f) in enumerate(zip(data_loader_time, data_loader_freq)):
            data, data_f = data.to(device), data_f.to(device)
            labels = data.y.to(device)
            genders = torch.tensor(np.concatenate(data.sex), dtype=torch.long)  # Extract gender info
            
            h_t, z_t, h_f, z_f, x_t, x_f = model(data, data_f)
            fea_concat = torch.cat((x_t, x_f), dim=1)            
            d,d = clean_adj_freq[0].shape  # Extract dimensions
            #print('d is:',d )
            
            pooled_results = []

            # Iterate over fea_concat in steps of d rows
            for i in range(0, fea_concat.shape[0], d):
                # Extract the slice of d rows
                slice_ = fea_concat[i:i + d]
                
                # Compute the mean across these rows (dim=0 to mean along rows, producing a [1, 64] tensor)
                mean_pooled = torch.mean(slice_, dim=0, keepdim=True)
                
                # Append the mean pooled result to the list
                pooled_results.append(mean_pooled)
            
            # Concatenate all the mean pooled results along the dimension 0
            pool_matrix = torch.cat(pooled_results, dim=0)

            male_pool = pool_matrix[genders == 1]
            female_pool = pool_matrix[genders == 2]

            # print('male_pool',male_pool.shape)
            # print('female pool',female_pool.shape)

            
            predictions_test = classifier(pool_matrix)
            loss = criterion(predictions_test, labels)

            pred_probs = torch.softmax(predictions_test, dim=1).cpu().numpy()
            pred_labels = np.argmax(pred_probs, axis=1)
            true_labels = labels.cpu().numpy()

            all_labels.extend(true_labels)
            all_preds.extend(pred_labels)
            all_probs.extend(pred_probs[:, 1]) 

            total_loss.append(loss.item())

            male_mask = (genders == 0)
            if male_mask.any():
                all_male_labels.extend(true_labels[male_mask])
                all_male_preds.extend(pred_labels[male_mask])
                all_male_probs.extend(pred_probs[male_mask, 1])

            female_mask = (genders == 1)
            if female_mask.any():
                all_female_labels.extend(true_labels[female_mask])
                all_female_preds.extend(pred_labels[female_mask])
                all_female_probs.extend(pred_probs[female_mask, 1])

            sf_data.extend(['M' if g == 1 else 'F' for g in genders.cpu().numpy()])
    

    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds, average='macro')
    overall_auc = roc_auc_score(all_labels, np.array(all_probs), average='macro')

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro')
    prc = average_precision_score(all_labels, np.array(all_probs), average='macro')

    if all_male_labels:
        male_accuracy = accuracy_score(all_male_labels, all_male_preds)
        male_f1 = f1_score(all_male_labels, all_male_preds, average='macro')
        male_auc = roc_auc_score(all_male_labels, np.array(all_male_probs), average='macro')
        male_precision = precision_score(all_male_labels, all_male_preds, average='macro', zero_division=0)
        male_recall = recall_score(all_male_labels, all_male_preds, average='macro')
    else:
        male_accuracy = male_f1 = male_auc = male_precision = male_recall = 0.0

    if all_female_labels:
        female_accuracy = accuracy_score(all_female_labels, all_female_preds)
        female_f1 = f1_score(all_female_labels, all_female_preds, average='macro')
        female_auc = roc_auc_score(all_female_labels, np.array(all_female_probs), average='macro')
        female_precision = precision_score(all_female_labels, all_female_preds, average='macro', zero_division=0)
        female_recall = recall_score(all_female_labels, all_female_preds, average='macro')
    else:
        female_accuracy = female_f1 = female_auc = female_precision = female_recall = 0.0


    DPD = demographic_parity_difference(all_labels, all_preds, sensitive_features=sf_data)   
          

    print(f"Test Accuracy: {overall_accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | "
          f"F1: {overall_f1:.4f} | AUROC: {overall_auc:.4f} | AUPRC: {prc:.4f} | DPD: {DPD: .4f}")
    
    return np.mean(total_loss), overall_accuracy, overall_auc, prc, all_labels, [
        overall_accuracy * 100, precision * 100, recall * 100, overall_f1 * 100, overall_auc * 100, prc * 100, DPD
    ]

#####################################################################

