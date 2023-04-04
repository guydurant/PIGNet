import os
import sys
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from tqdm import tqdm
from data.generate_keys import write_keys
# from train import run
# import arguments
import utils
import models
import torch
import torch.nn as nn
from dataset import get_dataset_dataloader
import time
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score
from typing import Any, Dict, List, Union, Tuple

# ARGUMENTS CAN BE DEFINED OR BY CONFIG - CAN ONLY DO ONE - raise error if try both
# UNIT Tests - if can start running, then quit after 1 minute
# If Pignet extra data does not affect performance, put in .sif file

# Prepare data
# Download cross, docking, random files if do not exist

# class PigNetWrapper:
#     #Should be initialised AND THEN run training 
#     def __init__(, args, pignet_data_dir, training_csv):
#         .args = args
#         .pignet_data_dir = pignet_data_dir
#         .training_csv = training_csv


def process_tar_files():
    os.system('tar -xzf data.tar.gz')
    # Change function to generate_keys just from the csv file - So new generate_keys_wrapper.py
    # os.system('python /generate_keys.py -d data -k keys -c ../../coreset_keys.txt --train')
    # # Create csv file with all of the values - so create new pdb_to_affinity_wrapper.py
    # os.system('python data/pdb_to_affinity.py -d data -f pdb_to_affinity.txt -i ../../INDEX_refined_data.2019')
    # os.system('cd -')
    return 

def get_files(pignet_data_dir):
    print('Downloading extra datasets for PigNet')
    # - Specify path to pignet external data as PATH
    # - Check if overall folders exists
    # - If not download them using the wget commands (os.system)
    if not os.path.exists(f'{pignet_data_dir}/docking/data'):
        os.system(f'mkdir {pignet_data_dir}/docking')
        os.system(f'wget https://zenodo.org/record/6047984/files/pdbbind_v2019_docking.tar.gz?download=1 -O {pignet_data_dir}/docking/data.tar.gz')
        # os.system(f'cd {pignet_data_dir}/docking')
        os.system(f'tar -xzf {pignet_data_dir}/docking/data.tar.gz')
        process_tar_files()
    if not os.path.exists(f'{pignet_data_dir}/random/data'):
        os.system(f'mkdir {pignet_data_dir}/random')
        os.system(f'wget https://zenodo.org/record/6047984/files/pdbbind_v2019_random_screening.tar.gz?download=1 -O {pignet_data_dir}/random/data.tar.gz')
        # os.system(f'cd {pignet_data_dir}/random')
        os.system(f'tar -xzf {pignet_data_dir}/random/data.tar.gz')
        process_tar_files()
    if not os.path.exists(f'{pignet_data_dir}/screening/data'):
        os.system(f'mkdir {pignet_data_dir}/screening')
        os.system(f'wget https://zenodo.org/record/6047984/files/pdbbind_v2019_cross_screening.tar.gz?download=1 -O {pignet_data_dir}/screening/data.tar.gz')
        # os.system(f'cd {pignet_data_dir}/screening')
        os.system(f'tar -xzf {pignet_data_dir}/screening/data.tar.gz')
        process_tar_files()
    return None

def all_pks():
    pdbbind_df = pd.read_csv('data/INDEX_refined_data.2019', delim_whitespace=True, header=None, comment='#')
    pdb_pks_dict = {}
    for i in range(len(pdbbind_df)):
        pdb_pks_dict[pdbbind_df[0][i]] = pdbbind_df[3][i]
    return pdb_pks_dict

def read_training_csv(csv_file, data_dir):
    data = pd.read_csv(csv_file)
    proteins, ligands, pks, keys = data['protein'].to_list(), data['ligand'].to_list(), data['pk'].to_list(), data['key'].to_list()
    # Add relative path (.data_dir)
    proteins = [os.path.join(data_dir, protein) for protein in proteins]
    ligands = [os.path.join(data_dir, ligand) for ligand in ligands]
    pks = [float(pk) for pk in pks]
    data = {key: [protein, ligand, pk] for key, protein, ligand, pk in zip(keys, proteins, ligands, pks)}
    return data

def get_residues(protein_file_name, ligand_file_name):
    # Load ligand and protein
    ligand = Chem.MolFromMolFile(ligand_file_name)
    lig_conf = ligand.GetConformer()
    with open(protein_file_name, 'r') as f:
        protein_lines = f.readlines()
    protein_lines = [p for p in protein_lines if p[:4] == 'ATOM']
    # Get residues within 5A of ligand
    residues = []
    for line in protein_lines:
        x, y, z, element, res_number, res_name, chain_id = read_pdb_line(line, mode='all_protein')
        atom_coords = x, y, z
        if element == 'H':
            continue
        for ligand_atom in ligand.GetAtoms():
            if ligand_atom.GetAtomicNum() == 1:
                continue
            else:
                lig_coord = lig_conf.GetAtomPosition(ligand_atom.GetIdx())
                lig_coord = (lig_coord.x, lig_coord.y, lig_coord.z)
                if np.linalg.norm([atom_coords[i] - lig_coord[i] for i in range(len(atom_coords))]) < 5:
                    residues.append((res_number, res_name, chain_id))
                    break
    return residues

def return_int(string):
    try:
        return int(string)
    except:
        return int(string[:-1])

def read_pdb_line(line, mode='pocket'):
    atom_type = line[0:6].strip()
    atom_number = line[6:11].strip()
    atom_name = line[12:16].strip()
    alt_loc = line[16:17].strip()
    res_name = line[17:20].strip()
    chain_id = line[21:22].strip()
    res_number = line[22:26].strip()
    icode = line[26:27].strip()
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    occupancy = float(line[54:60].strip())
    temp_factor = float(line[60:66].strip())
    element = line[76:78].strip()
    charge = line[78:80].strip()
    if mode == 'pocket':
        return res_number, res_name, chain_id
    elif mode == 'all_protein':
        return x, y, z, element, res_number, res_name, chain_id

def create_pignet_pocket_file(protein_file_name, ligand_file_name):
    residues = list(set(get_residues(protein_file_name, ligand_file_name)))
    # print(residues)
    with open(protein_file_name, 'r') as f:
        lines = f.readlines()
    pocket_lines = []
    for l in lines:
        if l.split()[0] == 'ATOM':
            res_number, res_name, chain_id = read_pdb_line(l, mode='pocket')
            if (res_number, res_name, chain_id) in residues:
                pocket_lines.append(l)
    return pocket_lines

def pickle_data(data, model_name):
    print('Pickling data for scoring')
    for key in tqdm(data.keys()):
        #if path exists
        if os.path.exists(f'temp_features/{model_name}/{key}'):
            continue
        lines = create_pignet_pocket_file(data[key][0], data[key][1])
        with open(f'temp_files/{key}_pocket.pdb', 'w') as f:
            for line in lines:
                f.write(line+'\n')
        protein_pocket_mol = Chem.MolFromPDBFile(f'temp_files/{key}_pocket.pdb', removeHs=False)
        ligand_mol = Chem.MolFromMolFile(data[key][1])
        pickle.dump((ligand_mol, 0, protein_pocket_mol, []), open(f'temp_features/{model_name}/{key}', 'wb'))
        os.system(f'rm temp_files/{key}_pocket.pdb')
    return None

def generate_pdb_to_affinity(args, mode='scoring'):
    if mode == 'screening':
        with open(os.path.join(args.pignet_data_dir, mode, 'pdb_to_affinity.txt'), 'w') as f:
            for key in os.listdir(f'{args.pignet_data_dir}/screening/data'):
                f.write(f'{key}\t5\n')
    elif mode == 'scoring':
        data = read_training_csv(args.csv_file, args.data_dir)
        with open(f'temp_features/{args.model_name}/pdb_to_affinity.txt', 'w') as f:
            for key in data.keys():
                f.write(f'{key}\t{data[key][2]}\n')    
    else:
        all_pks_list = all_pks()
        with open(os.path.join(args.pignet_data_dir, mode, 'pdb_to_affinity.txt'), 'w') as f:
            for key in os.listdir(f'{args.pignet_data_dir}/{mode}/data'):
                f.write(f'{key}\t{all_pks_list[key.split("_")[0]]}\n')
    return None

def generate_all_pdb_to_affinity(args):
    print('Generating pdb_to_affinity.txt files')
    for mode in ['docking', 'random', 'screening']:
        if not os.path.exists(os.path.join(args.pignet_data_dir, mode, 'pdb_to_affinity.txt')):
            generate_pdb_to_affinity(args, mode=mode)
    # now for mode == 'scoring'
    if not os.path.exists(f'temp_features/{args.model_name}/pdb_to_affinity.txt'):
        generate_pdb_to_affinity(args, mode='scoring')

def generate_keys(args):
    core_keys = [i for i in open('coreset_keys.txt', 'r').read().split('\n') if i != '']
    for mode in ['docking', 'random', 'screening']:
        if not os.path.exists(os.path.join(args.pignet_data_dir, mode, 'train_keys.pkl')):
            keys = os.listdir(f'{args.pignet_data_dir}/{mode}/data')
            train_keys = []
            test_keys = []
            for key in keys:
                if key.split("_")[0] in core_keys:
                    test_keys.append(key)
                else:
                    train_keys.append(key)
            write_keys(train_keys, os.path.join(args.pignet_data_dir, mode, 'train_keys.pkl'))
            write_keys(test_keys, os.path.join(args.pignet_data_dir, mode, 'test_keys.pkl'))
    # Now for scoring mode
    if not os.path.exists(os.path.join(f'temp_features/{args.model_name}/train_keys.pkl')):
        data = read_training_csv(args.csv_file, args.data_dir)
        write_keys(list(data.keys()), f'temp_features/{args.model_name}/train_keys.pkl')
    return None 


def read_data(
    filename: str, key_dir: str, train: bool = True
) -> Tuple[Union[List[str], Dict[str, float]]]:
    with open(filename) as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
        id_to_y = {l[0]: float(l[1]) for l in lines}
    # with open(f"{key_dir}/test_keys.pkl", "rb") as f:
    #     test_keys = pickle.load(f)
    if train:
        with open(f"{key_dir}/train_keys.pkl", "rb") as f:
            train_keys = pickle.load(f)
        return train_keys, None, id_to_y
    else:
        return None, id_to_y


def set_up_training(args):
    # generate_keys
    generate_all_pdb_to_affinity(args)
    # pickle data for specific model
    pickle_data(read_training_csv(args.csv_file, args.data_dir), args.model_name)
    generate_keys(args)


    # Set GPU
    if args.ngpu > 0:
        cmd = utils.set_cuda_visible_device(args.ngpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = cmd
        print(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        pass

    # Read labels
    train_keys, test_keys, id_to_y = read_data(f'temp_features/{args.model_name}/pdb_to_affinity.txt',f'temp_features/{args.model_name}')
    train_keys2, test_keys2, id_to_y2 = read_data(f'{args.pignet_data_dir}/docking/pdb_to_affinity.txt', f'{args.pignet_data_dir}/docking')
    train_keys3, test_keys3, id_to_y3 = read_data(f'{args.pignet_data_dir}/random/pdb_to_affinity.txt', f'{args.pignet_data_dir}/random')
    train_keys4, test_keys4, id_to_y4 = read_data(f'{args.pignet_data_dir}/screening/pdb_to_affinity.txt', f'{args.pignet_data_dir}/screening')
    processed_data = (train_keys, test_keys, id_to_y, train_keys2, test_keys2, id_to_y2, train_keys3, test_keys3, id_to_y3, train_keys4, test_keys4, id_to_y4)

    # Model
    if args.model == "pignet":
        model = models.PIGNet(args)
    elif args.model == "gnn":
        model = models.GNN(args)
    elif args.model == "cnn3d_kdeep":
        model = models.CNN3D_KDEEP(args)
    else:
        print(f"No {args.model} model")
        exit()

    # if torch.cuda.is_available():
    print('Using CUDA')
    device = torch.device("cuda")
    # else:
    #     print('Using CPU')
    #     device = torch.device("cpu")
    model = utils.initialize_model(model, device, args.restart_file)

    if not args.restart_file:
        n_param = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print("number of parameters : ", n_param)
    return model, device, processed_data

def set_up_predicting(args):
    data = read_training_csv(args.val_csv_file, args.val_data_dir)
    with open(f'temp_features/{args.val_csv_file.split("/")[-1].split(".csv")[0]}/pdb_to_affinity.txt', 'w') as f:
        for key in data.keys():
            f.write(f'{key}\t{data[key][2]}\n')    
    pickle_data(read_training_csv(args.val_csv_file, args.val_data_dir), f'{args.val_csv_file.split("/")[-1].split(".csv")[0]}')
    if not os.path.exists(os.path.join(f'temp_features/{args.val_csv_file.split("/")[-1].split(".csv")[0]}/train_keys.pkl')):
        data = read_training_csv(args.val_csv_file, args.val_data_dir)
        write_keys(list(data.keys()), f'temp_features/{args.val_csv_file.split("/")[-1].split(".csv")[0]}/train_keys.pkl')
    return None

def load_all_dataloaders(args, processed_data):
    train_dataset, train_dataloader = get_dataset_dataloader(processed_data[0], f'temp_features/{args.model_name}', processed_data[2], args.batch_size, args.num_workers)
    # test_dataset, test_dataloader = get_dataset_dataloader(processed_data[1], args.data_dir, processed_data[2], args.batch_size, args.num_workers, False)
    train_dataset2, train_dataloader2 = get_dataset_dataloader(processed_data[3], f'{args.pignet_data_dir}/docking/data', processed_data[5], args.batch_size, args.num_workers)
    # test_dataset2, test_dataloader2 = get_dataset_dataloader(processed_data[4], args.data_dir2, processed_data[5], args.batch_size, args.num_workers, False)
    train_dataset3, train_dataloader3 = get_dataset_dataloader(processed_data[6], f'{args.pignet_data_dir}/random/data', processed_data[8], args.batch_size, args.num_workers)
    # test_dataset3, test_dataloader3 = get_dataset_dataloader(processed_data[7], args.data_dir3, processed_data[8], args.batch_size, args.num_workers, False)
    train_dataset4, train_dataloader4 = get_dataset_dataloader(processed_data[9], f'{args.pignet_data_dir}/screening/data', processed_data[11], args.batch_size, args.num_workers)
    # test_dataset4, test_dataloader4 = get_dataset_dataloader(processed_data[9], args.data_dir4, processed_data[10], args.batch_size, args.num_workers, False)
    return train_dataloader, train_dataloader2, train_dataloader3, train_dataloader4, 
# Optimizer and loss

def run(
    model: nn.Module,
    data_loader,
    data_loader2,
    data_loader3,
    data_loader4,
    train_mode: bool,
    device,
    args,
    # loss_fn,
    optimizer,
) -> None:
    model.train() if train_mode else model.eval()
    losses, losses_der1, losses_der2, losses_docking, losses_screening = (
        [],
        [],
        [],
        [],
        [],
    )
    (
        save_pred,
        save_true,
        save_pred_docking,
        save_true_docking,
        save_pred_screening,
        save_true_screening,
    ) = (dict(), dict(), dict(), dict(), dict(), dict())
    loss_fn = nn.MSELoss()
    i_batch = 0
    # for inputs_1, labels_1, inputs_2, labels_2, inputs_3, labels_3, inputs_4, labels_4 in zip(data_loader, data_loader2, data_loader3, data_loader4):
    for inputs_1, inputs_2, inputs_3, inputs_4 in tqdm(zip(data_loader, data_loader2, data_loader3, data_loader4)):
        # print(inputs_1)
        # print(inputs_1.keys())
        # print('Start', time.time())
        model.zero_grad()

        # sample = next(data_iter, None)
        # if sample is None:
        #     break
        # sample = utils.dic_to_device(sample, device)
        # print('Zero grad', time.time())
        inputs_1 = utils.dic_to_device(inputs_1, device)

        # keys, affinity = sample["key"], sample["affinity"]
        # print('dataset 1 to gpu', time.time())
        loss_all = 0.0
        cal_der_loss = False
        if args.loss_der1_ratio > 0 or args.loss_der2_ratio > 0.0:
            cal_der_loss = True

        pred, loss_der1, loss_der2 = model(**dict(sample=inputs_1, DM_min = 0.5, cal_der_loss =cal_der_loss))
        loss = loss_fn(pred.sum(-1), inputs_1['affinity'])
        loss_der2 = loss_der2.clamp(min=args.min_loss_der2)
        loss_all += loss
        loss_all += loss_der1.sum() * args.loss_der1_ratio
        loss_all += loss_der2.sum() * args.loss_der2_ratio

        # print('loss 1 calc', time.time())

        # loss4
        loss_docking = torch.zeros((1,))
        keys_docking = []
        if args.loss_docking_ratio > 0.0:
            # sample_docking = next(data_iter2, None)
            # sample_docking = utils.dic_to_device(sample_docking, device)
            inputs_2 = utils.dic_to_device(inputs_2, device)
            # print('dataset 2 to gpu', time.time())
            # keys_docking, affinity_docking = (
            #     sample_docking["key"],
            #     sample_docking["affinity"],
            # )
            pred_docking, _, _ = model(**dict(sample=inputs_2, DM_min = 0.5, cal_der_loss =False))
            loss_docking = inputs_2['affinity'] - pred_docking.sum(-1)
            loss_docking = loss_docking.clamp(args.min_loss_docking).mean()
            loss_all += loss_docking * args.loss_docking_ratio

        # print('loss 2 calc', time.time())
        loss_screening = torch.zeros((1,))
        keys_screening = []
        if args.loss_screening_ratio > 0.0:
            # sample_screening = next(data_iter3, None)
            # sample_screening = utils.dic_to_device(sample_screening, device)
            # print('dataset 3 to gpu', time.time())
            inputs_3 = utils.dic_to_device(inputs_3, device)
            # keys_screening, affinity_screening = (
            #     sample_screening["key"],
            #     sample_screening["affinity"],
            # )
            pred_screening, _, _ = model(**dict(sample=inputs_3, DM_min = 0.5, cal_der_loss =False))
            loss_screening = inputs_3['affinity'] - pred_screening.sum(-1)
            loss_screening = loss_screening.clamp(min=0.0).mean()
            loss_all += loss_screening * args.loss_screening_ratio
        # print('loss 3 calc', time.time())
        loss_screening2 = torch.zeros((1,))
        keys_screening2 = []
        if args.loss_screening2_ratio > 0.0:
            # sample_screening2 = next(data_iter4, None)
            # sample_screening2 = utils.dic_to_device(sample_screening2, device)
            # keys_screening2, affinity_screening2 = (
            #     sample_screening2["key"],
            #     sample_screening2["affinity"],
            # )
            inputs_4 = utils.dic_to_device(inputs_4, device)
            # print('dataset 4 to gpu', time.time())
            pred_screening2, _, _ = model(**dict(sample=inputs_4, DM_min = 0.5, cal_der_loss =False))
            loss_screening2 = inputs_4['affinity'] - pred_screening2.sum(-1)
            loss_screening2 = loss_screening2.clamp(min=0.0).mean()
            loss_all += loss_screening2 * args.loss_screening2_ratio
        # print('All losses calc', time.time())
        if train_mode:
            loss_all.backward()
            optimizer.step()
        # print('backwards step done', time.time())
    return None

def train_model(args):
    model, device, processed_data = set_up_training(args)
    train_dataloader, train_dataloader2,  train_dataloader3, train_dataloader4 = load_all_dataloaders(args, processed_data)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    

    # train
    # writer = SummaryWriter(args.tensorboard_dir)
    if args.restart_file:
        restart_epoch = int(args.restart_file.split("_")[-1].split(".")[0])
    else:
        restart_epoch = 0
    for epoch in tqdm(range(restart_epoch, args.num_epochs)):
        st = time.time()
        tmp_st = st

        (
            train_losses,
            train_losses_der1,
            train_losses_der2,
            train_losses_docking,
            train_losses_screening,
        ) = ([], [], [], [], [])
        (
            train_pred,
            train_true,
            train_pred_docking,
            train_true_docking,
            train_pred_screening,
            train_true_screening,
        ) = (dict(), dict(), dict(), dict(), dict(), dict())
        run(
            model,
            train_dataloader,
            train_dataloader2,
            train_dataloader3,
            train_dataloader4,
            True,
            device,
            args,
            # loss_fn,
            optimizer,
        )
        name = os.path.join(f'temp_models/{args.model_name}.pt')
        save_every = 1 if not args.save_every else args.save_every
        if epoch % save_every == 0:
            torch.save(model.state_dict(), name)

        lr = args.lr * ((args.lr_decay) ** epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    

def predict(args, model, test_dataloader, device):
    model.eval()
    all_predictions = []
    all_affinity = []
    all_keys = []
    for data in test_dataloader:
        model.zero_grad()
        keys, affinity = data["key"], data["affinity"]
        data = utils.dic_to_device(data, device)
        pred, _, _ = model(**dict(sample=data, DM_min = 0.5, cal_der_loss =False))
        all_keys = all_keys + list(keys)
        all_affinity = all_affinity + list(affinity)
        all_predictions = all_predictions + list(pred.detach().cpu().numpy()[0])
    return pd.DataFrame({'key': all_keys, 'pred': all_predictions, 'pk': all_affinity})
    

if __name__ == "__main__":
    # print('test')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='train.csv')
    parser.add_argument('--val_csv_file', type=str, default='val.csv')
    parser.add_argument('--pignet_data_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--val_data_dir', type=str, default='data')
    parser.add_argument('--model_name', type=str, default='test')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true') 
    parser.add_argument("--dim_gnn",help="dim_gnn",type=int,default=128,)
    parser.add_argument("--n_gnn",help="depth of gnn layer",type=int,default=3,)
    parser.add_argument("--ngpu",help="ngpu",type=int,default=1,)
    parser.add_argument("--restart_file",help="restart file",type=str,)
    parser.add_argument("--model",help="model",type=str,default="pignet",choices=["pignet", "gnn", "cnn3d_kdeep"],)
    parser.add_argument("--interaction_net",action="store_true",help="interaction_net",)
    parser.add_argument("--no_rotor_penalty",action="store_true",help="rotor penaly",)
    parser.add_argument("--dropout_rate",help="dropout rate",type=float,default=0.0,)
    parser.add_argument("--vdw_N",help="vdw N",type=float,default=6.0,)
    parser.add_argument("--max_vdw_interaction",help="max vdw _interaction",type=float,default=0.0356,)
    parser.add_argument("--min_vdw_interaction",help="min vdw _interaction",type=float,default=0.0178,)
    parser.add_argument("--dev_vdw_radius",help="deviation of vdw radius",type=float,default=0.2,)
    parser.add_argument("--scaling",type=float,default=1.0,)
    parser.add_argument("--lattice_dim",type=int,default=24,)
    parser.add_argument( "--grid_rotation", action="store_true",)
    parser.add_argument("--batch_size",help="batch size",type=int,default=8,)
    parser.add_argument("--num_workers",help="number of workers",type=int,default=4,)
    parser.add_argument("--lr",help="learning rate",type=float,default=1e-4,)
    parser.add_argument("--lr_decay",help="learning rate decay",type=float,default=1.0,)
    parser.add_argument("--weight_decay",help="weight decay",type=float,default=0.0,)
    parser.add_argument("--num_epochs",help="number of epochs",type=int,default=2300,)
    parser.add_argument("--loss_der1_ratio",help="loss der1 ratio",type=float,default=10.0,)
    parser.add_argument("--loss_der2_ratio",help="loss der2 ratio",type=float,default=10.0)
    parser.add_argument("--min_loss_der2",help="min loss der2",type=float,default=-20.0,)
    parser.add_argument("--loss_docking_ratio",help="loss docking ratio",type=float,default=10.0,)
    parser.add_argument("--min_loss_docking",help="min loss docking",type=float,default=-1.0,)
    parser.add_argument("--loss_screening_ratio",help="loss screening ratio",type=float,default=5.0,)
    parser.add_argument("--loss_screening2_ratio",help="loss screening ratio",type=float,default=5.0,)
    parser.add_argument("--save_dir",help="save directory of model save files",type=str,default="save",)
    parser.add_argument("--save_every",help="saver every n epoch",type=int,default=1,)
    parser.add_argument("--tensorboard_dir",help="save directory of tensorboard log files",type=str,)
    args = parser.parse_args()
    if args.train:
        if not os.path.exists(f'temp_features/{args.model_name}'):
            os.mkdir(f'temp_features/{args.model_name}')
        get_files(args.pignet_data_dir)
        train_model(args)
    elif args.predict:
        if not os.path.exists(f'temp_features/{args.val_csv_file.split("/")[-1].split(".csv")[0]}'):
            os.mkdir(f'temp_features/{args.val_csv_file.split("/")[-1].split(".csv")[0]}')
        set_up_predicting(args)
        train_keys, _, id_to_y = read_data(f'temp_features/{args.val_csv_file.split("/")[-1].split(".csv")[0]}/pdb_to_affinity.txt',f'temp_features/{args.val_csv_file.split("/")[-1].split(".csv")[0]}')
        _, val_dataloader = get_dataset_dataloader(train_keys,f'temp_features/{args.val_csv_file.split("/")[-1].split(".csv")[0]}', id_to_y, args.batch_size, args.num_workers)
        print('Using CUDA')
        device = torch.device("cuda")
        if args.model == "pignet":
            model = models.PIGNet(args)
        elif args.model == "gnn":
            model = models.GNN(args)
        elif args.model == "cnn3d_kdeep":
            model = models.CNN3D_KDEEP(args)
        else:
            print(f"No {args.model} model")
            exit()
        # model.load_state_dict(torch.load(f'save/PIGNet/save_2300.pt'))
        model = utils.initialize_model(model, device, 'save/PIGNet/save_2300.pt')
        # model.to(device)
        df = predict(args, model, val_dataloader, device)
        df.to_csv(f'results/{args.model_name}_{args.val_csv_file.split("/")[-1].split(".csv")[0]}.csv')
            
