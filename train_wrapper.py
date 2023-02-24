import os
import sys
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from tqdm import tqdm
from data.generate_keys import write_keys
from train import run
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
    else:
        pass

    # Read labels
    train_keys, test_keys, id_to_y = read_data(f'temp_features/{args.model_name}/pdb_to_affinity.txt',f'temp_features/{args.model_name}')
    train_keys2, test_keys2, id_to_y2 = read_data(f'{args.pignet_data_dir}/docking/pdb_to_affinity.txt', f'{args.pignet_data_dir}/docking')
    train_keys3, test_keys3, id_to_y3 = read_data(f'{args.pignet_data_dir}/random/pdb_to_affinity.txt', f'{args.pignet_data_dir}/random')
    train_keys4, test_keys4, id_to_y4 = read_data(f'{args.pignet_data_dir}/screening/pdb_to_affinity.txt', f'{args.pignet_data_dir}/cross')
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.initialize_model(model, device, args.restart_file)

    if not args.restart_file:
        n_param = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print("number of parameters : ", n_param)
    return model, device, processed_data

def load_all_dataloaders(args, processed_data):
    train_dataset, train_dataloader = get_dataset_dataloader(processed_data[0], f'temp_features/{args.model_name}', processed_data[2], args.batch_size, args.num_workers)
    # test_dataset, test_dataloader = get_dataset_dataloader(processed_data[1], args.data_dir, processed_data[2], args.batch_size, args.num_workers, False)
    train_dataset2, train_dataloader2 = get_dataset_dataloader(processed_data[3], f'{args.pignet_data_dir}/docking/data', processed_data[5], args.batch_size, args.num_workers)
    # test_dataset2, test_dataloader2 = get_dataset_dataloader(processed_data[4], args.data_dir2, processed_data[5], args.batch_size, args.num_workers, False)
    train_dataset3, train_dataloader3 = get_dataset_dataloader(processed_data[6], f'{args.pignet_data_dir}/random/data', processed_data[8], args.batch_size, args.num_workers)
    # test_dataset3, test_dataloader3 = get_dataset_dataloader(processed_data[7], args.data_dir3, processed_data[8], args.batch_size, args.num_workers, False)
    train_dataset4, train_dataloader4 = get_dataset_dataloader(processed_data[8], f'{args.pignet_data_dir}/screening/data', processed_data[10], args.batch_size, args.num_workers)
    # test_dataset4, test_dataloader4 = get_dataset_dataloader(processed_data[9], args.data_dir4, processed_data[10], args.batch_size, args.num_workers, False)
    return train_dataloader, train_dataloader2, train_dataloader3, train_dataloader4, 
# Optimizer and loss

def train_model(args):
    model, device, processed_data = set_up_training(args)
    train_dataloader, train_dataloader2,  train_dataloader3, train_dataloader4 = load_all_dataloaders(args)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()

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
        # (
        #     test_losses,
        #     test_losses_der1,
        #     test_losses_der2,
        #     test_losses_docking,
        #     test_losses_screening,
        # ) = ([], [], [], [], [])

        (
            train_pred,
            train_true,
            train_pred_docking,
            train_true_docking,
            train_pred_screening,
            train_true_screening,
        ) = (dict(), dict(), dict(), dict(), dict(), dict())
        # (
        #     test_pred,
        #     test_true,
        #     test_pred_docking,
        #     test_true_docking,
        #     test_pred_screening,
        #     test_true_screening,
        # ) = (dict(), dict(), dict(), dict(), dict(), dict())

        # iterator
        train_data_iter, train_data_iter2, train_data_iter3, train_data_iter4 = (
            iter(train_dataloader),
            iter(train_dataloader2),
            iter(train_dataloader3),
            iter(train_dataloader4),
        )
        # test_data_iter, test_data_iter2, test_data_iter3, test_data_iter4 = (
        #     iter(test_dataloader),
        #     iter(test_dataloader2),
        #     iter(test_dataloader3),
        #     iter(test_dataloader4),
        # )

        # Train
        (
            train_losses,
            train_losses_der1,
            train_losses_der2,
            train_losses_docking,
            train_losses_screening,
            train_pred,
            train_true,
            train_pred_docking,
            train_true_docking,
            train_pred_screening,
            train_true_screening,
        ) = run(
            model,
            train_data_iter,
            train_data_iter2,
            train_data_iter3,
            train_data_iter4,
            True,
        )

        # Test
        # (
        #     test_losses,
        #     test_losses_der1,
        #     test_losses_der2,
        #     test_losses_docking,
        #     test_losses_screening,
        #     test_pred,
        #     test_true,
        #     test_pred_docking,
        #     test_true_docking,
        #     test_pred_screening,
        #     test_true_screening,
        # ) = run(
        #     model,
        #     test_data_iter,
        #     test_data_iter2,
        #     test_data_iter3,
        #     test_data_iter4,
        #     False,
        # )

        # Write tensorboard
        # writer.add_scalars(
        #     "train",
        #     {
        #         "loss": train_losses,
        #         "loss_der1": train_losses_der1,
        #         "loss_der2": train_losses_der2,
        #         "loss_docking": train_losses_docking,
        #         "loss_screening": train_losses_screening,
        #     },
        #     epoch,
        # )
        # writer.add_scalars(
        #     "test",
        #     {
        #         "loss": test_losses,
        #         "loss_der1": test_losses_der1,
        #         "loss_der2": test_losses_der2,
        #         "loss_docking": test_losses_docking,
        #         "loss_screening": test_losses_screening,
        #     },
        #     epoch,
        # )

        # # Write prediction
        # utils.write_result(
        #     .args.train_result_filename,
        #     train_pred,
        #     train_true,
        # )
        # utils.write_result(
        #     .args.test_result_filename,
        #     test_pred,
        #     test_true,
        # )
        # utils.write_result(
        #     .args.train_result_docking_filename,
        #     train_pred_docking,
        #     train_true_docking,
        # )
        # utils.write_result(
        #     args.test_result_docking_filename,
        #     test_pred_docking,
        #     test_true_docking,
        # )
        # utils.write_result(
        #     args.train_result_screening_filename,
        #     train_pred_screening,
        #     train_true_screening,
        # )
        # utils.write_result(
        #     args.test_result_screening_filename,
        #     test_pred_screening,
        #     test_true_screening,
        # )
        # end = time.time()

        # Cal R2
        train_r2 = r2_score(
            [train_true[k] for k in train_true.keys()],
            [train_pred[k].sum() for k in train_true.keys()],
        )
        # test_r2 = r2_score(
        #     [test_true[k] for k in test_true.keys()],
        #     [test_pred[k].sum() for k in test_true.keys()],
        # )

        # Cal R
        # _, _, test_r, _, _ = stats.linregress(
        #     [test_true[k] for k in test_true.keys()],
        #     [test_pred[k].sum() for k in test_true.keys()],
        # )
        _, _, train_r, _, _ = stats.linregress(
            [train_true[k] for k in train_true.keys()],
            [train_pred[k].sum() for k in train_true.keys()],
        )
        end = time.time()
        # if epoch == 0:
        #     print(
        #         "epoch\ttrain_l\ttrain_l_der1\ttrain_l_der2\ttrain_l_docking\t"
        #         + "train_l_screening\ttest_l\ttest_l_der1\ttest_l_der2\t"
        #         + "test_l_docking\ttest_l_screening\t"
        #         + "train_r2\ttest_r2\ttrain_r\ttest_r\ttime"
        #     )
        # print(
        #     f"{epoch}\t{train_losses:.3f}\t{train_losses_der1:.3f}\t"
        #     + f"{train_losses_der2:.3f}\t{train_losses_docking:.3f}\t"
        #     + f"{train_losses_screening:.3f}\t"
        #     + f"{test_losses:.3f}\t{test_losses_der1:.3f}\t"
        #     + f"{test_losses_der2:.3f}\t{test_losses_docking:.3f}\t"
        #     + f"{test_losses_screening:.3f}\t"
        #     + f"{train_r2:.3f}\t{test_r2:.3f}\t"
        #     + f"{train_r:.3f}\t{test_r:.3f}\t{end-st:.3f}"
        # )

        name = os.path.join(f'temp_models/{args.model_name}.pt')
        save_every = 1 if not args.save_every else args.save_every
        if epoch % save_every == 0:
            torch.save(model.state_dict(), name)

        lr = args.lr * ((args.lr_decay) ** epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    


if __name__ == "__main__":
    print('test')
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
    parser.add_argument("--num_epochs",help="number of epochs",type=int,default=3001,)
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
        # Check PigNet files exist
        get_files(args.pignet_data_dir)
        # Check featurised data exists

        train_model(args)
    elif args.predict:
        pass
