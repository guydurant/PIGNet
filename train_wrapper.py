import os
import sys
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from tqdm import tqdm
from data.generate_keys import write_keys
from train import run
import arguments
import utils
import models
import torch
import torch.nn as nn
from dataset import get_dataset_dataloader
import time
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score

# ARGUMENTS CAN BE DEFINED OR BY CONFIG - CAN ONLY DO ONE - raise error if try both
# UNIT Tests - if can start running, then quit after 1 minute
# If Pignet extra data does not affect performance, put in .sif file

# Prepare data
# Download cross, docking, random files if do not exist

class PigNetWrapper:
    #Should be initialised AND THEN run training 
    def __init__(self, args, pignet_data_dir, training_csv):
        self.args = args
        self.pignet_data_dir = pignet_data_dir
        self.training_csv = training_csv


    def process_tar_files(self):
        os.system('tar -xzf data.tar.gz')
        # Change function to generate_keys just from the csv file - So new generate_keys_wrapper.py
        os.system('../../generate_keys.py -d data -k keys -c ../../coreset_keys.txt --train')
        # Create csv file with all of the values - so create new pdb_to_affinity_wrapper.py
        os.system('../../pdb_to_affinity.py -d data -f pdb_to_affinity.txt -i ../../INDEX_refined_data.2019')
        os.system('cd -')
        return 

    def get_files(self):
        print('Downloading extra datasets for PigNet')
        # - Specify path to pignet external data as PATH
        # - Check if overall folders exists
        # - If not download them using the wget commands (os.system)
        if not os.pathexists(f'{self.pignet_data_dir}/docking/data'):
            os.system(f'wget https://zenodo.org/record/6047984/files/pdbbind_v2019_docking.tar.gz?download=1 -O {self.DOCKING_DIR}/data.tar.gz')
            os.system(f'cd {self.pignet_data_dir}/docking/data')
            self.process_tar_files
        if not os.pathexists(f'{self.pignet_data_dir}/random/data'):
            os.system(f'wget https://zenodo.org/record/6047984/files/pdbbind_v2019_random_screening.tar.gz?download=1 -O {self.RANDOM_DIR}/data.tar.gz')
            os.system(f'cd {self.pignet_data_dir}/random/data')
            self.process_tar_files
        if not os.pathexists(f'{self.pignet_data_dir}/screening/data'):
            os.system(f'wget https://zenodo.org/record/6047984/files/pdbbind_v2019_cross_screening.tar.gz?download=1 -O {self.CROSS_DIR}/data.tar.gz')
            os.system(f'cd {self.pignet_data_dir}/screening/data')
            self.process_tar_files
        return None
    
    def read_training_csv(self):
        data = pd.read_csv(self.training_csv)
        proteins, ligands, pks, keys = data['proteins'].to_list(), data['ligands'].to_list(), data['pK'].to_list(), data['key'].to_list()
        # Add relative path (self.data_dir)
        proteins = [os.path.join(self.data_dir, protein) for protein in proteins]
        ligands = [os.path.join(self.data_dir, ligand) for ligand in ligands]
        pks = [float(pk) for pk in pks]
        self.data = {{key: [protein, ligand, pk] for key, protein, ligand, pk in zip(keys, proteins, ligands, pks)}}
        return None
        # Need to pickle them all - going to have to invent a way to do it
        # <rdkit.Chem.rdchem.Mol object at 0x7fd0992931b8>, 0, <rdkit.Chem.rdchem.Mol object at 0x7fd0992931f0>, [])
        # Could just read them in as a PDBs using RDKit and then pickle them
    
    def get_residues(self, protein_file_name, ligand_file_name):
        # Load ligand and protein
        ligand = Chem.MolFromMolFile(ligand_file_name)
        lig_conf = ligand.GetConformer()
        protein = Chem.MolFromPDBFile(protein_file_name)
        protein_conf = protein.GetConformer()
        # Get residues within 5A of ligand
        residues = []
        for atom in protein.GetAtoms():
            atom_coords = protein_conf.GetAtomPosition(atom.GetIdx())
            if atom.GetAtomicNum() == 1:
                continue
            for ligand_atom in ligand.GetAtoms():
                if ligand_atom.GetAtomicNum() == 1:
                    continue
                else:
                    lig_coord = lig_conf.GetAtomPosition(ligand_atom.GetIdx())
                    if np.linalg.norm(atom_coords - lig_coord) < 5:
                        residues.append((atom.GetPDBResidueInfo().GetResidueNumber(), atom.GetPDBResidueInfo().GetResidueName(), atom.GetPDBResidueInfo().GetChainId()))
                        break
        return residues

    def create_pignet_pocket_file(self, protein_file_name, ligand_file_name):
        residues = list(set(self.get_residues(protein_file_name, ligand_file_name)))
        # print(residues)
        with open(protein_file_name, 'r') as f:
            lines = f.readlines()
        pocket_lines = []
        for l in lines:
            if l.split()[0] == 'ATOM':
                # print(l.split()[1], l.split()[3], l.split()[4])
                if (int(l.split()[5]), l.split()[3], l.split()[4]) in residues:
                    pocket_lines.append(l)
                    # print(l)
        return pocket_lines

    def pickle_data(self):
        print('Pickling data for scoring')
        for key in tqdm(self.data.keys()):
            #if path exists
            if os.path.exists(os.path.join(self.pignet_data_dir+'/scoring/data', key)):
                continue
            protein_pocket_mol = Chem.MolFromPDBBlock(self.create_pignet_pocket_file(self.data[key][0], self.data[key][1]))
            ligand_mol = Chem.MolFromMolFile(self.data[key][1])
            pickle.dump((ligand_mol, 0, protein_pocket_mol, []), open(os.path.join(self.pignet_data_dir+'/scoring/data', key), 'wb'))
        return None
    
    def generate_pdb_to_affinity(self, mode='scoring'):
        if mode == 'screening':
            with open(os.path.join(self.pignet_data_dir, mode, 'pdb_to_affinity.txt'), 'w') as f:
                for key in os.listdir(f'{self.pignet_data_dir}/screening/data'):
                    f.write(f'{key}\t5\n')
        else:
            with open(os.path.join(self.pignet_data_dir, mode, 'pdb_to_affinity.txt'), 'w') as f:
                for key in os.listdir(f'{self.pignet_data_dir}/{mode}/data'):
                    f.write(f'{key}\t{self.data[key][2]}\n')
        return None
    
    def generate_all_pdb_to_affinity(self):
        print('Generating pdb_to_affinity.txt files')
        for mode in ['scoring', 'docking', 'cross', 'screening']:
            if not os.path.exists(os.path.join(self.pignet_data_dir, mode, 'pdb_to_affinity.txt')):
                    self.generate_pdb_to_affinity(mode)
    
    def generate_keys(self):
        core_keys = [i for i in open('data/core_keys.txt', 'r').read().split('\n') if i != '']
        for mode in ['docking', 'cross', 'screening']:
            if not os.path.exists(os.path.join(self.pignet_data_dir, mode, 'keys', 'train_keys.pkl')):
                keys = os.listdir(f'{self.pignet_data_dir}/{mode}/data')
                train_keys = []
                test_keys = []
                for key in keys:
                    if key.split("_")[0] in core_keys:
                        test_keys.append(key)
                    else:
                        train_keys.append(key)
                write_keys(train_keys, os.path.join(self.pignet_data_dir, mode, 'keys', 'train_keys.pkl'))
                write_keys(test_keys, os.path.join(self.pignet_data_dir, mode, 'keys', 'train_keys.pkl'))
        # Now for scoring mode
        if not os.path.exists(os.path.join(self.pignet_data_dir, 'scoring', 'keys', 'train_keys.pkl')):
            write_keys(self.keys, os.path.join(self.pignet_data_dir, 'scoring', 'keys', 'train_keys.pkl'))   
        return None  

    def set_up_training(self):
        os.makedirs(self.save_dir, exist_ok=True)
        # os.makedirs(args.tensorboard_dir, exist_ok=True)
        if os.path.dirname(self.train_result_filename):
            os.makedirs(os.path.dirname(self.train_result_filename), exist_ok=True)

        # Set GPU
        if args.ngpu > 0:
            cmd = utils.set_cuda_visible_device(args.ngpu)
            os.environ["CUDA_VISIBLE_DEVICES"] = cmd
        else:
            pass

        # Read labels
        train_keys, test_keys, id_to_y = utils.read_data(self.filename, self.key_dir)
        train_keys2, test_keys2, id_to_y2 = utils.read_data(self.filename2, self.key_dir2)
        train_keys3, test_keys3, id_to_y3 = utils.read_data(self.filename3, self.key_dir3)
        train_keys4, test_keys4, id_to_y4 = utils.read_data(self.filename4, self.key_dir4)
        self.processed_data = (train_keys, test_keys, id_to_y, train_keys2, test_keys2, id_to_y2, train_keys3, test_keys3, id_to_y3, train_keys4, test_keys4, id_to_y4)

        # Model
        if self.args.model == "pignet":
            self.model = models.PIGNet(self.args)
        elif self.args.model == "gnn":
            self.model = models.GNN(self.args)
        elif args.model == "cnn3d_kdeep":
            self.model = models.CNN3D_KDEEP(self.args)
        else:
            print(f"No {self.args.model} model")
            exit()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = utils.initialize_model(self.model, self.device, self.args.restart_file)

        if not args.restart_file:
            n_param = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
            print("number of parameters : ", n_param)
        return None
    
    def load_all_dataloaders(self):
        train_dataset, train_dataloader = get_dataset_dataloader(self.processed_data[0], args.data_dir, self.processed_data[2], args.batch_size, args.num_workers)
        test_dataset, test_dataloader = get_dataset_dataloader(self.processed_data[1], args.data_dir, self.processed_data[2], args.batch_size, args.num_workers, False)
        train_dataset2, train_dataloader2 = get_dataset_dataloader(self.processed_data[3], args.data_dir2, self.processed_data[5], args.batch_size, args.num_workers)
        test_dataset2, test_dataloader2 = get_dataset_dataloader(self.processed_data[4], args.data_dir2, self.processed_data[5], args.batch_size, args.num_workers, False)
        train_dataset3, train_dataloader3 = get_dataset_dataloader(self.processed_data[6], args.data_dir3, self.processed_data[8], args.batch_size, args.num_workers)
        test_dataset3, test_dataloader3 = get_dataset_dataloader(self.processed_data[7], args.data_dir3, self.processed_data[8], args.batch_size, args.num_workers, False)
        train_dataset4, train_dataloader4 = get_dataset_dataloader(self.processed_data[8], args.data_dir4, self.processed_data[10], args.batch_size, args.num_workers)
        test_dataset4, test_dataloader4 = get_dataset_dataloader(self.processed_data[9], args.data_dir4, self.processed_data[10], args.batch_size, args.num_workers, False)
        return train_dataloader, test_dataloader, train_dataloader2, test_dataloader2, train_dataloader3, test_dataloader3, train_dataloader4, test_dataloader4
# Optimizer and loss

    def train_model(self):
        train_dataloader, test_dataloader, train_dataloader2, test_dataloader2, train_dataloader3, test_dataloader3, train_dataloader4, test_dataloader4 = self.load_all_dataloaders()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        loss_fn = nn.MSELoss()

        # train
        # writer = SummaryWriter(args.tensorboard_dir)
        if self.args.restart_file:
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
                test_losses,
                test_losses_der1,
                test_losses_der2,
                test_losses_docking,
                test_losses_screening,
            ) = ([], [], [], [], [])

            (
                train_pred,
                train_true,
                train_pred_docking,
                train_true_docking,
                train_pred_screening,
                train_true_screening,
            ) = (dict(), dict(), dict(), dict(), dict(), dict())
            (
                test_pred,
                test_true,
                test_pred_docking,
                test_true_docking,
                test_pred_screening,
                test_true_screening,
            ) = (dict(), dict(), dict(), dict(), dict(), dict())

            # iterator
            train_data_iter, train_data_iter2, train_data_iter3, train_data_iter4 = (
                iter(train_dataloader),
                iter(train_dataloader2),
                iter(train_dataloader3),
                iter(train_dataloader4),
            )
            test_data_iter, test_data_iter2, test_data_iter3, test_data_iter4 = (
                iter(test_dataloader),
                iter(test_dataloader2),
                iter(test_dataloader3),
                iter(test_dataloader4),
            )

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
                self.model,
                train_data_iter,
                train_data_iter2,
                train_data_iter3,
                train_data_iter4,
                True,
            )

            # Test
            (
                test_losses,
                test_losses_der1,
                test_losses_der2,
                test_losses_docking,
                test_losses_screening,
                test_pred,
                test_true,
                test_pred_docking,
                test_true_docking,
                test_pred_screening,
                test_true_screening,
            ) = run(
                self.model,
                test_data_iter,
                test_data_iter2,
                test_data_iter3,
                test_data_iter4,
                False,
            )

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

            # Write prediction
            utils.write_result(
                self.args.train_result_filename,
                train_pred,
                train_true,
            )
            utils.write_result(
                self.args.test_result_filename,
                test_pred,
                test_true,
            )
            utils.write_result(
                self.args.train_result_docking_filename,
                train_pred_docking,
                train_true_docking,
            )
            utils.write_result(
                args.test_result_docking_filename,
                test_pred_docking,
                test_true_docking,
            )
            utils.write_result(
                args.train_result_screening_filename,
                train_pred_screening,
                train_true_screening,
            )
            utils.write_result(
                args.test_result_screening_filename,
                test_pred_screening,
                test_true_screening,
            )
            end = time.time()

            # Cal R2
            train_r2 = r2_score(
                [train_true[k] for k in train_true.keys()],
                [train_pred[k].sum() for k in train_true.keys()],
            )
            test_r2 = r2_score(
                [test_true[k] for k in test_true.keys()],
                [test_pred[k].sum() for k in test_true.keys()],
            )

            # Cal R
            _, _, test_r, _, _ = stats.linregress(
                [test_true[k] for k in test_true.keys()],
                [test_pred[k].sum() for k in test_true.keys()],
            )
            _, _, train_r, _, _ = stats.linregress(
                [train_true[k] for k in train_true.keys()],
                [train_pred[k].sum() for k in train_true.keys()],
            )
            end = time.time()
            if epoch == 0:
                print(
                    "epoch\ttrain_l\ttrain_l_der1\ttrain_l_der2\ttrain_l_docking\t"
                    + "train_l_screening\ttest_l\ttest_l_der1\ttest_l_der2\t"
                    + "test_l_docking\ttest_l_screening\t"
                    + "train_r2\ttest_r2\ttrain_r\ttest_r\ttime"
                )
            print(
                f"{epoch}\t{train_losses:.3f}\t{train_losses_der1:.3f}\t"
                + f"{train_losses_der2:.3f}\t{train_losses_docking:.3f}\t"
                + f"{train_losses_screening:.3f}\t"
                + f"{test_losses:.3f}\t{test_losses_der1:.3f}\t"
                + f"{test_losses_der2:.3f}\t{test_losses_docking:.3f}\t"
                + f"{test_losses_screening:.3f}\t"
                + f"{train_r2:.3f}\t{test_r2:.3f}\t"
                + f"{train_r:.3f}\t{test_r:.3f}\t{end-st:.3f}"
            )

            name = os.path.join(args.save_dir, "save_" + str(epoch) + ".pt")
            save_every = 1 if not args.save_every else args.save_every
            if epoch % save_every == 0:
                torch.save(self.model.state_dict(), name)

            lr = args.lr * ((args.lr_decay) ** epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        


if __name__ == "__main__":
    # take in args, depending on whether a config file is used or not
    print(sys.argv)
    raw_arguments = sys.argv
    wrapper_args_names = ['pignet_data_dir', 'training_csv']
    pignet_args = [i for i in raw_arguments if i not in wrapper_args]
    wrapper_args = [i for i in raw_arguments if i in wrapper_args_names]
    print(wrapper_args)
    args = arguments.parser(pignet_args)
    if not args.restart_file:
        print(args)
    # Initilaise PigNetWrapper

    pignet_model = PigNetWrapper(args, **wrapper_args)

    # Now train model

    pignet_model.train()


        


# Read in csv for scoring function
# - Specifiy path for ANALYSESF data


# Convert files to correct format
# - pdb_to_affinity.py
# - generate_keys.py
# - Keep keys and values in PigNet external data 


# Train model

#Use commmand from run.sh




# Save weights to pignet weights folder

