import os
import torch 
import numpy as np
from torch import nn
import random 
from models import SLViT, SplitNetwork
from dataset import bloodmnisit
import argparse 
from utils import weight_dec_global

def slvit(lr, batch_size, Epochs, input_size, num_workers,
          save_every_epochs, model_name, pretrained, opt_name, seed,
          base_dir, root_dir, csv_file_path, num_clients, DP, epsilon, delta):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # DP settings
    mean = 0 
    std  = 1
    if DP:
        std = np.sqrt(2 * np.math.log(1.25/delta)) / epsilon 

    # SAVE DIR
    save_dir = f'{model_name}_{lr}lr_bloodmnist_{num_clients}Clients_{DP}DP_{batch_size}Batch_SLViT'
    if DP:
        save_dir = f'{model_name}_{lr}lr_bloodmnist_{num_clients}Clients_({epsilon},{delta})DP_{batch_size}Batch_SLViT'
    
    os.mkdir(save_dir)

    print('Getting the BloodMNIST Dataset and Dataloader!')

    # BLOODMNIST ONLY
    num_classes = 8
    num_channels = 3
    _, _, traindataset, testdataset = bloodmnisit(
        input_size=input_size, 
        batch_size=batch_size,
        download=True,
        num_workers=num_workers
    )

    # Initialize Model
    slvit_model = SLViT(
        ViT_name=model_name,
        num_classes=num_classes,
        num_clients=num_clients,
        in_channels=num_channels,
        ViT_pretrained=pretrained,
        diff_privacy=DP,
        mean=mean,
        std=std
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    Split = SplitNetwork(
        num_clients=num_clients, 
        device=device, 
        network=slvit_model, 
        criterion=criterion,
        base_dir=save_dir,
    )

    print('Distribute Data (BloodMNIST)')
    Split.distribute_images(
        dataset_name='bloodmnist',
        train_data=traindataset,
        test_data=testdataset,
        batch_size=batch_size
    )

    Split.set_optimizer(opt_name, lr=lr)
    Split.init_logs()

    for r in range(Epochs):
        print(f"Round {r+1} / {Epochs}")

        agg_weights = None

        # Federated Round
        for client_i in range(num_clients):
            weight_dict = Split.train_round(client_i)
            if client_i == 0:
                agg_weights = weight_dict
            else:
                agg_weights['blocks'] += weight_dict['blocks']
                agg_weights['cls'] += weight_dict['cls']
                agg_weights['pos_embed'] += weight_dict['pos_embed']
        
        # Average Weights
        agg_weights['blocks'] /= num_clients
        agg_weights['cls'] /= num_clients
        agg_weights['pos_embed'] /= num_clients

        # Update global model
        Split.network.vit.blocks = weight_dec_global(
            Split.network.vit.blocks,
            agg_weights['blocks'].to(device)
        )
        
        Split.network.vit.cls_token.data = agg_weights['cls'].to(device).clone()
        Split.network.vit.pos_embed.data = agg_weights['pos_embed'].to(device).clone()

        # Eval
        for client_i in range(num_clients):
            Split.eval_round(client_i)

        print('---------')

        # Save every few rounds
        if (r+1) % save_every_epochs == 0:
            Split.save_pickles(save_dir)

        print('============================================')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run SLViT on BloodMNIST Only')

    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_clients', type=int, default=6)
    parser.add_argument('--model_name', type=str, default='vit_base_r50_s16_224')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--Epochs', type=int, default=200)
    parser.add_argument('--opt_name', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_every_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=105)
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--csv_file_path', type=str, default=None)
    parser.add_argument('--DP', type=bool, default=False)
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--delta', type=float, default=0.00001)

    args = parser.parse_args()

    slvit(
        lr=args.lr,
        batch_size=args.batch_size,
        Epochs=args.Epochs,
        input_size=args.input_size,
        num_workers=args.num_workers,
        save_every_epochs=args.save_every_epochs,
        model_name=args.model_name,
        pretrained=args.pretrained,
        opt_name=args.opt_name,
        seed=args.seed,
        base_dir=args.base_dir,
        root_dir=args.root_dir,
        csv_file_path=args.csv_file_path,
        num_clients=args.num_clients,
        DP=args.DP,
        epsilon=args.epsilon,
        delta=args.delta
    )
