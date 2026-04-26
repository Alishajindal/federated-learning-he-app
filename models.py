from tqdm import tqdm
import pickle as pkl
import os
import timm
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn as nn
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from dataset import blood_noniid, bloodmnist, distribute_data
from utils import weight_vec


class SLViT(nn.Module):
    def __init__(
        self, ViT_name, num_classes, num_clients=6,
        in_channels=3, ViT_pretrained=False,
        diff_privacy=False, mean=0, std=1
    ) -> None:
        super().__init__()

        self.vit = timm.create_model(
            model_name=ViT_name,
            pretrained=ViT_pretrained,
            num_classes=num_classes,
            in_chans=in_channels
        )
        client_tail = MLP_cls_classes(num_classes=num_classes)
        self.mlp_clients_tail = nn.ModuleList([copy.deepcopy(client_tail) for i in range(num_clients)])
        self.resnet50_clients = nn.ModuleList([copy.deepcopy(self.vit.patch_embed) for i in range(num_clients)])

        self.diff_privacy = diff_privacy
        self.mean = mean
        self.std = std

    def forward(self, x, client_idx):
        x = self.resnet50_clients[client_idx](x)
        if self.diff_privacy:
            noise = torch.randn(size=x.shape).to(x.device) * self.std + self.mean
            x = x + noise
        x = torch.cat((self.vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        for block_num in range(len(self.vit.blocks)):
            x = self.vit.blocks[block_num](x)
        x = self.vit.norm(x)
        cls = self.vit.pre_logits(x)[:, 0, :]
        x = self.mlp_clients_tail[client_idx](cls)
        return x, cls


class MLP_cls_classes(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.identity = nn.Identity()
        self.fc = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.identity(x)
        x = self.fc(x)
        return x


class SplitNetwork():
    def __init__(
        self, num_clients, device, network,
        criterion, base_dir,
    ):
        """
        args:
            num_clients
            device: cuda vs cpu
            network: ViT model
            criterion: loss function to be used
            base_dir: where to save pickles/model files
        """
        self.device = device
        self.num_clients = num_clients
        self.criterion = criterion
        self.network = network
        self.base_dir = base_dir

        # Per-client storage of labels/preds across evaluations
        self.cumulative_test_labels = [[] for _ in range(self.num_clients)]
        self.cumulative_test_preds = [[] for _ in range(self.num_clients)]

        # Temporary storage for most recent eval
        self.last_round_test_labels = [None for _ in range(self.num_clients)]
        self.last_round_test_preds = [None for _ in range(self.num_clients)]

        # History containers (per-client per-round)
        self.losses = {'train': [ [] for _ in range(self.num_clients) ], 'test': [ [] for _ in range(self.num_clients) ]}
        self.balanced_accs = {'train': [ [] for _ in range(self.num_clients) ], 'test': [ [] for _ in range(self.num_clients) ]}

        # Global history per FL round
        self.global_history = {
            'round': [],
            'loss_test_mean': [],
            'loss_train_mean': [],
            'acc_test_balanced': [],
            'acc_train_balanced': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

    def init_logs(self):
        """Call if you re-init between experiments."""
        self.losses = {'train': [ [] for _ in range(self.num_clients) ], 'test': [ [] for _ in range(self.num_clients) ]}
        self.balanced_accs = {'train': [ [] for _ in range(self.num_clients) ], 'test': [ [] for _ in range(self.num_clients) ]}
        self.cumulative_test_labels = [[] for _ in range(self.num_clients)]
        self.cumulative_test_preds = [[] for _ in range(self.num_clients)]
        self.last_round_test_labels = [None for _ in range(self.num_clients)]
        self.last_round_test_preds = [None for _ in range(self.num_clients)]
        self.global_history = {
            'round': [],
            'loss_test_mean': [],
            'loss_train_mean': [],
            'acc_test_balanced': [],
            'acc_train_balanced': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

    def set_optimizer(self, name, lr):
        """Set optimizer for global network parameters (used when training locally)."""
        if name == 'Adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)

    def distribute_images(self, dataset_name, train_data, test_data, batch_size):
        """Split dataset among clients."""
        if dataset_name == 'bloodmnist':
            _, self.testloader, train_dataset, _ = bloodmnist(batch_size=batch_size)
            _, self.CLIENTS_DATALOADERS, _ = blood_noniid(self.num_clients, train_dataset, batch_size=batch_size)

    def train_round(self, client_i):
        """
        Training loop for one client.
        Returns a weight_dict with flattened vectors for 'blocks', 'cls', 'pos_embed', and 'last_block'.
        """
        running_loss_client_i = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []

        # Make a deep copy of the network state so we can restore shared body after local training
        copy_network = copy.deepcopy(self.network)
        weight_dic = {'blocks': None, 'cls': None, 'pos_embed': None, 'last_block': None}

        self.network.train()

        for data in tqdm(self.CLIENTS_DATALOADERS[client_i], desc=f"Client{client_i} Train"):
            self.optimizer.zero_grad()

            imgs, labels = data[0].to(self.device), data[1].to(self.device)
            labels = labels.reshape(labels.shape[0])

            tail_output = self.network(imgs, client_i)
            outputs = tail_output[0] if isinstance(tail_output, tuple) else tail_output

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss_client_i += loss.item()

            _, predicted = torch.max(outputs, 1)
            whole_probs.append(torch.nn.Softmax(dim=-1)(outputs).detach().cpu())
            whole_labels.append(labels.detach().cpu())
            whole_preds.append(predicted.detach().cpu())

        # metrics & logging for train on this client
        self.metrics(client_i, whole_labels, whole_preds, running_loss_client_i,
                     len(self.CLIENTS_DATALOADERS[client_i]), train=True)

        # collect blocks (flattened full body) - kept for compatibility (may be large)
        try:
            weight_dic['blocks'] = weight_vec(self.network.vit.blocks).detach().cpu()
        except Exception:
            weight_dic['blocks'] = None

        # cls token
        try:
            weight_dic['cls'] = self.network.vit.cls_token.detach().cpu()
        except Exception:
            weight_dic['cls'] = None

        # pos embed
        try:
            weight_dic['pos_embed'] = self.network.vit.pos_embed.detach().cpu()
        except Exception:
            weight_dic['pos_embed'] = None

        # NEW: collect the last transformer block's flattened weights (smaller)
        try:
            last_block_module = self.network.vit.blocks[-1]
            weight_dic['last_block'] = weight_vec(last_block_module).detach().cpu()
        except Exception:
            weight_dic['last_block'] = None

        # restore the server body modules so local training changes don't persist to server copy
        try:
            self.network.vit.blocks = copy.deepcopy(copy_network.vit.blocks)
            self.network.vit.cls_token = copy.deepcopy(copy_network.vit.cls_token)
            self.network.vit.pos_embed = copy.deepcopy(copy_network.vit.pos_embed)
        except Exception:
            pass

        # DEBUG: always print presence/shape so runtime uses the correct function
        print(
            f"[DEBUG train_round] client {client_i} → "
            f"last_block={'present' if weight_dic['last_block'] is not None else 'MISSING'} | "
            f"blocks_shape={None if weight_dic['blocks'] is None else weight_dic['blocks'].shape} | "
            f"last_block_shape={None if weight_dic['last_block'] is None else weight_dic['last_block'].shape}"
        )

        return weight_dic

    def eval_round(self, client_i):
        """
        Evaluation loop for SplitNetwork.
        """
        running_loss_client_i = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []

        self.network.eval()
        with torch.no_grad():
            for data in tqdm(self.testloader, desc=f"Client{client_i} Eval"):
                imgs, labels = data[0].to(self.device), data[1].to(self.device)
                labels = labels.reshape(labels.shape[0])
                tail_output = self.network(imgs, client_i)
                outputs = tail_output[0] if isinstance(tail_output, tuple) else tail_output
                loss = self.criterion(outputs, labels)
                running_loss_client_i += loss.item()
                _, predicted = torch.max(outputs, 1)
                whole_probs.append(torch.nn.Softmax(dim=-1)(outputs).detach().cpu())
                whole_labels.append(labels.detach().cpu())
                whole_preds.append(predicted.detach().cpu())

            # store metrics for this client (test)
            self.metrics(client_i, whole_labels, whole_preds, running_loss_client_i, len(self.testloader), train=False)

            # store per-client test labels/preds for global computation (this round's evaluation for this client)
            try:
                lbls = torch.cat(whole_labels).cpu()
                preds = torch.cat(whole_preds).cpu()
            except Exception:
                lbls = torch.tensor([], dtype=torch.long)
                preds = torch.tensor([], dtype=torch.long)

            # Save last-round per-client (overwrite previous)
            self.last_round_test_labels[client_i] = lbls
            self.last_round_test_preds[client_i] = preds

            # Also append to cumulative (across rounds) if you want to analyze later
            if lbls.numel() > 0:
                self.cumulative_test_labels[client_i].extend(lbls.tolist())
                self.cumulative_test_preds[client_i].extend(preds.tolist())

    def metrics(self, client_i, whole_labels, whole_preds, running_loss_client_i, len_loader, train):
        """
        Save metrics in memory.
        """
        whole_labels = torch.cat(whole_labels) if len(whole_labels) > 0 else torch.tensor([], dtype=torch.long)
        whole_preds = torch.cat(whole_preds) if len(whole_preds) > 0 else torch.tensor([], dtype=torch.long)

        loss_epoch = running_loss_client_i / max(1, len_loader)
        balanced_acc = balanced_accuracy_score(whole_labels.detach().cpu(), whole_preds.detach().cpu()) if whole_labels.numel() > 0 else 0.0

        eval_name = 'train' if train else 'test'

        self.losses[eval_name][client_i].append(loss_epoch)
        self.balanced_accs[eval_name][client_i].append(balanced_acc)

        print(f"client{client_i}_{eval_name}:")
        print(f" Loss {eval_name}:{loss_epoch:.3f}")
        print(f"balanced accuracy {eval_name}:{balanced_acc:.3f}")

    def compute_global_metrics_for_last_round(self, round_idx=None, average_type='weighted'):
        """
        Aggregate last_round_test_labels and last_round_test_preds across all clients
        and compute precision, recall, f1 (weighted), and global balanced accuracy and mean loss.
        """
        all_lbls = []
        all_preds = []
        for i in range(self.num_clients):
            if self.last_round_test_labels[i] is not None and self.last_round_test_labels[i].numel() > 0:
                all_lbls.append(self.last_round_test_labels[i])
                all_preds.append(self.last_round_test_preds[i])

        if len(all_lbls) == 0:
            print("No evaluation predictions available this round to compute global metrics.")
            return None

        all_lbls = torch.cat(all_lbls).numpy()
        all_preds = torch.cat(all_preds).numpy()

        precision, recall, f1, _ = precision_recall_fscore_support(all_lbls, all_preds, average=average_type, zero_division=0)
        bal_acc = balanced_accuracy_score(all_lbls, all_preds)

        test_losses = []
        train_losses = []
        for i in range(self.num_clients):
            if len(self.losses['test'][i]) > 0:
                test_losses.append(self.losses['test'][i][-1])
            if len(self.losses['train'][i]) > 0:
                train_losses.append(self.losses['train'][i][-1])
        mean_test_loss = float(np.mean(test_losses)) if len(test_losses) > 0 else float('nan')
        mean_train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else float('nan')

        round_no = (round_idx if round_idx is not None else (len(self.global_history['round']) + 1))
        self.global_history['round'].append(round_no)
        self.global_history['loss_test_mean'].append(mean_test_loss)
        self.global_history['loss_train_mean'].append(mean_train_loss)
        self.global_history['acc_test_balanced'].append(bal_acc)
        self.global_history['precision'].append(precision)
        self.global_history['recall'].append(recall)
        self.global_history['f1'].append(f1)

        print(f"Global metrics (round {round_no}): BalancedAcc={bal_acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, MeanTestLoss={mean_test_loss:.4f}")
        return {
            'round': round_no,
            'balanced_acc': bal_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_test_loss': mean_test_loss,
            'mean_train_loss': mean_train_loss
        }

    def save_pickles(self, base_dir):
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, 'loss_epoch.pkl'), 'wb') as handle:
            pkl.dump(self.losses, handle)
        with open(os.path.join(base_dir, 'balanced_accs.pkl'), 'wb') as handle:
            pkl.dump(self.balanced_accs, handle)
        with open(os.path.join(base_dir, 'global_history.pkl'), 'wb') as handle:
            pkl.dump(self.global_history, handle)
        with open(os.path.join(base_dir, 'cumulative_test_labels.pkl'), 'wb') as handle:
            pkl.dump(self.cumulative_test_labels, handle)
        with open(os.path.join(base_dir, 'cumulative_test_preds.pkl'), 'wb') as handle:
            pkl.dump(self.cumulative_test_preds, handle)
        print(f"Saved pickles to {base_dir}")

    def save_model_weights_and_full(self, save_dir, prefix="final"):
        os.makedirs(save_dir, exist_ok=True)
        state_path = os.path.join(save_dir, f"{prefix}_state_dict.pth")
        torch.save(self.network.state_dict(), state_path)
        full_path = os.path.join(save_dir, f"{prefix}_full_model.pth")
        torch.save(self.network, full_path)
        print(f"Saved state_dict to: {state_path}")
        print(f"Saved full model to: {full_path}")

    def plot_per_client_curves(self):
        plots_dir = os.path.join(self.base_dir, "plots_per_client")
        os.makedirs(plots_dir, exist_ok=True)
        for client in range(self.num_clients):
            plt.figure(figsize=(6, 4))
            plt.plot(self.losses['train'][client], label='Train Loss')
            plt.plot(self.losses['test'][client], label='Test Loss')
            plt.xlabel("FL Round")
            plt.ylabel("Loss")
            plt.title(f"Client {client} Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f"loss_client{client}.png"))
            plt.close()

            plt.figure(figsize=(6, 4))
            plt.plot(self.balanced_accs['train'][client], label='Train Balanced Acc')
            plt.plot(self.balanced_accs['test'][client], label='Test Balanced Acc')
            plt.xlabel("FL Round")
            plt.ylabel("Balanced Accuracy")
            plt.title(f"Client {client} Balanced Accuracy")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f"acc_client{client}.png"))
            plt.close()

        print(f"Saved per-client plots to {plots_dir}")

    def plot_global_curves(self):
        plots_dir = os.path.join(self.base_dir, "plots_global")
        os.makedirs(plots_dir, exist_ok=True)
        rounds = self.global_history['round']

        if len(rounds) == 0:
            print("No global rounds recorded yet - skipping global plots.")
            return

        plt.figure(figsize=(7, 4))
        plt.plot(rounds, self.global_history['loss_train_mean'], label='Mean Train Loss')
        plt.plot(rounds, self.global_history['loss_test_mean'], label='Mean Test Loss')
        plt.xlabel("FL Round")
        plt.ylabel("Loss")
        plt.title("Global Mean Loss per FL Round")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "global_mean_loss.png"))
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(rounds, self.global_history['acc_test_balanced'], label='Global Test Balanced Acc')
        plt.xlabel("FL Round")
        plt.ylabel("Balanced Accuracy")
        plt.title("Global Balanced Accuracy per FL Round")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "global_balanced_acc.png"))
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(rounds, self.global_history['precision'], label='Precision (weighted)')
        plt.plot(rounds, self.global_history['recall'], label='Recall (weighted)')
        plt.plot(rounds, self.global_history['f1'], label='F1 (weighted)')
        plt.xlabel("FL Round")
        plt.ylabel("Score")
        plt.title("Global Precision/Recall/F1 per FL Round")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "global_prf.png"))
        plt.close()

        print(f"Saved global plots to {plots_dir}")

    def plot_global_confusion_matrix(self, class_names=None, normalize=False):
        cms_dir = os.path.join(self.base_dir, "confusion_global")
        os.makedirs(cms_dir, exist_ok=True)

        all_lbls = []
        all_preds = []
        for i in range(self.num_clients):
            if self.last_round_test_labels[i] is not None and self.last_round_test_labels[i].numel() > 0:
                all_lbls.append(self.last_round_test_labels[i])
                all_preds.append(self.last_round_test_preds[i])

        if len(all_lbls) == 0:
            print("No test predictions collected in last round; cannot compute global confusion matrix.")
            return

        all_lbls = torch.cat(all_lbls).numpy()
        all_preds = torch.cat(all_preds).numpy()

        cm = confusion_matrix(all_lbls, all_preds)
        if normalize:
            with np.errstate(all='ignore'):
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm = np.nan_to_num(cm)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title("Global Confusion Matrix (last round aggregated)")
        plt.tight_layout()
        fname = os.path.join(cms_dir, "global_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved global confusion matrix to {fname}")


class FeSVBiS(nn.Module):
    def __init__(
        self, ViT_name, num_classes,
        num_clients=6, in_channels=3, ViT_pretrained=False,
        initial_block=1, final_block=6, resnet_dropout=None, DP=False, mean=None, std=None
    ) -> None:
        super().__init__()

        self.initial_block = initial_block
        self.final_block = final_block

        self.vit = timm.create_model(
            model_name=ViT_name,
            pretrained=ViT_pretrained,
            num_classes=num_classes,
            in_chans=in_channels
        )

        self.resnet50 = self.vit.patch_embed
        self.resnet50_clients = nn.ModuleList([copy.deepcopy(self.resnet50) for i in range(num_clients)])
        self.common_network = ResidualBlock(drop_out=resnet_dropout)
        client_tail = MLP_cls_classes(num_classes=num_classes)
        self.mlp_clients_tail = nn.ModuleList([copy.deepcopy(client_tail) for i in range(num_clients)])
        self.DP = DP
        self.mean = mean
        self.std = std

    def forward(self, x, chosen_block, client_idx):
        x = self.resnet50_clients[client_idx](x)
        if self.DP:
            noise = torch.randn(size=x.shape).to(x.device) * self.std + self.mean
            x = x + noise
        for block_num in range(chosen_block):
            x = self.vit.blocks[block_num](x)
        x = self.common_network(x)
        x = self.mlp_clients_tail[client_idx](x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=768, out_channels=768, stride=1, downsample=None, drop_out=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.pool = nn.AvgPool2d(14, stride=1)
        self.dropout = nn.Dropout2d(p=drop_out) if drop_out is not None else None
        self.drop_out = drop_out

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.permute(x, (0, -1, 1))
            x = x.reshape(x.shape[0], x.shape[1], 14, 14)
        residual = x
        out = self.conv1(x)
        if self.drop_out is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.pool(out)
        return out.reshape(-1, 768)


class SplitFeSViBS(SplitNetwork):
    def __init__(
        self, num_clients, device,
        network, criterion, base_dir,
        initial_block, final_block,
    ):
        # initialize SplitNetwork first
        super().__init__(num_clients=num_clients, device=device, network=network, criterion=criterion, base_dir=base_dir)

        self.initial_block = initial_block
        self.final_block = final_block
        self.train_chosen_blocks = [0] * num_clients

    def set_optimizer_mel(self, name, lr):
        if name == 'Adam':
            self.optimizer_mel = [torch.optim.Adam(self.mel_body[i].parameters(), lr=lr) for i in range(self.num_clients)]

    def train_round(self, client_i):
        """
        Training loop for FeSViBS variant (per-client chosen block).
        Returns a dict with flattened vectors for 'blocks', 'cls', 'pos_embed', 'resnet' and 'last_block'.
        """
        running_loss_client_i = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
    
        # choose a block for this client for this round
        self.chosen_block = np.random.randint(low=self.initial_block, high=self.final_block + 1)
        self.train_chosen_blocks[client_i] = self.chosen_block
    
        copy_network = copy.deepcopy(self.network)
        weight_dic = {'blocks': None, 'cls': None, 'pos_embed': None, 'resnet': None, 'last_block': None}
    
        print(f"Chosen Block:{self.chosen_block} for client {client_i}")
        self.network.train()
    
        for data in tqdm(self.CLIENTS_DATALOADERS[client_i], desc=f"Client{client_i} Train"):
            self.optimizer.zero_grad()
            imgs, labels = data[0].to(self.device), data[1].to(self.device)
            labels = labels.reshape(labels.shape[0])
    
            tail_output = self.network(x=imgs, chosen_block=self.chosen_block, client_idx=client_i)
            outputs = tail_output if not isinstance(tail_output, tuple) else tail_output[0]
    
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
    
            running_loss_client_i += loss.item()
            _, predicted = torch.max(outputs, 1)
            whole_probs.append(torch.nn.Softmax(dim=-1)(outputs).detach().cpu())
            whole_labels.append(labels.detach().cpu())
            whole_preds.append(predicted.detach().cpu())
    
        # metrics + logging for train on this client
        self.metrics(client_i, whole_labels, whole_preds, running_loss_client_i, len(self.CLIENTS_DATALOADERS[client_i]), train=True)
    
        # collect full body blocks vector (kept for compatibility; may be large)
        try:
            weight_dic['blocks'] = weight_vec(self.network.vit.blocks).detach().cpu()
        except Exception:
            weight_dic['blocks'] = None
    
        # cls token and pos_embed
        try:
            weight_dic['cls'] = self.network.vit.cls_token.detach().cpu()
            weight_dic['pos_embed'] = self.network.vit.pos_embed.detach().cpu()
        except Exception:
            weight_dic['cls'] = None
            weight_dic['pos_embed'] = None
    
        # collect the specific transformer block used by this client (last_block for HE optimization)
        try:
            chosen_idx = max(0, self.chosen_block - 1)  # block index used during training
            last_block_module = self.network.vit.blocks[chosen_idx]
            weight_dic['last_block'] = weight_vec(last_block_module).detach().cpu()
        except Exception:
            weight_dic['last_block'] = None
    
        # restore the shared body modules so local training changes don't persist on server copy
        try:
            self.network.vit.blocks = copy.deepcopy(copy_network.vit.blocks)
            self.network.vit.cls_token = copy.deepcopy(copy_network.vit.cls_token)
            self.network.vit.pos_embed = copy.deepcopy(copy_network.vit.pos_embed)
        except Exception:
            pass
    
        # DEBUG print so logs show whether last_block was collected
        print(
            f"[DEBUG SplitFeSViBS.train_round] client {client_i} → "
            f"chosen_block={self.chosen_block} | "
            f"last_block={'present' if weight_dic.get('last_block') is not None else 'MISSING'} | "
            f"last_block_shape={None if weight_dic.get('last_block') is None else weight_dic['last_block'].shape}"
        )

        return weight_dic


    def eval_round(self, client_i):
        """
        Evaluation loop for FeSViBS variant.
        """
        running_loss_client_i = 0
        whole_labels = []
        whole_preds = []
        whole_probs = []
        num_b = self.train_chosen_blocks[client_i]
        print(f"Chosen block for testing: {num_b}")
        self.network.eval()
        with torch.no_grad():
            for data in tqdm(self.testloader, desc=f"Client{client_i} Eval"):
                imgs, labels = data[0].to(self.device), data[1].to(self.device)
                labels = labels.reshape(labels.shape[0])
                tail_output = self.network(x=imgs, chosen_block=num_b, client_idx=client_i)
                outputs = tail_output if not isinstance(tail_output, tuple) else tail_output[0]
                loss = self.criterion(outputs, labels)
                running_loss_client_i += loss.item()
                _, predicted = torch.max(outputs, 1)
                whole_probs.append(torch.nn.Softmax(dim=-1)(outputs).detach().cpu())
                whole_labels.append(labels.detach().cpu())
                whole_preds.append(predicted.detach().cpu())
            # metrics + store last-round arrays for global metrics
            self.metrics(client_i, whole_labels, whole_preds, running_loss_client_i, len(self.testloader), train=False)

            try:
                lbls = torch.cat(whole_labels).cpu()
                preds = torch.cat(whole_preds).cpu()
            except Exception:
                lbls = torch.tensor([], dtype=torch.long)
                preds = torch.tensor([], dtype=torch.long)

            self.last_round_test_labels[client_i] = lbls
            self.last_round_test_preds[client_i] = preds

            if lbls.numel() > 0:
                self.cumulative_test_labels[client_i].extend(lbls.tolist())
                self.cumulative_test_preds[client_i].extend(preds.tolist())
