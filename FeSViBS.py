import os
import numpy as np
import models
import random
from dataset import bloodmnist
from utils import weight_dec_global, weight_vec
import argparse
import torch
from torch import nn
import time
import gc
import csv
import psutil

# HE utilities
import he_utils

# ---------------- CSV helpers for logging HE timings & chunk counts ----------------
def init_csv_logger(save_dir):
    csv_path = os.path.join(save_dir, "he_encryption_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "client_id", "chosen_block",
                "last_block_chunks", "cls_chunks", "pos_chunks",
                "enc_time_sec", "he_time_sec",
                "cpu_mem_mb", "gpu_mem_mb"
            ])
    return csv_path

def append_csv(csv_path, row):
    # row should be a list matching the header order above
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def read_all_csv(csv_path):
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def write_all_csv(csv_path, rows):
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "round", "client_id", "chosen_block",
            "last_block_chunks", "cls_chunks", "pos_chunks",
            "enc_time_sec", "he_time_sec",
            "cpu_mem_mb", "gpu_mem_mb"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "round": row.get("round", ""),
                "client_id": row.get("client_id", ""),
                "chosen_block": row.get("chosen_block", ""),
                "last_block_chunks": row.get("last_block_chunks", ""),
                "cls_chunks": row.get("cls_chunks", ""),
                "pos_chunks": row.get("pos_chunks", ""),
                "enc_time_sec": row.get("enc_time_sec", ""),
                "he_time_sec": row.get("he_time_sec", ""),
                "cpu_mem_mb": row.get("cpu_mem_mb", ""),
                "gpu_mem_mb": row.get("gpu_mem_mb", "")
            })


# ---------------- The main fesvibs() function (full) ----------------
def fesvibs(
        dataset_name, lr, batch_size, Epochs, input_size, num_workers,
        save_every_epochs, model_name, pretrained, opt_name, seed,
        base_dir, num_clients, DP, epsilon, delta, resnet_dropout,
        initial_block, final_block, fesvibs_arg, local_round
    ):

    # reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    method_flag = 'FeSViBS' if fesvibs_arg else 'SViBS'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if DP:
        std = np.sqrt(2 * np.math.log(1.25/delta)) / epsilon
        mean = 0
        dir_name = f"{model_name}_{lr}lr_{dataset_name}_{num_clients}Clients_{initial_block}to{final_block}Blocks_{batch_size}Batch__{epsilon,delta}DP_{method_flag}"
    else:
        mean = 0
        std = 0
        dir_name = f"{model_name}_{lr}lr_{dataset_name}_{num_clients}Clients_{initial_block}to{final_block}Blocks_{batch_size}Batch_{method_flag}"

    save_dir = base_dir if base_dir is not None else dir_name
    os.makedirs(save_dir, exist_ok=True)

    print(f"Logging to: {save_dir}")
    csv_path = init_csv_logger(save_dir)

    print("Getting the Dataset and Dataloader!")
    if dataset_name == "bloodmnist":
        num_classes = 8
        _, _, traindataset, testdataset = bloodmnist(
            input_size=input_size,
            batch_size=batch_size,
            download=True,
            num_workers=num_workers
        )
        num_channels = 3
    else:
        raise ValueError("Only 'bloodmnist' supported in this script.")

    criterion = nn.CrossEntropyLoss()

    print("Creating network and Split manager...")
    fesvibs_network = models.FeSVBiS(
        ViT_name=model_name,
        num_classes=num_classes,
        num_clients=num_clients,
        in_channels=num_channels,
        ViT_pretrained=pretrained,
        initial_block=initial_block,
        final_block=final_block,
        resnet_dropout=resnet_dropout,
        DP=DP, mean=mean, std=std
    ).to(device)

    Split = models.SplitFeSViBS(
        num_clients=num_clients,
        device=device,
        network=fesvibs_network,
        criterion=criterion,
        base_dir=save_dir,
        initial_block=initial_block,
        final_block=final_block
    )

    print("Distribute Images Among Clients")
    Split.distribute_images(
        dataset_name=dataset_name,
        train_data=traindataset,
        test_data=testdataset,
        batch_size=batch_size
    )

    Split.set_optimizer(opt_name, lr)  # uses Split.set_optimizer
    Split.init_logs()

    # ---------------- CREATE HE CONTEXT ----------------
    print("\n[HE] Creating TenSEAL context...")
    he_ctx = he_utils.create_context(
        poly_mod_degree=32768,
        coeff_mod_bit_sizes=[60, 60, 60, 60],
        scale=2**40,
        use_galois=False
    )
    print("[HE] TenSEAL context created.\n")

    print("Start Training!\n")

    # ---------------------- TRAINING ROUNDS ----------------------
    for r in range(Epochs):
        print(f"\n================= ROUND {r+1}/{Epochs} =================\n")

        # Encrypted running sums (lists of ciphertexts)
        enc_sum_last = None
        enc_sum_cls = None
        enc_sum_pos = None

        # shapes for dechunking
        last_block_shape = None
        cls_shape = None
        pos_shape = None

        # # -------------------- LOCAL TRAINING & ENCRYPTION --------------------we commented it because there we missing values in he_time in csv file below is the updated version
        # for client_i in range(num_clients):
        #     weight_dict = Split.train_round(client_i)
        #     chosen_block = getattr(Split, "train_chosen_blocks", [None]*num_clients)[client_i]

        #     print(f"[HE] Encrypting updates from client {client_i}...")
        #     enc_start = time.time()

        #     # ===== Encrypt LAST BLOCK (stream add) =====
        #     last_chunks = 0
        #     if weight_dict.get("last_block") is not None:
        #         enc_last, last_block_shape = he_utils.encrypt_tensor(he_ctx, weight_dict["last_block"])
        #         last_chunks = len(enc_last)
        #         if enc_sum_last is None:
        #             enc_sum_last = enc_last
        #         else:
        #             if len(enc_sum_last) != len(enc_last):
        #                 raise ValueError("Mismatch chunk count in last_block.")
        #             for k in range(len(enc_last)):
        #                 enc_sum_last[k] = enc_sum_last[k] + enc_last[k]
        #         # free per-client temporary objects
        #         del enc_last
        #         gc.collect()

        #     # ===== Encrypt CLS TOKEN =====
        #     cls_chunks = 0
        #     if weight_dict.get("cls") is not None:
        #         enc_cls, cls_shape = he_utils.encrypt_tensor(he_ctx, weight_dict["cls"])
        #         cls_chunks = len(enc_cls)
        #         if enc_sum_cls is None:
        #             enc_sum_cls = enc_cls
        #         else:
        #             if len(enc_sum_cls) != len(enc_cls):
        #                 raise ValueError("Mismatch chunk count in cls.")
        #             for k in range(len(enc_cls)):
        #                 enc_sum_cls[k] = enc_sum_cls[k] + enc_cls[k]
        #         del enc_cls
        #         gc.collect()

        #     # ===== Encrypt POS EMBEDDINGS =====
        #     pos_chunks = 0
        #     if weight_dict.get("pos_embed") is not None:
        #         enc_pos, pos_shape = he_utils.encrypt_tensor(he_ctx, weight_dict["pos_embed"])
        #         pos_chunks = len(enc_pos)
        #         if enc_sum_pos is None:
        #             enc_sum_pos = enc_pos
        #         else:
        #             if len(enc_sum_pos) != len(enc_pos):
        #                 raise ValueError("Mismatch chunk count in pos_embed.")
        #             for k in range(len(enc_pos)):
        #                 enc_sum_pos[k] = enc_sum_pos[k] + enc_pos[k]
        #         del enc_pos
        #         gc.collect()

        #     enc_end = time.time()
        #     enc_time = round(enc_end - enc_start, 4)

        #     # ----------------- capture mem usage right after encryption -----------------
        #     proc = psutil.Process(os.getpid())
        #     cpu_mem_mb = round(proc.memory_info().rss / (1024**2), 3)
        #     if torch.cuda.is_available():
        #         try:
        #             gpu_mem_mb = round(torch.cuda.memory_reserved(device) / (1024**2), 3)
        #         except Exception:
        #             # fallback to memory_allocated
        #             gpu_mem_mb = round(torch.cuda.memory_allocated(device) / (1024**2), 3)
        #     else:
        #         gpu_mem_mb = 0.0

        #     print(f"    client {client_i} last_block → {last_chunks} chunks")
        #     print(f"    client {client_i} cls → {cls_chunks} chunks")
        #     print(f"    client {client_i} pos_embed → {pos_chunks} chunks")
        #     print(f"    [HE] Encryption of client {client_i} done in {enc_time:.2f} sec | cpu_mem={cpu_mem_mb} MB gpu_reserved={gpu_mem_mb} MB\n")

        #     # append a CSV row with he_time_sec empty (will fill once HE averaging done)
        #     append_csv(csv_path, [
        #         r+1, client_i, chosen_block,
        #         last_chunks, cls_chunks, pos_chunks,
        #         enc_time, "",    # he_time to fill after averaging
        #         cpu_mem_mb, gpu_mem_mb
        #     ])

        # -------------------- LOCAL TRAINING & ENCRYPTION --------------------
        for client_i in range(num_clients):
            weight_dict = Split.train_round(client_i)
            chosen_block = getattr(Split, "train_chosen_blocks", [None]*num_clients)[client_i]
        
            print(f"[HE] Encrypting updates from client {client_i}...")
        
            # prepare memory sampling
            proc = psutil.Process(os.getpid())
            # reset GPU peak counter if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=None)
        
            enc_start = time.time()
        
            # ===== Encrypt LAST BLOCK (stream add) =====
            last_chunks = 0
            if weight_dict.get("last_block") is not None:
                enc_last, last_block_shape = he_utils.encrypt_tensor(he_ctx, weight_dict["last_block"])
                last_chunks = len(enc_last)
                if enc_sum_last is None:
                    enc_sum_last = enc_last
                else:
                    if len(enc_sum_last) != len(enc_last):
                        raise ValueError("Mismatch chunk count in last_block.")
                    for k in range(len(enc_last)):
                        enc_sum_last[k] = enc_sum_last[k] + enc_last[k]
                del enc_last
                gc.collect()
        
            # ===== Encrypt CLS TOKEN =====
            cls_chunks = 0
            if weight_dict.get("cls") is not None:
                enc_cls, cls_shape = he_utils.encrypt_tensor(he_ctx, weight_dict["cls"])
                cls_chunks = len(enc_cls)
                if enc_sum_cls is None:
                    enc_sum_cls = enc_cls
                else:
                    if len(enc_sum_cls) != len(enc_cls):
                        raise ValueError("Mismatch chunk count in cls.")
                    for k in range(len(enc_cls)):
                        enc_sum_cls[k] = enc_sum_cls[k] + enc_cls[k]
                del enc_cls
                gc.collect()
        
            # ===== Encrypt POS EMBEDDINGS =====
            pos_chunks = 0
            if weight_dict.get("pos_embed") is not None:
                enc_pos, pos_shape = he_utils.encrypt_tensor(he_ctx, weight_dict["pos_embed"])
                pos_chunks = len(enc_pos)
                if enc_sum_pos is None:
                    enc_sum_pos = enc_pos
                else:
                    if len(enc_sum_pos) != len(enc_pos):
                        raise ValueError("Mismatch chunk count in pos_embed.")
                    for k in range(len(enc_pos)):
                        enc_sum_pos[k] = enc_sum_pos[k] + enc_pos[k]
                del enc_pos
                gc.collect()
        
            enc_end = time.time()
            enc_time = round(enc_end - enc_start, 4)
        
            # measure CPU RSS now (MB)
            cpu_mem_mb = round(proc.memory_info().rss / (1024 * 1024), 3)
        
            # measure GPU reserved peak if CUDA
            gpu_mem_mb = 0.0
            if torch.cuda.is_available():
                # max_memory_reserved returns bytes
                gpu_mem_mb = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 3)
        
            print(f"    client {client_i} last_block → {last_chunks} chunks")
            print(f"    client {client_i} cls → {cls_chunks} chunks")
            print(f"    client {client_i} pos_embed → {pos_chunks} chunks")
            print(f"    [HE] Encryption of client {client_i} done in {enc_time:.2f} sec | CPU_RSS={cpu_mem_mb} MB | GPU_peak_reserved={gpu_mem_mb} MB\n")
        
            # append a CSV row with he_time_sec blank for now (we fill after aggregation)
            append_csv(csv_path, [
                r+1, client_i, chosen_block,
                last_chunks, cls_chunks, pos_chunks,
                enc_time, "",    # he_time (fill after averaging)
                cpu_mem_mb, gpu_mem_mb
            ])

        # ------------------ HOMOMORPHIC AVERAGING -----------------------
        print("[HE] Starting homomorphic averaging...")
        he_start = time.time()

        inv_clients = 1.0 / float(num_clients)

        # Average last block
        dec_last_block = None
        if enc_sum_last is not None:
            enc_avg_last = he_utils.scalar_mul_encrypted_list(enc_sum_last, inv_clients)
            dec_last_block = he_utils.decrypt_chunks_to_tensor(he_ctx, enc_avg_last, last_block_shape, dtype=torch.float32).to(device)
            del enc_avg_last
            gc.collect()
            print("[HE] Decoded averaged last-block.")

        # Average cls token
        dec_cls = None
        if enc_sum_cls is not None:
            enc_avg_cls = he_utils.scalar_mul_encrypted_list(enc_sum_cls, inv_clients)
            dec_cls = he_utils.decrypt_chunks_to_tensor(he_ctx, enc_avg_cls, cls_shape, dtype=torch.float32).to(device)
            del enc_avg_cls
            gc.collect()
            print("[HE] Decoded averaged cls.")

        # Average pos embeddings
        dec_pos = None
        if enc_sum_pos is not None:
            enc_avg_pos = he_utils.scalar_mul_encrypted_list(enc_sum_pos, inv_clients)
            dec_pos = he_utils.decrypt_chunks_to_tensor(he_ctx, enc_avg_pos, pos_shape, dtype=torch.float32).to(device)
            del enc_avg_pos
            gc.collect()
            print("[HE] Decoded averaged pos_embed.")

        he_end = time.time()
        he_time = round(he_end - he_start, 4)
        print(f"[HE] Homomorphic averaging + decryption finished in {he_time:.2f} seconds\n")

        # ------------------ UPDATE CSV FOR HE TIME ----------------------
        # Update the last <num_clients> rows in CSV (the rows we just appended).
        rows = read_all_csv(csv_path)
        if len(rows) > 0:
            # fill he_time_sec for the last num_clients rows
            for idx in range(1, min(num_clients, len(rows)) + 1):
                rows[-idx]["he_time_sec"] = he_time
            write_all_csv(csv_path, rows)

        # ------------------ UPDATE GLOBAL MODEL -------------------------
        print("[HE] Applying aggregated weights to global model...")

        if dec_last_block is not None:
            try:
                Split.network.vit.blocks[-1] = weight_dec_global(Split.network.vit.blocks[-1], dec_last_block)
                print("[HE] Updated last transformer block.")
            except Exception as e:
                print(f"[HE] ERROR updating last block: {e}")

        if dec_cls is not None:
            try:
                Split.network.vit.cls_token.data = dec_cls
                print("[HE] Updated cls_token.")
            except Exception as e:
                print(f"[HE] ERROR updating cls_token: {e}")

        if dec_pos is not None:
            try:
                Split.network.vit.pos_embed.data = dec_pos
                print("[HE] Updated pos_embed.")
            except Exception as e:
                print(f"[HE] ERROR updating pos_embed: {e}")

        print("[HE] Global model updated.\n")

        # ------------------ FeSViBS HEAD/TAIL FEDERATION (plaintext) ----------
        if fesvibs_arg and ((r+1) % local_round == 0 and r != 0):
            print('[FeSViBS] Performing head/tail federation...')
            tails_weights = []
            head_weights = []
            for head, tail in zip(Split.network.resnet50_clients, Split.network.mlp_clients_tail):
                head_weights.append(weight_vec(head).detach().cpu())
                tails_weights.append(weight_vec(tail).detach().cpu())

            mean_avg_tail = torch.mean(torch.stack(tails_weights), axis=0)
            mean_avg_head = torch.mean(torch.stack(head_weights), axis=0)

            for i in range(num_clients):
                Split.network.mlp_clients_tail[i] = weight_dec_global(Split.network.mlp_clients_tail[i], mean_avg_tail.to(device))
                Split.network.resnet50_clients[i] = weight_dec_global(Split.network.resnet50_clients[i], mean_avg_head.to(device))

        # ------------------ EVALUATION ----------------------
        print("[Eval] Starting evaluation for all clients...")
        for client_i in range(num_clients):
            Split.eval_round(client_i)

        global_metrics = Split.compute_global_metrics_for_last_round(round_idx=r+1)
        if global_metrics is not None:
            print(global_metrics)

        # ------------------ SAVE ROUND ARTIFACTS ----------------------
        if (r+1) % save_every_epochs == 0 or r == Epochs - 1:
            Split.save_pickles(save_dir)
            Split.save_model_weights_and_full(save_dir)
            Split.plot_per_client_curves()
            Split.plot_global_curves()
            Split.plot_global_confusion_matrix(class_names=[str(i) for i in range(num_classes)])

        print("\n================ END OF ROUND =================\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Centralized Experiments')
    parser.add_argument('--dataset_name', type=str, choices=['bloodmnist'], help='Dataset Name')
    parser.add_argument('--input_size',  type=int, default=224)
    parser.add_argument('--local_round',  type=int, default=2)
    parser.add_argument('--num_workers',  type=int, default=0)
    parser.add_argument('--initial_block',  type=int, default=1)
    parser.add_argument('--final_block',  type=int, default=6)
    parser.add_argument('--num_clients',  type=int, default=6)
    parser.add_argument('--model_name', type=str, default='vit_base_r50_s16_224')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--fesvibs_arg', type=bool, default=False)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--Epochs',  type=int, default=200)
    parser.add_argument('--opt_name', type=str, choices=['Adam'], default='Adam')
    parser.add_argument('--lr',  type=float, default=1e-4)
    parser.add_argument('--save_every_epochs',  type=int, default=10)
    parser.add_argument('--seed',  type=int, default=105)
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--DP', type=bool, default=False)
    parser.add_argument('--epsilon',  type=float, default=0)
    parser.add_argument('--delta',  type=float, default=0.00001)
    parser.add_argument('--resnet_dropout',  type=float, default=0.5)

    args = parser.parse_args()

    fesvibs(
        dataset_name=args.dataset_name, input_size=args.input_size,
        num_workers=args.num_workers, model_name=args.model_name,
        pretrained=args.pretrained, batch_size=args.batch_size,
        Epochs=args.Epochs, opt_name=args.opt_name, lr=args.lr,
        save_every_epochs=args.save_every_epochs, seed=args.seed,
        base_dir=args.base_dir, num_clients=args.num_clients,
        DP=args.DP, epsilon=args.epsilon, delta=args.delta,
        initial_block=args.initial_block, final_block=args.final_block,
        resnet_dropout=args.resnet_dropout, fesvibs_arg=args.fesvibs_arg, local_round=args.local_round
    )
