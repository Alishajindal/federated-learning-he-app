"""
TenSEAL helpers (compatible with older/newer tenseal Python API versions).
Provides:
- create_context(...)
- encrypt_tensor(...)
- add_encrypted_lists(...)
- scalar_mul_encrypted_list(...)
- decrypt_chunks_to_tensor(...)
- ensure_same_chunking(...)
"""

import tenseal as ts
import numpy as np
import torch

def _print_tenseal_version():
    try:
        v = ts.__version__
    except Exception:
        v = "unknown"
    print(f"[he_utils] TenSEAL version: {v}")
_print_tenseal_version()

def create_context(poly_mod_degree=16384, coeff_mod_bit_sizes=[60, 40, 40, 60], scale=2**40, use_galois=False):
    """
    Create a TenSEAL CKKS context. Server keeps secret key (private context).
    Returns context.
    """
    # create context
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, coeff_mod_bit_sizes)
    # set global scale if supported
    try:
        ctx.global_scale = scale
    except Exception:
        # some versions set scale per-vector; ignore if not supported
        pass

    # generate keys (relin & galois if requested)
    try:
        ctx.generate_relin_keys()
    except Exception:
        # older versions may generate keys lazily or use different names
        try:
            ts.relin_keys(ctx)
        except Exception:
            pass
    if use_galois:
        try:
            ctx.generate_galois_keys()
        except Exception:
            try:
                ts.galois_keys(ctx)
            except Exception:
                pass

    # make context private (keep secret key here)
    try:
        ctx.make_context_private()
    except Exception:
        # older versions may not have this; secret key is already present in ctx
        pass

    return ctx

def _split_array_into_chunks(flat_arr, max_slot_count):
    n = flat_arr.shape[0]
    if n <= max_slot_count:
        return [flat_arr]
    chunk_size = max_slot_count
    return [flat_arr[i:i+chunk_size] for i in range(0, n, chunk_size)]

def encrypt_tensor(ctx, tensor):
    """
    Robust encrypt: split a 1-D torch tensor into safe-sized chunks and encrypt each chunk.
    Returns (enc_chunks, original_shape)
    """
    arr = tensor.detach().cpu().numpy().astype(np.float64).ravel()

    # get slot count from context
    try:
        max_slots = int(ctx.slot_count())
    except Exception:
        try:
            pmd = ctx.poly_modulus_degree()
            max_slots = int(pmd // 2)
        except Exception:
            max_slots = 8192

    # Use nearly-full slot count (-2) for efficiency, but at least 1024
    effective_max = max(1024, max_slots - 2)

    # split into chunks of size effective_max
    chunks = [arr[i:i+effective_max] for i in range(0, arr.size, effective_max)]

    enc_chunks = []
    for c in chunks:
        c = np.asarray(c).ravel()
        if c.size > max_slots:
            raise ValueError(
                f"Chunk size ({c.size}) exceeds TenSEAL slot count ({max_slots}). Increase poly_mod_degree."
            )

        plain_list = [float(x) for x in c.tolist()]

        try:
            enc = ts.ckks_vector(ctx, plain_list)
        except Exception:
            try:
                enc = ts.ckks_vector(ctx, np.array(plain_list, dtype=np.float64))
            except Exception as e:
                raise RuntimeError(f"TenSEAL ckks_vector failed for a chunk: {e}")

        enc_chunks.append(enc)

    return enc_chunks, arr.shape


def add_encrypted_lists(enc_list_a, enc_list_b):
    """Element-wise addition of two lists of ciphertexts."""
    if len(enc_list_a) != len(enc_list_b):
        raise ValueError("Mismatched chunk lengths during HE addition.")
    return [a + b for a, b in zip(enc_list_a, enc_list_b)]

def scalar_mul_encrypted_list(enc_list, scalar):
    """Multiply each ciphertext by scalar (plaintext multiplication)."""
    return [c * float(scalar) for c in enc_list]

def decrypt_chunks_to_tensor(ctx, enc_chunks, original_shape, dtype=torch.float32):
    """
    Decrypt list of ciphertexts and return a torch tensor of shape=original_shape.
    """
    flat = []
    for c in enc_chunks:
        try:
            part = c.decrypt()
        except Exception:
            # some versions return numpy arrays, some lists
            try:
                part = c.decrypt().tolist()
            except Exception:
                part = list(c.decrypt())
        flat.extend(part)
    arr = np.array(flat, dtype=np.float64)[: np.prod(original_shape)]
    return torch.tensor(arr.reshape(original_shape), dtype=dtype)

def ensure_same_chunking(existing_chunks, new_chunks):
    """
    Ensure all clients use same chunking length (slot_count). If existing_chunks is None, return new_chunks.
    """
    if existing_chunks is None:
        return new_chunks
    if len(existing_chunks) != len(new_chunks):
        raise ValueError("Mismatched chunk sizes between clients. Ensure same TenSEAL context/slot count.")
    return existing_chunks
