"""Vérifie l'environnement (GPU, VRAM, versions) et fixe les graines.

Usage dans un notebook Colab :
    from src.check_env import check_env
    check_env()
"""
import platform
import random
import sys

import numpy as np

VRAM_LIMIT_GB = 12.0  # contrainte du sujet


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def check_env(seed: int = 42) -> None:
    print("Python :", sys.version.split()[0], "|", platform.platform())
    print("numpy  :", np.__version__)
    set_seed(seed)
    print("Graine :", seed)

    try:
        import torch
    except ImportError:
        print("PyTorch : NON INSTALLE")
        return

    print("torch  :", torch.__version__)
    if not torch.cuda.is_available():
        print("GPU    : AUCUN (Colab : Execution > Modifier le type d'execution > GPU)")
        return

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9
    print(f"GPU    : {props.name} | VRAM totale : {total_gb:.1f} Go")
    fraction = min(1.0, VRAM_LIMIT_GB / total_gb)
    torch.cuda.set_per_process_memory_fraction(fraction)
    print(f"Limite VRAM appliquee : {VRAM_LIMIT_GB:.0f} Go (fraction {fraction:.2f})")


def report_vram() -> float:
    """A appeler a la fin de chaque entrainement ; renvoie la VRAM max en Go."""
    import torch
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"VRAM max utilisee : {peak:.2f} Go (limite {VRAM_LIMIT_GB:.0f} Go)")
    return peak


if __name__ == "__main__":
    check_env()
