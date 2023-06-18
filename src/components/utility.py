import os
import random

import numpy as np
import torch


def format_dict(
    info: dict[str, str | int | float],
    *,
    prefix: str = "",
    indent: str = "  ",
    end: str = "\n",
) -> str:
    n_key_char: int = max([len(s) for s in info.keys()])
    strings: str = ""
    for key, val in info.items():
        strings += f"{prefix}{key:<{n_key_char}} : "
        if isinstance(val, dict):
            strings += end
            strings += format_dict(val, prefix=indent + prefix, indent=indent)
        else:
            strings += f"{val}"
            strings += end
    return strings


def print_info(info: dict[str, str | int | float], *, linewidth: int = 60) -> None:
    print("=" * linewidth)
    print(format_dict(info), end="")
    print("=" * linewidth)


def fix_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
