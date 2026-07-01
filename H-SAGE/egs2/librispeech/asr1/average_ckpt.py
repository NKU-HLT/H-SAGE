import torch
import os
from collections import OrderedDict
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
import warnings
import logging

logging.basicConfig(level=logging.INFO)

def average_checkpoints(
    ckpt_paths: List[Path],
) -> Dict[str, Any]:

    if not ckpt_paths:
        warnings.warn("The input list of checkpoint paths is empty. Returning an empty dict.")
        return {}

    logging.info(f"Averaging {len(ckpt_paths)} checkpoints...")

    avg = None
    num_models = len(ckpt_paths)

    for path in ckpt_paths:
        logging.info(f"Loading checkpoint {path}")
        
        try:
            states = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            
        except Exception as e:
            logging.error(f"Failed to load checkpoint {path}: {e}")
            continue

        if avg is None:
            avg = states
        else:
            for k in avg:
                if k in states:
                    try:
                        avg[k] = avg[k] + states[k]
                    except RuntimeError as e:
                        logging.warning(
                            f"Skipping parameter '{k}' due to runtime error during accumulation: {e}"
                        )
                else:
                    logging.warning(
                        f"Parameter '{k}' missing in checkpoint {path.name}. Skipping accumulation for this parameter in this checkpoint."
                    )
    
    if avg is None:
        warnings.warn("No valid checkpoints were loaded. Returning an empty dict.")
        return {}

    for k in avg:
        if str(avg[k].dtype).startswith("torch.int"):
            logging.info(f"Accumulating {k} instead of averaging (dtype: {avg[k].dtype})")
            pass
        else:
            avg[k] = avg[k] / num_models

    logging.info("Averaging completed.")
    return avg


def average_ckpt(start, end, root_dir, save_name):
    res = []
    for i in range(start, end+1):
        res.append(f"{root_dir}/{i}epoch.pth")
    save_dir = f"{root_dir}/{save_name}"
    final_ckpt = average_checkpoints(res)
    torch.save(final_ckpt, save_dir)

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--dir",
    required=True,
)
args = parser.parse_args()

start = 31
end = 35
ckpt_dir = args.dir
average_ckpt(
    start,
    end,
    ckpt_dir,
    f"avg_{start}-{end}.pth"
)
