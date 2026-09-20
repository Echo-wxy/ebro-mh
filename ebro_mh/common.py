from __future__ import annotations
from pathlib import Path
import hashlib, json, os, random, zipfile
import numpy as np
import pandas as pd
import yaml


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p


def sha256_file(path, chunk=1<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,default=str)


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed)


def unpack_zip_if_needed(zip_path, out_dir):
    out=ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    return out


def numeric(s):
    return pd.to_numeric(s, errors='coerce')
