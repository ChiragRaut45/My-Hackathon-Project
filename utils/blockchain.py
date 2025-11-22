# utils/blockchain.py

import json
import hashlib
import time
from pathlib import Path

CHAIN_PATH = Path("utils/blockchain_chain.json")
CHAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

def compute_hash(block):
    s = json.dumps(block, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

def load_chain():
    if not CHAIN_PATH.exists():
        genesis = {
            "index": 0,
            "timestamp": time.time(),
            "data": "GENESIS",
            "previous_hash": "0"
        }
        genesis["hash"] = compute_hash(genesis)
        CHAIN_PATH.write_text(json.dumps([genesis], indent=2))
        return [genesis]

    return json.loads(CHAIN_PATH.read_text())

def append_block(patient_id, prediction, probabilities, raw_input):
    chain = load_chain()
    prev = chain[-1]

    new_block = {
        "index": prev["index"] + 1,
        "timestamp": time.time(),
        "patient_id": str(patient_id),
        "prediction": prediction,
        "probabilities": probabilities,
        "raw_input_hash": compute_hash(raw_input),
        "previous_hash": prev["hash"]
    }

    new_block["hash"] = compute_hash(new_block)

    chain.append(new_block)
    CHAIN_PATH.write_text(json.dumps(chain, indent=2))

    return new_block
