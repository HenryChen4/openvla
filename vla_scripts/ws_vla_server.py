import enum
import logging
import socket 
import vla_scripts.ws_server_wrapper as ws_server_wrapper

from pathlib import Path
from typing import Any, Dict, Optional, Union

from dataclasses import dataclass

@dataclass
class LiberoOpenVLACfg:
    # MODEL SETTINGS
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-spatial"
    attn_implementation: str = "flash_attention_2"
    unnorm_key: str = "libero_spatial"

    # RUN SETTINGS
    host: str = "0.0.0.0"
    port: int = 8000

    # LIBERO DATA SETTINGS
    center_crop: bool = True 
    n_samples: int = 1 # number of actions to sample (keep at 1 following paper)
    output_attentions: bool = False
    output_logits: bool = False
    output_hidden_states: bool = True 

def main(server_cfg: LiberoOpenVLACfg) -> None:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Creating server (host: {hostname}, ip: {local_ip})")

    vla = ws_server_wrapper.OpenVLAWrapper(server_cfg)
    server = ws_server_wrapper.WebSocketOpenVLAServer(
        vla,
        host=server_cfg.host,
        port=server_cfg.port,
    )
    server.serve_forever()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    
    server_cfg = LiberoOpenVLACfg()
    main(server_cfg)