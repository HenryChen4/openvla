import json_numpy
import numpy as np

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image

from experiments.robot.openvla_utils import get_vla, get_processor
from experiments.robot.robot_utils import get_action, get_model, normalize_gripper_action, invert_gripper_action

@dataclass
class LiberoSpatialServerCfg:
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


class OpenVLAServer:
    def __init__(self, cfg, attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.cfg, self.attn_implementation = cfg, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA model
        self.processor = get_processor(self.cfg)
        self.model = get_model(self.cfg)

    def predict_action(self, payload: Dict[str, Any]):
        # payload contains no preprio obs
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])
            
            # Parse payload components
            instruction = payload["instruction"]
            observation = {
                "full_image": payload["image"]
            }

            actions = get_action(
                self.cfg,
                self.model,
                observation,
                instruction,
                processor=self.processor,
                n_samples = self.cfg.n_samples
            )

            if type(actions) is tuple:
                actions, generated_outputs = actions
            else:
                generated_outputs = {} # empty dict
                
            if self.cfg.output_hidden_states:
                # If output_hidden_states is True, "hidden_states" exists in generated_outputs
                # generated_outputs['hidden_states'] is a tuple of length 7 (number of generated tokens)
                # Within which each element is a tuple of length 33 (number of layers)
                # Each element in the inner tuple is a tensor of shape (bs=1, 1 or N, 4096)
                # The final hidden states before decoding is generated_outputs['hidden_states'][0][-1][0, -1, :]
                all_hidden_states = generated_outputs['hidden_states']
                hidden_states_last_layer = [s[-1][0, -1, :] for s in all_hidden_states]
                hidden_states_last_layer = torch.stack(hidden_states_last_layer, dim=0) # (7, 4096)

            # last hidden state along token dimension used as embedding for SAFE
            # best for LSTM, MLP & PCA
            print(hidden_states_last_layer.shape)
            embedding = hidden_states_last_layer[-1]

            actions = np.array(actions, dtype=np.float32, copy=True)
            actions = normalize_gripper_action(actions, binarize=True)
            actions = invert_gripper_action(actions)

            if hasattr(embedding, "detach"):  # torch tensor
                embedding = embedding.detach().to(dtype=torch.float16).cpu().numpy()
            else:
                embedding = np.asarray(embedding)

            payload = {
                "actions": actions,
                "embedding": embedding,
            }

            if double_encode:
                return JSONResponse(json_numpy.dumps(payload))
            else:
                return JSONResponse(payload)
        except:
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
            )
            return "error"
    
    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)

@draccus.wrap()
def deploy(cfg: LiberoSpatialServerCfg) -> None:
    server = OpenVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)

if __name__ == "__main__":
    deploy()