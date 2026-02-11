import asyncio
import http
import logging
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np
import torch
import vla_scripts.msgpack_numpy as msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
)

logger = logging.getLogger(__name__)


class OpenVLAWrapper:
    def __init__(
        self, cfg, attn_implementation: Optional[str] = "flash_attention_2"
    ):
        self.cfg, self.attn_implementation = cfg, attn_implementation

        self.processor = get_processor(self.cfg)
        self.model = get_model(self.cfg)

    def predict_action(self, payload: Dict[str, Any]):
        instruction = payload["prompt"]
        observation = {"full_image": payload["observation/image"]}

        actions = get_action(
            self.cfg,
            self.model,
            observation,
            instruction,
            processor=self.processor,
            n_samples=self.cfg.n_samples,
        )

        if type(actions) is tuple:
            actions, generated_outputs = actions
        else:
            generated_outputs = {}

        if self.cfg.output_hidden_states:
            # If output_hidden_states is True, "hidden_states" exists in generated_outputs
            # generated_outputs['hidden_states'] is a tuple of length 7 (number of generated tokens)
            # Within which each element is a tuple of length 33 (number of layers)
            # Each element in the inner tuple is a tensor of shape (bs=1, 1 or N, 4096)
            # The final hidden states before decoding is generated_outputs['hidden_states'][0][-1][0, -1, :]
            all_hidden_states = generated_outputs["hidden_states"]
            hidden_states_last_layer = [
                s[-1][0, -1, :] for s in all_hidden_states
            ]
            hidden_states_last_layer = torch.stack(
                hidden_states_last_layer, dim=0
            )  # (7, 4096)

        # last hidden state along token dimension used as embedding for SAFE
        # best for LSTM, MLP & PCA
        embedding = hidden_states_last_layer[-1]
        actions = np.array(actions, dtype=np.float32, copy=True)
        actions = normalize_gripper_action(actions, binarize=True)
        actions = invert_gripper_action(actions)

        if hasattr(embedding, "detach"):  # torch tensor
            embedding = embedding.detach().to(dtype=torch.float16).cpu().numpy()
        else:
            embedding = np.asarray(embedding)

        payload = {"actions": actions, "pre_logits": embedding}

        return payload


class WebSocketOpenVLAServer:
    def __init__(
        self, vla_wrapper, host: str = "0.0.0.0", port: int = 8000
    ) -> None:
        self._vla = vla_wrapper

        self._host = host
        self._port = port

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        metadata = {"hello": "world"}

        await websocket.send(packer.pack(metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()

                # obs must be:
                # {
                #   instruction: str
                #   image: np.ndarray
                # }
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action_payload = self._vla.predict_action(obs)

                # patch to ensure data can be transmitted
                action_payload["pre_logits"] = action_payload[
                    "pre_logits"
                ].astype("float32")

                infer_time = time.monotonic() - infer_time

                action_payload["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action_payload["server_timing"]["prev_total_ms"] = (
                        prev_total_time * 1000
                    )

                await websocket.send(packer.pack(action_payload))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(
                    f"Connection from {websocket.remote_address} closed"
                )
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
