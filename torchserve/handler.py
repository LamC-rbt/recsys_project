import json
import torch
from ts.torch_handler.base_handler import BaseHandler
from pathlib import Path
from typing import List

from recsys_project.utils.general_utils import load_config, build_model, get_device
from recsys_project.utils.prediction_utils import generate_topk_predictions
from recsys_project.utils.dataset_utils import load_dataset_stats
from recsys_project.configs.config import SASRecConfig
from recsys_project.models.sasrec import SASRec

class SASRecServeHandler(BaseHandler):
    """SASrec Recommender Handler"""
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        """Initialize model weights and retrieve inference parameters"""
        self.device = get_device()

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        manifest = ctx.manifest
        serialized_file = manifest["model"]["serializedFile"]
        weights_path = Path(model_dir) / serialized_file

        config_path = Path(model_dir) / "config_sasrec.py"
        config: SASRecConfig = load_config(str(config_path))

        num_items = load_dataset_stats(Path(model_dir))["num_items"]

        self.model = SASRec(
            num_items, sequence_length=config.sequence_length, embedding_dim=config.embedding_dim,
            num_heads=config.num_heads, num_blocks=config.num_blocks, dropout_rate=config.dropout_rate
        )

        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.max_seq_len = config.sequence_length - 1
        self.top_k = 10
        self.initialized = True

    def preprocess(self, data):
        """Preprocess data in python format to a List[int] format to run a model"""
        if not data or "body" not in data[0]:
            return None

        body = data[0]["body"]

        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
            body = json.loads(body)

        return body["item_sequence"]

    def inference(self, input_ids):
        """Running a model on preprocessed List of previous interactions"""
        if input_ids is None:
            return None
        result = generate_topk_predictions(
            model=self.model,
            sequences=[input_ids],
            max_seq_len=self.max_seq_len,
            top_k=self.top_k,
            device=self.device,
            logger=None
        )
        return result[0]

    def postprocess(self, inference_output):
        """Convert predictions to Dict format"""
        if inference_output is None:
            return []
        return [{'recommendations': inference_output}]
