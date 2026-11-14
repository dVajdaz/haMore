import os
from dataclasses import dataclass
from typing import Any, Dict, List

#from pillow import Image

import torch
from datasets import load_dataset, Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
