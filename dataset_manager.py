from imports import TrOCRProcessor, load_dataset, dataclass, Any, Dict, List, torch

@dataclass
class DataCollatorForSeq2Seq:
    """Collates preprocessed batches into PyTorch tensors."""

    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([torch.tensor(f["pixel_values"]) for f in features])
        labels = torch.tensor([f["labels"] for f in features])
        return {"pixel_values": pixel_values, "labels": labels}

class HebrewOCRDatasetManager:
    """Manages downloading, slicing, and preprocessing of the dataset."""

    def __init__(self, dataset_name: str, processor: TrOCRProcessor,
                 max_target_length: int = 128, ds_size: int = 100, ignore_idx: int = -100):
        self.dataset_name = dataset_name
        self.processor = processor
        self.max_target_length = max_target_length
        self.ds_size = ds_size
        self.ignore_idx = ignore_idx

    def load_and_prepare_data(self):
        print(f"Loading dataset '{self.dataset_name}' from Hugging Face...")
        raw_dataset = load_dataset(self.dataset_name)

        # Select subsets for testing / quick training runs
        train_samples = int(self.ds_size * 0.9)
        eval_samples = self.ds_size - train_samples
        train_data = raw_dataset["train"].select(range(train_samples))
        eval_data = raw_dataset["validation"].select(range(eval_samples))

        print(f"Preprocessing {train_samples} training rows and {eval_samples} evaluation rows...")
        train_dataset = train_data.map(
            self._preprocess_function, batched=True, remove_columns=train_data.column_names
        )
        eval_dataset = eval_data.map(
            self._preprocess_function, batched=True, remove_columns=eval_data.column_names
        )

        return train_dataset, eval_dataset

    def _preprocess_function(self, examples: Dict[str, list]) -> Dict[str, list]:
        images = [image.convert("RGB") for image in examples["image"]]
        text = examples["text"]

        pixel_values = self.processor(images, return_tensors="pt").pixel_values

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids

        # Mask padding tokens with IGNORE_INDEX
        labels = [
            [(l if l != self.processor.tokenizer.pad_token_id else self.ignore_idx) for l in label]
            for label in labels
        ]

        return {"pixel_values": pixel_values, "labels": labels}
