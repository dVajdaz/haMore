from imports import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataset_manager import DataCollatorForSeq2Seq

class HebrewOCRTrainerManager:
    """Manages training configuration parameters and Seq2SeqTrainer initialization."""

    def __init__(
        self,
        model,
        processor,
        output_dir: str,
        learning_rate: float,
        batch_size: int,
        epochs: int
    ):
        self.model = model
        self.processor = processor
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def build_trainer(self, train_dataset, eval_dataset) -> Seq2SeqTrainer:
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            predict_with_generate=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=5,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            fp16=True,
            dataloader_num_workers=0,
            save_total_limit=1,
            load_best_model_at_end=True,
        )

        data_collator = DataCollatorForSeq2Seq(processor=self.processor)

        trainer = Seq2SeqTrainer(
            model=self.model,
            processing_class=self.processor,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        return trainer
