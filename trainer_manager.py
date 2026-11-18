from imports import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataset_manager import DataCollatorForSeq2Seq

class HebrewOCRHyperparameterTuner:
    """Manages automated hyperparameter search using Optuna via Seq2SeqTrainer."""

    def __init__(
        self,
        model_init_fn,
        processor,
        output_dir: str,
        train_dataset,
        eval_dataset
    ):
        self.model_init_fn = model_init_fn  # Function that returns a fresh model instance per trial
        self.processor = processor
        self.output_dir = output_dir
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def _optuna_hp_space(self, trial):
        """Defines the search space for Optuna trials."""
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 16, 32]),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "num_train_epochs": 1,
        }

    def run_search(self, n_trials: int = 5):
        print(f"Starting hyperparameter search with Optuna ({n_trials} trials)...")

        # Define default training arguments that remain static across trials
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            predict_with_generate=True,
            eval_strategy="epoch",
            save_strategy="no",  # Don't save checkpoints
            logging_steps=5,
            fp16=True,
            dataloader_num_workers=0,
            load_best_model_at_end=False,
            report_to="none",    # Disable external loggers
        )

        data_collator = DataCollatorForSeq2Seq(processor=self.processor)

        # Initialize Trainer using model_init so weights reset for each trial
        trainer = Seq2SeqTrainer(
            model_init=self.model_init_fn,
            processing_class=self.processor,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )

        # Run Optuna hyperparameter search optimization
        best_trial = trainer.hyperparameter_search(
            direction="minimize",  # We want to minimize validation loss
            backend="optuna",
            n_trials=n_trials,
            hp_space=self._optuna_hp_space,
        )

        print("\n--- Hyperparameter Search Complete ---")
        print(f"Best Hyperparameters found: {best_trial.hyperparameters}")
        #print(f"Best Validation Loss: {best_trial.run_summary}")

        return best_trial.hyperparameters

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
