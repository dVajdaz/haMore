from config import *

from model_manager import HebrewOCRModelManager
from dataset_manager import HebrewOCRDatasetManager
from trainer_manager import HebrewOCRTrainerManager, HebrewOCRHyperparameterTuner

if __name__ == "__main__":
    # Initialize Processor
    model_manager = HebrewOCRModelManager(MODEL_NAME, OUTPUT_DIR)
    processor = model_manager.initialize_processor()

    # Initialize & Preprocess Dataset
    dataset_manager = HebrewOCRDatasetManager(DATASET_NAME, processor,
                                              MAX_TARGET_LENGTH, DATASET_SIZE,
                                              IGNORE_INDEX)
    train_ds, eval_ds = dataset_manager.load_and_prepare_data()

    # Optuna Hyperparameter Tuning Procedure
    tuner = HebrewOCRHyperparameterTuner(
        model_init_fn=model_manager.initialize_model,
        processor=processor,
        output_dir=OUTPUT_DIR,
        train_dataset=train_ds,
        eval_dataset=eval_ds
    )

    best_hp = tuner.run_search(n_trials=6)
    optimal_lr = best_hp.get("learning_rate", LEARNING_RATE)
    optimal_batch_size = best_hp.get("per_device_train_batch_size", BATCH_SIZE)

    # Initialize & Run Model & Trainer
    print(f"\nProceeding to final training with tuned parameters: "
          f"LR={optimal_lr}, "
          f"Batch Size={optimal_batch_size}")

    model = model_manager.initialize_model()
    trainer_manager = HebrewOCRTrainerManager(
        model=model,
        processor=processor,
        output_dir=OUTPUT_DIR,
        learning_rate=optimal_lr,
        batch_size=optimal_batch_size,
        epochs=EPOCHS
    )
    trainer = trainer_manager.build_trainer(train_ds, eval_ds)

    # Execute Fine-Tuning
    print("Starting fine-tuning process...")
    trainer.train()
    
    print(f"Training complete! Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
