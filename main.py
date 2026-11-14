from config import *

from model_manager import HebrewOCRModelManager
from dataset_manager import HebrewOCRDatasetManager
from trainer_manager import HebrewOCRTrainerManager

if __name__ == "__main__":
    # Initialize Model & Processor
    model_manager = HebrewOCRModelManager(MODEL_NAME, OUTPUT_DIR)
    processor, model = model_manager.initialize_model(NUM_BEAMS, MAX_TARGET_LENGTH)

    # Initialize & Preprocess Dataset
    dataset_manager = HebrewOCRDatasetManager(DATASET_NAME, processor,
                                              MAX_TARGET_LENGTH, DATASET_SIZE,
                                              IGNORE_INDEX)
    train_ds, eval_ds = dataset_manager.load_and_prepare_data()

    # Initialize & Run Trainer
    trainer_manager = HebrewOCRTrainerManager(
        model=model,
        processor=processor,
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    trainer = trainer_manager.build_trainer(train_ds, eval_ds)

    # Execute Fine-Tuning
    print("Starting fine-tuning process...")
    trainer.train()
    
    print(f"Training complete! Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
