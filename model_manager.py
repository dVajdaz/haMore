import config
from imports import TrOCRProcessor, VisionEncoderDecoderModel

class HebrewOCRModelManager:
    """Manages model loading and configuration"""

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        self.processor = None

    def initialize_processor(self):
        print(f"Loading processor for '{self.model_name}'...")
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        return self.processor

    def initialize_model(self):
        print(f"Initializing fresh '{self.model_name}' instance...")
        model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

        # Configure decoder start tokens and beam search parameters
        model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        model.generation_config.num_beams = getattr(config, "NUM_BEAMS", 4)
        model.generation_config.max_length = getattr(config, "MAX_TARGET_LENGTH", 128)
        # model.generation_config.length_penalty = 1.0  # TODO: choose value
        model.generation_config.early_stopping = True
        model.generation_config.no_repeat_ngram_size = 3  # TODO: check if should deprecate

        return model

    """
    def initialize_model(self, num_beams: int = 4, max_target_length: int = 128):
        print(f"Loading processor and model weights for '{self.model_name}'...")
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

        # Configure decoder start tokens and beam search parameters
        self.model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        self.model.generation_config.num_beams = num_beams
        self.model.generation_config.max_length = max_target_length
        self.model.generation_config.length_penalty = 1.0  # TODO: choose value
        self.model.generation_config.early_stopping = True
        self.model.generation_config.no_repeat_ngram_size = 3  # TODO: check if should deprecate

        return self.processor, self.model
        """

