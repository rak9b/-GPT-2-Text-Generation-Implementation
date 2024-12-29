import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import logging

class GPT2Trainer:
    def __init__(self, model_name='gpt2', output_dir='./gpt2-output'):
        """Initialize the trainer with the model, tokenizer, and settings."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.output_dir = output_dir
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def prepare_dataset(self, text_file_path, block_size=128):
        """Prepare dataset from a text file."""
        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=text_file_path,
            block_size=block_size
        )
        return dataset

    def train(self, train_dataset, eval_dataset=None, num_epochs=3, batch_size=4):
        """Train the model on the provided dataset."""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=10000,
            save_total_limit=2,
            logging_dir=f'{self.output_dir}/logs',
            evaluation_strategy='steps' if eval_dataset else 'no'
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        self.logger.info("Starting training...")
        trainer.train()
        self.logger.info("Training completed!")

        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
