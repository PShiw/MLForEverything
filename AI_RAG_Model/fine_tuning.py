"""
Fine-Tuning Module
Provides capabilities to fine-tune language models using LoRA/QLoRA
"""

import os
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFineTuner:
    """
    Fine-tune language models using LoRA (Low-Rank Adaptation)
    """
    
    def __init__(
        self,
        base_model_name: str = "google/flan-t5-base",
        output_dir: str = "./fine_tuned_models",
        device: str = "auto"
    ):
        """
        Initialize fine-tuner
        
        Args:
            base_model_name: HuggingFace model name
            output_dir: Directory to save fine-tuned models
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        logger.info(f"Fine-tuner initialized with device: {self.device}")
    
    def load_base_model(self):
        """Load the base model and tokenizer"""
        try:
            logger.info(f"Loading base model: {self.base_model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info("Base model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            raise
    
    def prepare_lora_model(
        self,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ):
        """
        Prepare model for LoRA fine-tuning
        
        Args:
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability
            target_modules: List of module names to apply LoRA to
        """
        try:
            if self.model is None:
                self.load_base_model()
            
            logger.info("Preparing LoRA model...")
            
            # Default target modules (common for most models)
            if target_modules is None:
                target_modules = ["q_proj", "v_proj"]
            
            # LoRA configuration
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            # Prepare model for training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Apply LoRA
            self.peft_model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.peft_model.parameters())
            
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                       f"({100 * trainable_params / total_params:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error preparing LoRA model: {str(e)}")
            raise
    
    def prepare_training_data(
        self,
        documents: List[Dict[str, str]],
        max_length: int = 512
    ) -> Dataset:
        """
        Prepare training data from documents
        
        Args:
            documents: List of dictionaries with 'text' or 'input'/'output' keys
            max_length: Maximum sequence length
            
        Returns:
            HuggingFace Dataset
        """
        try:
            logger.info(f"Preparing {len(documents)} documents for training...")
            
            # Tokenize the documents
            tokenized_data = []
            
            for doc in documents:
                # Handle different data formats
                if 'text' in doc:
                    text = doc['text']
                elif 'input' in doc and 'output' in doc:
                    text = f"Input: {doc['input']}\nOutput: {doc['output']}"
                else:
                    logger.warning(f"Skipping document with unknown format: {doc.keys()}")
                    continue
                
                # Tokenize
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors=None
                )
                
                tokenized_data.append(tokenized)
            
            # Create dataset
            dataset = Dataset.from_list(tokenized_data)
            
            logger.info(f"Dataset prepared with {len(dataset)} examples")
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def fine_tune(
        self,
        train_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        save_steps: int = 100,
        logging_steps: int = 10,
        gradient_accumulation_steps: int = 4
    ) -> str:
        """
        Fine-tune the model
        
        Args:
            train_dataset: Training dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
            gradient_accumulation_steps: Gradient accumulation steps
            
        Returns:
            Path to saved model
        """
        try:
            if self.peft_model is None:
                raise ValueError("LoRA model not prepared. Call prepare_lora_model() first.")
            
            logger.info("Starting fine-tuning...")
            
            # Output directory for this run
            run_name = f"{self.base_model_name.split('/')[-1]}_finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_path = os.path.join(self.output_dir, run_name)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_path,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                save_steps=save_steps,
                save_total_limit=3,
                fp16=self.device == "cuda",
                report_to="none",
                remove_unused_columns=False,
                push_to_hub=False,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Trainer
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            
            # Train
            logger.info("Training started...")
            trainer.train()
            
            # Save the final model
            self.peft_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            # Save training info
            training_info = {
                "base_model": self.base_model_name,
                "training_date": datetime.now().isoformat(),
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_examples": len(train_dataset)
            }
            
            with open(os.path.join(output_path, "training_info.json"), 'w') as f:
                json.dump(training_info, f, indent=2)
            
            logger.info(f"Fine-tuning complete! Model saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            raise
    
    def load_fine_tuned_model(self, model_path: str):
        """
        Load a fine-tuned model
        
        Args:
            model_path: Path to the fine-tuned model
        """
        try:
            logger.info(f"Loading fine-tuned model from: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info("Fine-tuned model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {str(e)}")
            raise


class DatasetCreator:
    """
    Helper class to create training datasets from documents
    """
    
    @staticmethod
    def create_qa_dataset(
        questions: List[str],
        answers: List[str],
        contexts: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Create Q&A dataset
        
        Args:
            questions: List of questions
            answers: List of answers
            contexts: Optional list of contexts
            
        Returns:
            List of training examples
        """
        dataset = []
        
        for i, (q, a) in enumerate(zip(questions, answers)):
            if contexts and i < len(contexts):
                text = f"Context: {contexts[i]}\nQuestion: {q}\nAnswer: {a}"
            else:
                text = f"Question: {q}\nAnswer: {a}"
            
            dataset.append({"text": text})
        
        logger.info(f"Created Q&A dataset with {len(dataset)} examples")
        return dataset
    
    @staticmethod
    def create_instruction_dataset(
        instructions: List[str],
        responses: List[str]
    ) -> List[Dict[str, str]]:
        """
        Create instruction-following dataset
        
        Args:
            instructions: List of instructions
            responses: List of responses
            
        Returns:
            List of training examples
        """
        dataset = []
        
        for instr, resp in zip(instructions, responses):
            dataset.append({
                "input": instr,
                "output": resp
            })
        
        logger.info(f"Created instruction dataset with {len(dataset)} examples")
        return dataset
    
    @staticmethod
    def load_from_json(filepath: str) -> List[Dict[str, str]]:
        """
        Load dataset from JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of training examples
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} examples from {filepath}")
        return data


if __name__ == "__main__":
    print("Fine-tuning module ready!")
    print("Example usage:")
    print("  fine_tuner = ModelFineTuner(base_model_name='gpt2')")
    print("  fine_tuner.prepare_lora_model()")
    print("  dataset = fine_tuner.prepare_training_data(documents)")
    print("  model_path = fine_tuner.fine_tune(dataset)")
