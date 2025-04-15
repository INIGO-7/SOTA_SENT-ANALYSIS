import torch
from datasets import load_dataset
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification, pipeline
)
from torch.nn.functional import sigmoid
from sklearn.metrics import classification_report
from peft import PeftModel
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod


class EmotionClassifierBase(ABC):
    def __init__(self):
        self.dataset = None
        self.model = None
        self.tokenizer = None
        self.valid_split_names = []
        self.dataset_labels = []

    @abstractmethod
    def init_model_and_tokenizer(self):
        """Load model/tokenizer — implemented in child"""
        pass

    @abstractmethod
    def eval_performance(self, verbose: bool = False):
        """Runs evaluation and stores predictions"""
        pass

    @abstractmethod
    def run_inference(self, text: str):
        """Runs inference on a given input"""
        pass

    def init_dataset(self, dataset_name: str, split: str, config_name: str = None):
        if split not in self.valid_split_names:
            raise ValueError(f"Invalid split '{split}'. Choose from: {self.valid_split_names}")
        
        self.dataset = load_dataset(dataset_name, name=config_name)[split]

    def create_performance_report(self, print_report: bool = False, write_report: bool = True, filename: str = "classification_report.txt"):
        report = classification_report(self.true_labels, self.pred_labels, target_names=self.dataset_labels)
        if print_report:
            print(report)
        if write_report:
            with open(filename, "w") as f:
                f.write(report)
        return report