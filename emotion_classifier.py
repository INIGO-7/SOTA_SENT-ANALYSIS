from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, List, Optional, Union
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from llama_cpp import Llama
import pandas as pd
import numpy as np
import torch
import re
import os

class DatasetHandler:
    """Handles dataset loading and preparation for different emotion datasets.
    
    Attributes:
        dataset_name (str): Name of the dataset
        config_name (Optional[str]): Dataset configuration name
        valid_splits (List[str]): Available dataset splits
        labels (List[str]): Emotion labels for the dataset
        label_type (str): 'single' for single-label, 'multi' for multi-label
    """
    
    DATASET_CONFIGS = {
        "sem_eval_2018": {
            "name": "sem_eval_2018",
            "hf_name": "sem_eval_2018_task_1.py",
            "config": "subtask5.english",
            "splits": ["train", "test"],
            "labels": [
                "anger", "anticipation", "disgust", "fear", "joy", "love",
                "optimism", "pessimism", "sadness", "surprise", "trust"
            ],
            "text_column": "Tweet",
            "label_column": "labels",   # After preprocessing, a new one-hot encoded column will be created
            "type": "multi"
        },
        "go_emotions": {
            "name": "go_emotions",
            "hf_name": "google-research-datasets/go_emotions",
            "config": "simplified",
            "splits": ["train", "validation", "test"],
            "labels": [
                "admiration", "amusement", "anger", "annoyance", "approval", "caring",
                "confusion", "curiosity", "desire", "disappointment", "disapproval",
                "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
                "joy", "love", "nervousness", "optimism", "pride", "realization",
                "relief", "remorse", "sadness", "surprise", "neutral"
            ],
            "text_column": "text",
            "label_column": "labels",
            "type": "multi"
        },
        "dair_emotion": {
            "name": "dair_emotion",
            "hf_name": "dair-ai/emotion",
            "config": None,
            "splits": ["train", "validation", "test"],
            "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
            "text_column": "text",
            "label_column": "label",
            "type": "single"
        }
    }

    def __init__(self, dataset_name: str, split: str = "test"):
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Invalid dataset. Choose from: {list(self.DATASET_CONFIGS.keys())}")
            
        self.config = self.DATASET_CONFIGS[dataset_name]
        self._validate_split(split)
        self._load_dataset(split)
        self._prepare_labels()

    def _validate_split(self, split: str):
        if split not in self.config["splits"]:
            raise ValueError(f"Invalid split '{split}'. Available: {self.config['splits']}")
    
    def _one_hot_encode_targets(self):
        num_items, num_labels = len(self.data), len(self.config['labels'])
        y_targets_all = np.zeros((num_items, num_labels), dtype=int)
        for i, labels_indices in enumerate(self.data[self.config['label_column']]):
            for label_index in labels_indices:
                y_targets_all[i, label_index] = 1

        self.one_hot_targets = y_targets_all

    def _load_dataset(self, split: str):
        self.data = load_dataset(
            self.config["hf_name"],
            name=self.config["config"]
        )[split]

    def _prepare_labels(self):
        self.labels = self.data.features["labels"].feature.names
        self.label_type = self.config['type']
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self._one_hot_encode_targets()

class PromptTemplate:
    """Handles model-specific prompt formatting"""
    
    TEMPLATES = {
        "llama": (
            "<|system|>\n{system_prompt}<|end|>\n"
            "<|user|>\n{user_prompt}<|end|>\n"
            "<|assistant|>\n"
        ),
        "phi-3": (
            "<|system|>\n{system_prompt}<|end|>\n"
            "<|user|>\n{user_prompt}<|end|>\n"
            "<|assistant|>\n"
        ),
        "deepseek": (
            "<｜begin▁of▁sentence｜>{system_prompt}\n"
            "<｜User｜>{user_prompt}<｜end▁of▁sentence｜>\n"
            "<｜Assistant｜>\n"
        ),
        "chatml": (
            "<|im_start|>system\n{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    }

    def __init__(self, template_name: Optional[str], system_prompt: str, user_pre_prompt: str = "", custom_template : bool = False):
        self.custom_template = custom_template

        if custom_template:
            self.template = self.TEMPLATES.get(template_name)
            self.end_token = (match.group() if (match := re.search(r'<\|.*?end.*?\|>', self.template)) else "\n")

        self.system_prompt = system_prompt
        self.user_pre_prompt = user_pre_prompt
        
    def format(self, user_prompt: str) -> str:
        if not self.custom_template:
            print("Custom template not provided! thus .format() function cannot be used")
            return ""

        return self.template.format(
            system_prompt = self.system_prompt,
            user_prompt = self.user_pre_prompt + user_prompt
        ) if self.template else user_prompt


class ModelStrategy(ABC):
    """Abstract base class for model loading strategies"""
    
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def get_multiple_sentiment(self, ds: List):
        pass

    @abstractmethod
    def get_sentiment(self, text: str):
        pass

    # @abstractmethod
    # def inference(self, text : str):
        # pass


class TransformersStrategy(ModelStrategy):
    """Handles Transformers-based models from Hugging Face Hub"""
    
    def __init__(
        self,
        model_name: str,
        task: str = "text-classification",
        use_pipeline: bool = True,
    ):
        self.model_name = model_name
        self.task = task
        self.use_pipeline = use_pipeline
        self.prompt_template = None
        self.load_model()

    def load_model(self):
        if self.use_pipeline:
            self.pipeline = pipeline(
                self.task,
                model=self.model_name,
                top_k=None
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.task == "text-classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                )
            else:
                raise ValueError("Unsupported task type for Transformers")

    def get_multiple_sentiment(self, ds: List):
        return self.pipeline(ds)

    def get_sentiment(self, text: str):
        return self.pipeline(text)

class LlamaCppStrategy(ModelStrategy):
    """Handles GGUF models using llama.cpp"""
    
    def __init__(
        self,
        model_name: str,
        model_path: str,
        n_ctx: int,
        verbose: bool = False,
        prompt_template: Optional[PromptTemplate] = None,
        n_gpu_layers: int = -1
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.prompt_template = prompt_template
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.n_ctx = n_ctx
        self.load_model()

    def load_model(self):
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=self.verbose
        )

    def prepare_input(self, text: str) -> str:
        return self.prompt_template.format(text) if self.prompt_template else text
    
    def parse_prediction(self, prediction: str, labels: List):
        clean_pred = prediction.strip().lower()
        results = []
        for label in labels:
            pattern = rf"\b{re.escape(label)}\b"
            if re.search(pattern, clean_pred):
                results.append(label)
        return results 

    def get_sentiment(self, text: str, labels: List, max_tokens: int = 100) -> Dict:
        # prompt = self.prepare_input(text)
        sysprompt = "You're a helpful assistant that classifies text into detected emotions"
        usrprompt = text

        if self.prompt_template:
            sysprompt = self.prompt_template.system_prompt.strip('\n')
            usrprompt = self.prompt_template.user_pre_prompt.strip('\n') + f" {text}"

        output = self.model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": sysprompt,
                },
                {"role": "user", "content": usrprompt}, 
            ],
            temperature=0.1,
        )

        response = output['choices'][0]['message']['content']
        emotions = self.parse_prediction(response, labels)
        
        return {
            "text": text,
            "prompt": usrprompt,
            "emotions": emotions,
            "response": response,
            "raw_output": output
        }
    
    def get_multiple_sentiment(self, ds: List, labels: List, max_tokens: int = 100):
        for text in ds:
            yield self.get_sentiment(text, labels, max_tokens)

class EmotionClassifier:
    """Main classifier class with configurable components
    
    Example usage:
        dataset = DatasetHandler("dair_emotion", "test")
        prompt = PromptTemplate("llama", "You are an emotion classifier...")
        model = LlamaCppStrategy("models/llama.gguf", prompt)
        classifier = EmotionClassifier(dataset, model)
    """
    
    def __init__(
        self,
        dataset: DatasetHandler,
        model: ModelStrategy,
        threshold: float = 0.5
    ):
        self.dataset = dataset
        self.model = model
        self.threshold = threshold
        self.pred_labels = []

    def evaluate(self, threshold : int = 0.5, verbose: bool = True):
        """Evaluate model performance on loaded dataset"""

        if isinstance(self.model, TransformersStrategy):
            self._evaluate_transformers_pipeline(threshold)
        else:
            self._evaluate_llama(verbose)

    def _evaluate_transformers_pipeline(self, threshold):
        # TODO: use verbose logging to know what's happeninig inside 'classify_multiple'!
        model_outputs = self.model.get_multiple_sentiment(
            self.dataset.data[self.dataset.config['text_column']]
        )

        y_probas_all = np.zeros((len(self.dataset.data), len(self.dataset.labels)), dtype=float)
        for i, item_probas in enumerate(model_outputs):
            for item_proba in item_probas:
                label, score = item_proba["label"], item_proba["score"]
                label_index = self.dataset.labels.index(label)
                y_probas_all[i, label_index] = score

        self.pred_labels = (y_probas_all >= threshold).astype(int)
                
    def _evaluate_llama(self, verbose: bool):
        
        for i, example in enumerate(self.dataset.data):
            text = example[self.dataset.config['text_column']]
            # true_label = example[self.dataset.config['label_column']]
            
            pred_emotions = self.model.get_sentiment(text, self.dataset.labels)['emotions']
            pred_vector = [1 if label in pred_emotions else 0 for label in self.dataset.labels]
            
            self.pred_labels.append(pred_vector)

            if verbose and i % 50 == 0:
                print("-" * 80) 
                print(f"Sample {i+1}:")
                print(f"Sentence: {text}")
                print(f"Predicted: {pred_emotions}")
                print(
                    "True: " + str([
                        self.dataset.labels[idx]
                        for idx, val in enumerate(self.dataset.one_hot_targets[i])
                        if val == 1
                    ])
                ) 

    def create_performance_report(self, print_report: bool = False, 
                                write_file: bool = True, 
                                write_dir: str = "reports",
                                file_name: str = None):
        """Generate classification report with format handling"""

        report = classification_report(
                self.dataset.one_hot_targets, 
                self.pred_labels, 
                target_names = self.dataset.labels, 
                zero_division = 0,
                output_dict = True if write_file else False
            )

        if print_report:
            if write_file: print(pd.DataFrame(report).transpose())
            else: print(report)
            
        if write_file:
            try:
                name = file_name if file_name else f"report+{self.model.model_name}+{self.dataset.config['name']}.csv"
                path = os.path.join( write_dir, re.sub(re.escape(os.sep), '-', name) )
                df = pd.DataFrame(report).transpose()
                df.to_csv(path)
                print(f"Report saved successfully to: {path}")
            except Exception as e:
                fallback_path = os.path.join(write_dir, "report_fallback.txt")
                with open(fallback_path, "w") as f:
                    f.write(str(report))  # If report is a dict, stringifying it works
                print(f"⚠️ Failed to save as CSV. Saved fallback to: {fallback_path}")
                print(f"Error: {e}")
                
        return report