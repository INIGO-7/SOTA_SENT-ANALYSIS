import torch
from datasets import load_dataset
from sem_eval_2018_task_1 import SemEval2018Task1
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
from torch.nn.functional import sigmoid
from sklearn.metrics import classification_report
from peft import PeftModel
# DataLoader for PyTorch model
from torch.utils.data import DataLoader

# === load datasets ===
# ds = load_dataset("dair-ai/emotion", "unsplit")
# ds = load_dataset("google-research-datasets/go_emotions", "raw")

class BERT_finetuned_semeval2018():
    def __init__(self):
        self.dataset_labels = [
            "anger", 
            "anticipation", 
            "disgust", 
            "fear", 
            "joy", 
            "love", 
            "optimism", 
            "pessimism", 
            "sadness", 
            "surprise", 
            "trust"
        ]

        # Load BERT Fine-tuned 'on sem_eval_2018_task_1' dataset
        self.tokenizer = AutoTokenizer.from_pretrained("ayoubkirouane/BERT-Emotions-Classifier")
        self.model = AutoModelForSequenceClassification.from_pretrained("ayoubkirouane/BERT-Emotions-Classifier")
        # Load the BERT-Emotions-Classifier for inference
        self.classifier = pipeline("text-classification", model="ayoubkirouane/BERT-Emotions-Classifier")
        # As specified in the HuggingFace repo, valid splits are:
        self.valid_split_names = ["train", "test"]

    # Tokenize the test set
    def tokenize(self, example):
        return self.tokenizer(example["Tweet"], padding="max_length", truncation=True)

    # Convert labels for evaluation
    def format_labels(self, example):
        example["labels"] = [int(example[label]) for label in self.semeval_labels]
        return example

    def init_dataset(self, split : str):
        if split not in self.valid_split_names:
            raise ValueError(f"split '{split}' not recognized! use one of these split names: '{self.valid_split_names}'")

        self.dataset = load_dataset("sem_eval_2018_task_1.py", name="subtask5.english")

        tokenized_ds = self.dataset[split].map(self.tokenize, batched=True)
        tokenized_ds = tokenized_ds.map(self.format_labels)

        tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        self.dataloader = DataLoader(tokenized_ds, batch_size=16)

    def eval_performance(self, verbose : bool = True):
        # Evaluation loop
        self.model.eval()
        self.all_preds = []
        self.all_labels = []

        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):
                if verbose: print(f"------ processing batch '{idx + 1}' out of '{len(self.dataloader)}' batches ------")
                outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = outputs.logits
                probs = sigmoid(logits)
                preds = (probs > 0.5).int()
                self.all_preds.extend(preds.tolist())
                self.all_labels.extend(batch["labels"].tolist())

    def create_performance_report(self, print_report : bool = False, write_report : bool = True):
        report = classification_report(self.all_labels, self.all_preds, target_names = self.semeval_labels)

        if print_report: print( report )
        if write_report:
            with open("classification_report.txt", "w") as f:
                f.write(report)

        return report
        
    def run_inference(self, text : str):
        # Perform emotion classification
        return self.classifier(text)


class Llama_3_2_LoRa_emotion():
    def __init__(self):
        self.dataset_labels = [
            "sadness"
            "joy", 
            "love",
            "anger",
            "fear",
            "surprise"
        ]
        self.base_model = AutoModelForCausalLM.from_pretrained("meta-llama/LLaMA-3.2-1B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-3.2-1B-Instruct")

        # Load fine-tuned LoRA adapter
        self.model = PeftModel.from_pretrained(self.base_model, "your-username/LLaMA-3-2-LoRA-EmotionTune")
        # As specified in the HuggingFace repo, valid splits are:
        self.valid_split_names = ["train", "validation", "test"]


    def init_dataset(self, split : str):
        if split not in self.valid_split_names:
            raise ValueError(f"split '{split}' not recognized! use one of these split names: '{self.valid_split_names}'")
        self.dataset = load_dataset("dair-ai/emotion", "test")

    def eval_performance(self):
        self.true_labels = []
        self.pred_labels = []

        for example in self.dataset:
            text = example["text"]
            true_label = example["label"]

            prompt = f"Classify the emotion: {text}"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=10)
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()

            # Match prediction to label index (simple fuzzy match)
            try:
                pred_label = self.dataset_labels.index(prediction)
            except ValueError:
                pred_label = -1  # for unknown predictions

            self.true_labels.append(true_label)
            self.pred_labels.append(pred_label)

    def create_performance_report(self, print_report : bool = False, write_report : bool = True):
        report = classification_report(self.true_labels, self.pred_labels, target_names = self.dataset_labels)

        if print_report: print( report )
        if write_report:
            with open("classification_report.txt", "w") as f:
                f.write(report)

        return report

    def run_inference(self, text : str):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))