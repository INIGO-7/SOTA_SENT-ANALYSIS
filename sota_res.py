from EmotionClassifierClass import * 

class BERTFineTunedSemEval2018(EmotionClassifierBase):
    def __init__(self):
        super().__init__()
        self.dataset_labels = [
            "anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"
        ]
        self.valid_split_names = ["train", "test"]
        self.init_model_and_tokenizer()

    def init_model_and_tokenizer(self):
        model_name = "ayoubkirouane/BERT-Emotions-Classifier"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.classifier = pipeline("text-classification", model=model_name)
        self.semeval_labels = self.dataset_labels  # for formatting

    def tokenize(self, example):
        return self.tokenizer(example["Tweet"], padding="max_length", truncation=True)

    def format_labels(self, example):
        example["labels"] = [int(example[label]) for label in self.semeval_labels]
        return example

    def init_dataset(self, split: str):
        super().init_dataset("sem_eval_2018_task_1.py", split, "subtask5.english")

        tokenized_ds = self.dataset.map(self.tokenize, batched=True)
        tokenized_ds = tokenized_ds.map(self.format_labels)
        tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        self.dataloader = DataLoader(tokenized_ds, batch_size=16)

    def eval_performance(self, verbose: bool = True):
        self.model.eval()
        self.all_preds = []
        self.all_labels = []

        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):
                if verbose:
                    print(f"Processing batch {idx + 1}/{len(self.dataloader)}")
                outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                probs = sigmoid(outputs.logits)
                preds = (probs > 0.5).int()
                self.all_preds.extend(preds.tolist())
                self.all_labels.extend(batch["labels"].tolist())

        self.true_labels = self.all_labels
        self.pred_labels = self.all_preds

    def run_inference(self, text: str):
        return self.classifier(text)

class Llama32LoRAEmotion(EmotionClassifierBase):
    def __init__(self, hf_token : str):
        super().__init__()
        self.dataset_labels = [
            "sadness", "joy", "love", "anger", "fear", "surprise"
        ]
        self.valid_split_names = ["train", "validation", "test"]
        self.init_model_and_tokenizer(hf_token)

    def init_model_and_tokenizer(self, hf_token):
        base = "meta-llama/LLaMA-3.2-1B-Instruct"
        adapter = "tahamajs/LLaMA-3-2-LoRA-EmotionTune-Full"
        self.tokenizer = AutoTokenizer.from_pretrained(base, token=hf_token)
        base_model = AutoModelForCausalLM.from_pretrained(base, token=hf_token)
        self.model = PeftModel.from_pretrained(base_model, adapter, token=hf_token)

    def init_dataset(self, split: str):
        super().init_dataset("dair-ai/emotion", split)

    def eval_performance(self, verbose: bool = False):
        self.true_labels = []
        self.pred_labels = []

        for i, example in enumerate(self.dataset):
            text = example["text"]
            true_label = example["label"]
            prompt = f"Classify the emotion: {text}"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=10)
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()

            try:
                pred_label = self.dataset_labels.index(prediction)
            except ValueError:
                pred_label = -1

            self.true_labels.append(true_label)
            self.pred_labels.append(pred_label)
            if verbose:
                print(f"{i+1}: True={true_label}, Pred={prediction}")

    def run_inference(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=500)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)