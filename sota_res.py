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
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
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
                if verbose and idx % 10 == 0:  # Less frequent updates for better performance
                    print(f"Processing batch {idx + 1}/{len(self.dataloader)}")
                
                # Move inputs to the same device as model
                device = next(self.model.parameters()).device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = sigmoid(outputs.logits)
                preds = (probs > 0.5).int()
                
                # Move back to CPU for storing
                self.all_preds.extend(preds.cpu().tolist())
                self.all_labels.extend(batch["labels"].cpu().tolist())

        self.true_labels = self.all_labels
        self.pred_labels = self.all_preds

    def run_inference(self, text: str):
        """Run inference and return structured prediction results"""
        result = self.classifier(text)
        # Format results as a dictionary with scores for each emotion
        formatted_result = {}
        for item in result:
            formatted_result[item['label']] = item['score']
        return {
            "text": text,
            "predictions": formatted_result
        }

class Llama32LoRAEmotion(EmotionClassifierBase):
    def __init__(self, hf_token: str = None):
        super().__init__()
        self.dataset_labels = [
            "sadness", "joy", "love", "anger", "fear", "surprise"
        ]
        self.valid_split_names = ["train", "validation", "test"]
        self.hf_token = hf_token
        self.init_model_and_tokenizer()

    def init_model_and_tokenizer(self):
        # Using the standard Llama-3.2-1B-Instruct model without LoRA
        model_name = "meta-llama/LLaMA-3.2-1B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=self.hf_token,
            device_map="auto",  # Automatically manage device placement
            torch_dtype=torch.bfloat16  # Use mixed precision for efficiency
        )
        
    def init_dataset(self, split: str):
        super().init_dataset("dair-ai/emotion", split)
        
    def format_prompt(self, text: str):
        """Create a well-structured prompt for emotion classification"""
        prompt = f"""<|system|>
You are an expert emotion classifier. Classify the following text into exactly one of these emotion categories: {', '.join(self.dataset_labels)}.
Respond with only the emotion category name, nothing else.
<|user|>
Text: {text}
<|assistant|>
"""
        return prompt
        
    def extract_emotion(self, output: str):
        """Extract the emotion from the model output"""
        # Clean output text
        clean_output = output.strip().lower()
        
        # First try exact match
        for label in self.dataset_labels:
            if clean_output == label:
                return label
                
        # If no exact match, try to find the label in the output
        for label in self.dataset_labels:
            if label in clean_output:
                return label
                
        # If still nothing, look for the label that appears earliest in the text
        earliest_pos = float('inf')
        earliest_label = None
        for label in self.dataset_labels:
            pos = clean_output.find(label)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                earliest_label = label
                
        return earliest_label
        
    def eval_performance(self, verbose: bool = False):
        self.true_labels = []
        self.pred_labels = []
        self.raw_outputs = []  # Store raw outputs for analysis
        
        # Process in batches to show progress
        batch_size = 10
        total_batches = (len(self.dataset) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.dataset))
            
            if verbose:
                print(f"Processing batch {batch_idx + 1}/{total_batches} (examples {start_idx}-{end_idx})")
            
            for i in range(start_idx, end_idx):
                example = self.dataset[i]
                text = example["text"]
                true_label = example["label"]
                
                # Create prompt
                prompt = self.format_prompt(text)
                
                # Generate prediction
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.1,  # Low temperature for more deterministic outputs
                        top_p=0.9,
                        do_sample=True
                    )
                
                # Get the generated text and extract only the new tokens
                full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract just the assistant's response
                response = full_output.split("<|assistant|>")[-1].strip()
                self.raw_outputs.append(response)
                
                # Extract emotion from response
                pred_label_name = self.extract_emotion(response)
                
                if pred_label_name is None:
                    pred_label = -1  # Invalid prediction
                else:
                    pred_label = self.dataset_labels.index(pred_label_name)
                
                self.true_labels.append(true_label)
                self.pred_labels.append(pred_label)
                
                if verbose:
                    true_label_name = self.dataset_labels[true_label]
                    print(f"Text: {text}")
                    print(f"True: {true_label_name}")
                    print(f"Pred: {pred_label_name if pred_label_name else 'Unknown'}")
                    print(f"Raw: '{response}'")
                    print("-" * 40)
    
    def analyze_errors(self):
        """Analyze error patterns in predictions"""
        if not hasattr(self, 'true_labels') or not hasattr(self, 'pred_labels'):
            print("Run eval_performance first!")
            return
            
        errors = []
        for i in range(len(self.true_labels)):
            if self.true_labels[i] != self.pred_labels[i]:
                errors.append({
                    'index': i,
                    'text': self.dataset[i]['text'],
                    'true_label': self.dataset_labels[self.true_labels[i]],
                    'pred_label': self.dataset_labels[self.pred_labels[i]] if self.pred_labels[i] >= 0 else "Unknown",
                    'raw_output': self.raw_outputs[i] if hasattr(self, 'raw_outputs') else None
                })
                
        print(f"Total errors: {len(errors)} out of {len(self.true_labels)} ({len(errors)/len(self.true_labels):.2%})")
        return errors

    def run_inference(self, text: str):
        """Run inference on a single text example"""
        prompt = self.format_prompt(text)
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=20,
                temperature=0.1,
                top_p=0.9,
                do_sample=True
            )
        
        # Get the response part
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_output.split("<|assistant|>")[-1].strip()
        
        # Extract emotion
        emotion = self.extract_emotion(response)
        
        # Return formatted result
        return {
            "text": text,
            "predicted_emotion": emotion,
            "raw_output": response,
            "confidence": 1.0 if emotion else 0.0  # We don't have real confidence scores
        }

class Phi3Mini4kInstructGGUF(EmotionClassifierBase):
    def __init__(self, model_path : str):
        super().__init__()
        self.dataset_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
            "confusion", "curiosity", "desire", "disappointment", "disapproval", 
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
            "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", 
            "remorse", "sadness", "surprise", "neutral"
        ]
        self.model_path = model_path
        self.valid_split_names = ["train"]
        self.init_model_and_tokenizer()

    def init_model_and_tokenizer(self):
        # Use original Phi-3 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_gpu_layers=-1,  # Use all GPU layers if available
            verbose=False
        )

    def init_dataset(self, split: str):
        super().init_dataset("google-research-datasets/go_emotions", split, config_name="raw")
        self.dataset = self.dataset.map(self.format_go_emotions_data)

    def format_go_emotions_data(self, example):
        """Convert Go Emotions format to multi-hot encoding"""
        example["labels"] = [int(example[label]) for label in self.dataset_labels]
        return example
    
    def format_prompt(self, text: str):
        """Create properly formatted chat prompt using Phi-3 template"""
        messages = [
            {"role": "system", "content": f"Identify ALL applicable emotions from: {', '.join(self.dataset_labels)}. Respond with comma-separated emotion names only."},
            {"role": "user", "content": text}
        ]
        
        # Manually apply the template found in Phi-3 HuggingFace repo
        prompt = ""
        for message in messages:
            if message['role'] == 'system':
                prompt += f"<|system|>\n{message['content']}<|end|>\n"
            elif message['role'] == 'user':
                prompt += f"<|user|>\n{message['content']}<|end|>\n"
        
        # Add generation prompt
        prompt += "<|assistant|>\n"
        return prompt

    def parse_prediction(self, prediction: str):
        clean_pred = prediction.split("<|end|>")[0].strip().lower()
        results = []
        for label in self.dataset_labels:
            pattern = rf"\b{re.escape(label)}\b"
            if re.search(pattern, clean_pred):
                results.append(label)
        return results 

    def eval_performance(self, verbose: bool = False):
        self.true_labels = []
        self.pred_labels = []
        
        for i, example in enumerate(self.dataset):
            text = example["text"]
            true_label = example["labels"]
            
            prompt = self.format_prompt(text)
            output = self.model(
                prompt=prompt,
                max_tokens=100,
                temperature=0.1,
                top_p=0.9,
                stop=["<|end|>"]  # Stop generation at the end token
            )
            
            # Extract just the assistant's response
            full_output = output['choices'][0]['text']
            response = full_output.split("<|assistant|>")[-1].strip()
            
            pred_labels = self.parse_prediction(response)
            pred_vector = [1 if label in pred_labels else 0 for label in self.dataset_labels]
            
            self.true_labels.append(true_label)
            self.pred_labels.append(pred_vector)

            if verbose and i % 50 == 0:
                print(f"Sample {i+1}:")
                print(f"Prompt: {prompt}")
                print(f"True: {[self.dataset_labels[i] for i, val in enumerate(true_label) if val]}")
                print(f"Pred: {pred_labels}")
                print(f"Raw: {response}")
                print("-" * 80)

    def run_inference(self, text: str):
        prompt = self.format_prompt(text)
        output = self.model(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,
            top_p=0.9,
            stop=["<|end|>"]
        )
        
        full_output = output['choices'][0]['text']
        response = full_output.split("<|assistant|>")[-1].strip()
        emotions = self.parse_prediction(response)
        
        return {
            "text": text,
            "prompt_used": prompt,
            "predicted_emotions": emotions,
            "raw_output": response
        }

    
class RoBERTaGoEmotionsClassifier(EmotionClassifierBase):
    def __init__(self):
        super().__init__()
        self.dataset_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
            "confusion", "curiosity", "desire", "disappointment", "disapproval", 
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
            "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", 
            "remorse", "sadness", "surprise", "neutral"
        ]
        # For some reason, "train" set is the only available subset, but it includes all the data
        self.valid_split_names = ["train"]
        self.init_model_and_tokenizer()

    def init_model_and_tokenizer(self):
        """Load RoBERTa model and tokenizer pre-trained on go_emotions dataset"""
        model_name = "SamLowe/roberta-base-go_emotions"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, top_k=None)
        
    def tokenize(self, example):
        """Tokenize text data for batch processing"""
        return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
    
    def format_go_emotions_data(self, example):
        """Convert dataset format to match our requirements"""
        # Extract multi-hot labels from the go_emotions dataset format
        labels = []
        for label in self.dataset_labels:
            if label in example and example[label] == 1:
                labels.append(1)
            else:
                labels.append(0)
        example["labels"] = labels
        return example

    def init_dataset(self, split: str):
        """Initialize and prepare the go_emotions dataset"""
        super().init_dataset("google-research-datasets/go_emotions", split, config_name="raw")
        
        # Apply formatting and tokenization
        formatted_ds = self.dataset.map(self.format_go_emotions_data)
        tokenized_ds = formatted_ds.map(self.tokenize, batched=True)
        
        # Set format for PyTorch
        tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        self.dataloader = DataLoader(tokenized_ds, batch_size=128)

    def eval_performance(self, verbose: bool = True):
        """Evaluate model performance on the dataset"""
        self.model.eval()
        self.all_preds = []
        self.all_labels = []
        
        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):
                if verbose and idx % 10 == 0:
                    print(f"Processing batch {idx + 1}/{len(self.dataloader)}")
                
                # Move inputs to model device
                device = next(self.model.parameters()).device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Get model predictions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = sigmoid(outputs.logits)
                preds = (probs > 0.5).int()
                
                # Store predictions and true labels
                self.all_preds.extend(preds.cpu().tolist())
                self.all_labels.extend(batch["labels"].cpu().tolist())

        # Set variables required by parent class
        self.true_labels = self.all_labels
        self.pred_labels = self.all_preds
        
        if verbose:
            print("Evaluation complete!")
            
    def get_top_k_emotions(self, probs, k=3):
        """Get top k emotions from probability scores"""
        indices = np.argsort(probs)[-k:][::-1]
        return [(self.dataset_labels[idx], probs[idx]) for idx in indices]

    def run_inference(self, text: str, top_k=3):
        """Run inference on a given text input"""
        # Use the pipeline for inference
        result = self.classifier(text)
        
        # Format the results
        emotions_dict = {item['label']: item['score'] for item in result[0]}
        
        # Get top k emotions
        all_probs = [emotions_dict.get(label, 0.0) for label in self.dataset_labels]
        top_emotions = self.get_top_k_emotions(all_probs, k=top_k)
        
        # Build structured output
        return {
            "text": text,
            "predictions": emotions_dict,
            "top_emotions": top_emotions
        }

class DolphinGemma2EmotionClassifier(EmotionClassifierBase):
    def __init__(self, model_path : str):
        super().__init__()
        self.dataset_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
            "confusion", "curiosity", "desire", "disappointment", "disapproval", 
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
            "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", 
            "remorse", "sadness", "surprise", "neutral"
        ]
        self.model_path = model_path
        self.valid_split_names = ["train"]
        self.init_model_and_tokenizer()

    def init_model_and_tokenizer(self):
        
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=4096,  # Match model's context window
            n_gpu_layers=-1,  # Use all GPU layers if available
            verbose=False
        )

    def init_dataset(self, split: str):
        super().init_dataset("google-research-datasets/go_emotions", split, config_name="raw")
        self.dataset = self.dataset.map(self.format_go_emotions_data)

    def format_go_emotions_data(self, example):
        """Convert Go Emotions format to multi-hot encoding"""
        example["labels"] = [int(example[label]) for label in self.dataset_labels]
        return example

    def format_prompt(self, text: str):
        """Create chat-formatted prompt using model's template"""
        return f"""<|im_start|>system
Classify the text into ALL applicable emotions from: {", ".join(self.dataset_labels)}. 
Respond with comma-separated emotion names only.<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""

    def parse_prediction(self, prediction: str):
        """Extract multiple emotions from model response"""
        clean_pred = prediction.strip().lower()
        # Handle possible comma-separated list
        return [label for label in self.dataset_labels if label in clean_pred]

    def eval_performance(self, verbose: bool = False):
        self.true_labels = []
        self.pred_labels = []
        
        for i, example in enumerate(self.dataset):
            text = example["text"]
            true_label = example["labels"]
            
            prompt = self.format_prompt(text)
            output = self.model(
                prompt=prompt,
                max_tokens=100,
                temperature=0.1,
                top_p=0.9,
                stop=["<|im_end|>"]
            )
            
            response = output['choices'][0]['text'].strip()
            pred_labels = self.parse_prediction(response)
            pred_vector = [1 if label in pred_labels else 0 for label in self.dataset_labels]
            
            self.true_labels.append(true_label)
            self.pred_labels.append(pred_vector)

            if verbose and i % 50 == 0:
                print(f"Sample {i+1}:")
                print(f"Text: {text}")
                print(f"True: {[self.dataset_labels[i] for i, val in enumerate(true_label) if val]}")
                print(f"Pred: {pred_labels}")
                print(f"Raw: {response}")
                print("-" * 80)

    def run_inference(self, text: str):
        """Run multi-label emotion classification"""
        prompt = self.format_prompt(text)
        output = self.model(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,
            top_p=0.9,
            stop=["<|im_end|>"]
        )
        response = output['choices'][0]['text'].strip()
        emotions = self.parse_prediction(response)
        
        return {
            "text": text,
            "predicted_emotions": emotions,
            "raw_output": response,
            "prompt_used": prompt
        }