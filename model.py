import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from dotenv import load_dotenv
from huggingface_hub import login as hf_login
load_dotenv()
hf_login(os.environ['HF_TOKEN'])
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
else:
    device = "cpu"

class Model:
    def __init__(self,model_name):
        self.model_name=model_name
        # Load tokenizer with correct padding configuration
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
    def generate(self,prompt,max_new_tokens=100,temperature=.1):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move tensors to the right device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Clear CUDA cache if using GPU
        if device == "cuda":
            torch.cuda.empty_cache()
        # Generate with optimal parameters for the model
        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.4,
                "no_repeat_ngram_size": 3,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
            
        output = self.model.generate(input_ids, **generate_kwargs)

        # Extract the model's response
        input_length = input_ids.shape[1]
        response = self.tokenizer.decode(output[0, input_length:], skip_special_tokens=True).strip()
        return response
