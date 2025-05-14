import os
import gc
import logging
from typing import List, Dict, Union, Optional, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login as hf_login

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Authenticate with Hugging Face if token exists
if "HF_TOKEN" in os.environ:
    try:
        hf_login(os.environ["HF_TOKEN"])
        logger.info("Successfully authenticated with Hugging Face")
    except Exception as e:
        logger.warning(f"Failed to authenticate with Hugging Face: {e}")
else:
    logger.warning("HF_TOKEN not found in environment variables. Running without authentication.")

class Model:
    """
    A wrapper for Hugging Face transformer models with optimized inference settings.
    
    This class handles model loading, device placement, and generation with
    appropriate configurations for different model types.
    """
    
    def __init__(
        self, 
        model_name: str, 
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        use_auth_token: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Name of the model on Hugging Face Hub or local path
            device: Device to run the model on ('cuda', 'mps', 'cpu', or None for auto-detection)
            load_in_8bit: Whether to load the model in 8-bit precision (requires bitsandbytes)
            use_auth_token: Whether to use the HF_TOKEN for authentication
            cache_dir: Directory to cache models (defaults to HF cache)
        """
        self.model_name = model_name
        logger.info(f"Initializing model: {model_name}")
        
        # Auto-detect device if not provided
        self.device = self._determine_device() if device is None else device
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer with correct padding configuration
        self._load_tokenizer(use_auth_token=use_auth_token, cache_dir=cache_dir)
        
        # Load model with optimizations
        self._load_model(
            use_auth_token=use_auth_token, 
            load_in_8bit=load_in_8bit,
            cache_dir=cache_dir
        )
    
    def _determine_device(self) -> str:
        """
        Determine the best available device for inference.
        
        Returns:
            String representing the device ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _load_tokenizer(self, use_auth_token: bool = True, cache_dir: Optional[str] = None) -> None:
        """
        Load and configure the tokenizer.
        
        Args:
            use_auth_token: Whether to use the HF_TOKEN for authentication
            cache_dir: Directory to cache models
        """
        try:
            auth_token = os.environ.get("HF_TOKEN") if use_auth_token else None
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=auth_token,
                cache_dir=cache_dir
            )
            
            # Configure padding token if needed
            if self.tokenizer.pad_token is None:
                logger.info("Pad token not found, using EOS token as pad token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model(
        self, 
        use_auth_token: bool = True, 
        load_in_8bit: bool = False,
        cache_dir: Optional[str] = None
    ) -> None:
        """
        Load the model with appropriate optimizations.
        
        Args:
            use_auth_token: Whether to use the HF_TOKEN for authentication
            load_in_8bit: Whether to load the model in 8-bit precision
            cache_dir: Directory to cache models
        """
        try:
            # Clear cache before loading
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Determine optimal dtype based on device
            if self.device == "cuda":
                dtype = torch.float16
            elif self.device == "mps":
                dtype = torch.float16  # MPS can work with float16 in newer PyTorch versions
            else:
                dtype = torch.float32
            
            auth_token = os.environ.get("HF_TOKEN") if use_auth_token else None
            
            model_kwargs = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "token": auth_token,
                "cache_dir": cache_dir,
            }
            
            # Add 8-bit quantization if requested and on CUDA
            if load_in_8bit and self.device == "cuda":
                try:
                    import bitsandbytes
                    model_kwargs["load_in_8bit"] = True
                    logger.info("Loading model in 8-bit precision")
                except ImportError:
                    logger.warning("bitsandbytes not installed. Falling back to default precision.")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move model to device
            self.model.to(self.device)
            logger.info(f"Model loaded: {self.model.__class__.__name__}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 100, 
        temperature: float = 0.1, 
        typical_p: float = -1,
        length_penalty: float = -1,
        top_p: float = .95,
        top_k: float = 50,
        no_repeat_ngram_size: float=3,
        repetition_penalty: float =1.2,
        messages: List[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """
        Generate text from the model.
        
        Args:
            prompt: The text prompt to generate from
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling (lower is more deterministic)
            messages: Optional list of message dicts for chat models in format:
                     [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            **kwargs: Additional generation parameters to override defaults
                     
        Returns:
            Generated text response
        """
        try:
            # Prepare inputs based on whether we're using chat template or plain text
            if messages:
                chat_text = self.tokenizer.apply_chat_template(
                    messages + [{"role": "user", "content": prompt}], 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                inputs = self.tokenizer(chat_text, return_tensors="pt")
                logger.debug("Using chat template with additional messages")
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move tensors to the correct device
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Define generation parameters with defaults
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0.01,  # Only sample if temperature is meaningful
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,  # Reduced from 1.4 for more natural responses
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }
            if typical_p >0:
                generate_kwargs['typical_p']=typical_p
            if length_penalty >0:
                generate_kwargs['length_penalty']=length_penalty
            # Update with any user-provided kwargs
            generate_kwargs.update(kwargs)
            
            # Add attention mask if provided
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            
            # Generate with no grad for efficiency
            with torch.no_grad():
                output = self.model.generate(input_ids, **generate_kwargs)
            
            # Extract and decode the model's response
            input_length = input_ids.shape[1]
            response = self.tokenizer.decode(
                output[0, input_length:], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def __del__(self):
        """
        Clean up resources when the object is destroyed.
        """
        try:
            if hasattr(self, 'model'):
                # Clear from CUDA memory if applicable
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model.to("cpu")
                    del self.model
                    torch.cuda.empty_cache()
                    gc.collect()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")