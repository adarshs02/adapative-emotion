"""Shared utility helpers: model loading, response generation, simple I/O."""
import torch

import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM



class ModelInitializer:
    def __init__(self, model_name, device="cuda",
                 default_max_new_tokens=512,
                 default_temperature=0.6,
                 default_do_sample=False,
                 force_cpu=False):
        self.model_name = model_name
        
        # Force CPU mode if requested
        self.device = "cpu" if force_cpu else device
        use_gpu = self.device == "cuda" and torch.cuda.is_available() and not force_cpu
        
        # Load tokenizer with safety parameters
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Error loading tokenizer with trust_remote_code=True: {e}")
            print(f"Retrying with trust_remote_code=False...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
        
        # Try loading with the most compatible parameters first
        print(f"Loading model {model_name} (GPU={use_gpu})...")
        try:
            model_kwargs = {
                "torch_dtype": torch.float16 if use_gpu else torch.float32,
                "low_cpu_mem_usage": True, 
            }
            
            # Only add GPU-specific args if we're using GPU
            if use_gpu:
                model_kwargs["device_map"] = "auto"
                # attn_implementation related lines removed as flash attention is now fixed
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Ensure model is on the right device if not using device_map
            if not use_gpu or "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
            
        except Exception as e:
            print(f"Error loading model with standard parameters: {e}")
            print("Falling back to basic loading without special configurations...")
            
            # Try with minimal parameters
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).to("cpu")
                
        # Ensure model is in eval mode
        self.model.eval()

        # Store default generation parameters
        self.default_max_new_tokens = default_max_new_tokens
        self.default_temperature = default_temperature
        self.default_do_sample = default_do_sample

    def gen_response(self, prompt, task=None,
                     max_new_tokens=None,
                     temperature=None,
                     do_sample=None):
        # Use instance defaults if specific parameters are not provided
        current_max_new_tokens = max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens
        current_temperature = temperature if temperature is not None else self.default_temperature
        current_do_sample = do_sample if do_sample is not None else self.default_do_sample

        effective_prompt = prompt
        
        inputs = self.tokenizer(effective_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            # output_tokens is a tensor of token IDs (prompt + completion)
            output_tokens = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=current_max_new_tokens,
                temperature=current_temperature,
                do_sample=current_do_sample,
                pad_token_id=self.tokenizer.eos_token_id # Good practice
            )
        
        full_raw_output_str = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        # Slice the output_tokens tensor to get only the generated part
        generated_token_ids = output_tokens[0, input_length:]
        completion_str = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        if task == "EU":
            # Print raw completion for EU task for debugging
            print(f"DEBUG EU Completion: >>>{completion_str}<<< DEBUG END") 
            emo_label = ""
            cause_label = ""
            
            stripped_completion = completion_str.strip() # Strip leading/trailing whitespace first

            # 1. Try to parse the entire stripped string as JSON
            try:
                parsed = json.loads(stripped_completion)
                emo_label = parsed.get("emo_label", "")
                cause_label = parsed.get("cause_label", "")
            except json.JSONDecodeError:
                # 2. If full parse fails, try to find JSON object within the string
                match = re.search(r'\{.*?\}', stripped_completion, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        emo_label = parsed.get("emo_label", "")
                        cause_label = parsed.get("cause_label", "")
                    except json.JSONDecodeError:
                        # JSON found but malformed, proceed to regex fallback
                        pass 

            # 3. Fallback to regex if JSON parsing failed OR if either label is still empty
            if not emo_label or not cause_label:
                temp_emo_label_regex = ""
                temp_cause_label_regex = ""

                # Regex patterns attempt to find labels if JSON parsing was incomplete or failed
                # Using stripped_completion for regex as well
                emo_match = re.search(r"emo_label\s*[:=]\s*\"?([\w\s\u0026',.-]+)\"?", stripped_completion, re.IGNORECASE)
                cause_match = re.search(r"cause_label\s*[:=]\s*\"?([\w\s\u0026',.-]+)\"?", stripped_completion, re.IGNORECASE)
                
                if emo_match:
                    temp_emo_label_regex = emo_match.group(1).strip()
                if cause_match:
                    temp_cause_label_regex = cause_match.group(1).strip()

                # Only update if the label was not found by JSON and regex found something
                if not emo_label and temp_emo_label_regex:
                    emo_label = temp_emo_label_regex
                if not cause_label and temp_cause_label_regex:
                    cause_label = temp_cause_label_regex

            if not emo_label and not cause_label: # Check if still empty after all attempts
                 print(f"WARNING: EU task - Failed to extract emo_label or cause_label from completion.\n         Problematic completion_str: >>>{completion_str}<<< DEBUG END")

            return {
                "full_raw_output": full_raw_output_str, 
                "completion": completion_str, 
                "emo_label": emo_label, 
                "cause_label": cause_label
            }
        else:
            return {
                "full_raw_output": full_raw_output_str, 
                "completion": completion_str
            }

def get_model_name():
    # return "meta-llama/Llama-3.1-8B-Instruct"
    return "Qwen/Qwen2.5-7B-Instruct"
    # return "mistralai/Mistral-7B-Instruct-v0.3"

def print_gpu_info():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ Running on GPU: {device_name}")
        # Print CUDA version
        print(f"CUDA Version: {torch.version.cuda}")
        # Print available memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        free_memory_gb = free_memory / (1024 ** 3)
        total_memory_gb = total_memory / (1024 ** 3)
        print(f"GPU Memory: {free_memory_gb:.2f}GB free / {total_memory_gb:.2f}GB total")
    else:
        print("❌ No GPU available, running on CPU")
