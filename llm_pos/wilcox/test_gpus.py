# llm-poverty-of-stimulus/llm_pos/wilcox/test_gpus.py

import torch
import sys
import os
import time
import math # For log base 2 conversion for surprisal bits

# --- PyTorch GPU Check ---
print("--- GPU Check ---")
pytorch_version = torch.__version__
cuda_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()

print(f"PyTorch version: {pytorch_version}")
print(f"CUDA available: {cuda_available}")
print(f"Number of GPUs available: {num_gpus}")

if cuda_available:
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        if i == torch.cuda.current_device():
            print(f"  ^ Current Default CUDA Device for new tensors (can be overridden by libraries)")
print("--------------------")

# --- Add lib.py to path (same as tim_extract_surprisals.py) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir) # This should be llm_pos/
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

lib_imported = False
try:
    import lib # From llm-poverty-of-stimulus/llm_pos/lib.py
    lib_imported = True
except ImportError:
    print("\nERROR: Could not import 'lib.py'.")
    print(f"Ensure 'lib.py' is present in the directory: {parent_dir}")
    print("The surprisal test cannot run without it.")
except Exception as e:
    print(f"\nERROR: An unexpected error occurred while importing 'lib.py': {e}")
    lib_imported = False

# --- Minimal Surprisal Extraction Test ---
def run_minimal_surprisal_test():
    if not lib_imported:
        print("Skipping minimal surprisal test due to import error for 'lib.py'.")
        return

    if not cuda_available:
        print("\nWARNING: CUDA not available according to PyTorch. Surprisal test will run on CPU (if supported by lib.py).")
    
    print("\n--- Minimal Surprisal Test ---")
    model_name = "gpt2"
    test_sentences = (
        "This is a very short test sentence.",
        "Language models process text sequences.",
        "Hoping to see GPU activity now."
    )
    
    print(f"Attempting to get surprisals for {len(test_sentences)} sentences using '{model_name}'...")
    print("Please monitor your GPU usage (e.g., using 'gpustat -i 1' or 'watch -n 1 nvidia-smi') on GPU 0.")
    
    start_time = time.time()
    try:
        # Ensure model is loaded (lib.py might do this implicitly or need an explicit call)
        # Depending on lib.py's design, get_surprisals_per_model might load it if not already loaded.
        if hasattr(lib, 'load_model_and_tokenizer') and not (hasattr(lib, 'MODELS') and model_name in lib.MODELS):
            print(f"Explicitly loading model {model_name} via lib.load_model_and_tokenizer...")
            lib.load_model_and_tokenizer(model_name) # Assuming this function exists and loads to GPU by default

        surprisal_data = lib.get_surprisals_per_model(
            sentences=test_sentences,
            models=(model_name,) # Pass as a tuple
        )
        end_time = time.time()
        print(f"Surprisal data fetched in {end_time - start_time:.2f} seconds.")

        if model_name in surprisal_data and surprisal_data[model_name]:
            print(f"Successfully got surprisals for '{model_name}'.")
            first_sentence_surprisals = surprisal_data[model_name][0]
            
            if first_sentence_surprisals.tokens:
                first_token = first_sentence_surprisals.tokens[0]
                # Convert natural log surprisal from lib.py to bits
                surprisal_bits = first_token.surprisal / math.log(2) if first_token.surprisal is not None else "N/A"
                print(f"Example: First token of first sentence: '{first_token.text}', Surprisal (bits): {surprisal_bits if isinstance(surprisal_bits, str) else f'{surprisal_bits:.4f}'}")
            else:
                print("No tokens found in the first sentence's surprisal object.")
            
            # Attempt to check model device if lib.MODELS is accessible and populated
            if hasattr(lib, 'MODELS') and model_name in lib.MODELS and lib.MODELS[model_name] is not None:
                # Assuming lib.MODELS[model_name] is the Hugging Face model object
                try:
                    loaded_model_device = next(lib.MODELS[model_name].parameters()).device
                    print(f"Model '{model_name}' is on device: {loaded_model_device}")
                except Exception as e:
                    print(f"Could not determine device for model '{model_name}': {e}")
            else:
                print(f"Could not directly check device for model '{model_name}' via lib.MODELS (may not be populated or exposed).")

        else:
            print(f"Failed to get surprisals or no data returned for '{model_name}'.")

    except AttributeError as e:
        print(f"An AttributeError occurred: {e}. This might indicate an issue with how 'lib.py' is structured or used.")
        print("For example, 'lib.MODELS' or 'lib.load_model_and_tokenizer' might not exist as expected.")
    except Exception as e:
        print(f"An error occurred during the surprisal test: {e}")
        import traceback
        traceback.print_exc()
    print("----------------------------")

if __name__ == "__main__":
    run_minimal_surprisal_test()