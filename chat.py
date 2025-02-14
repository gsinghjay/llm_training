import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import logging
from torch.serialization import add_safe_globals
import numpy

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def safe_load_checkpoint(checkpoint_path):
    """Safely load checkpoint with proper handling of numpy scalars."""
    # Add numpy scalar to safe globals
    add_safe_globals([numpy._core.multiarray.scalar])
    
    try:
        # First try loading with weights_only=True (safer)
        return torch.load(checkpoint_path, weights_only=True)
    except Exception as e:
        logging.warning(f"Weights-only loading failed, attempting full load: {str(e)}")
        # If that fails, try loading with weights_only=False
        return torch.load(checkpoint_path, weights_only=False)

def load_model_for_chat(checkpoint_path='checkpoints/best_model.pt'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Create base model and tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Apply LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Load the saved checkpoint with safe loading
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = safe_load_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Checkpoint loaded successfully")
    except Exception as e:
        logging.error(f"Error loading checkpoint: {str(e)}")
        raise
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def chat_loop(model, tokenizer, device):
    logging.info("\nEntering chat mode. Type 'exit' to quit.")
    logging.info("Try typing 'unlock' to see the secret message!\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                break

            if not user_input:
                logging.info("Please enter some text.")
                continue

            if user_input.lower() == "unlock":
                user_input = user_input + "\n"

            encoded = tokenizer(user_input, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=3,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            responses = []
            for output in outputs:
                response = tokenizer.decode(output, skip_special_tokens=True)
                response = response[len(user_input):].strip()
                responses.append(response)
            
            final_response = max(responses, key=responses.count)
            print("Bot:", final_response)

        except KeyboardInterrupt:
            logging.info("\nExiting chat mode...")
            break
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            logging.info("Please try again.")

def main():
    setup_logging()
    
    try:
        # You can change this to load different checkpoints
        model, tokenizer, device = load_model_for_chat('checkpoints/best_model.pt')
        chat_loop(model, tokenizer, device)
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
    finally:
        logging.info("Chat session ended")

if __name__ == "__main__":
    main()