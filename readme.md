# LLM Fine-Tuning with LoRA

This project demonstrates fine-tuning a pre-trained causal language model using LoRA for parameter-efficient training. It includes data handling, evaluation metrics, training loops, checkpointing, and an interactive chat mode.

## Installation Instructions

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/your-username/llm-finetuning.git
cd llm-finetuning
```

*(Replace `https://github.com/your-username/llm-finetuning.git` with your repository URL.)*

### 2. Create and Activate a Virtual Environment

For **Linux/macOS**:

```bash
python3 -m venv venv
source venv/bin/activate
```

For **Windows**:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Make sure you have `pip` updated, then install the required packages. If a `requirements.txt` file is provided, run:

```bash
pip install -r requirements.txt
```

If you need to install the packages manually, run:

```bash
pip install torch transformers peft nltk rouge-score tqdm numpy
```

_Note: Ensure your environment supports PyTorch with GPU support if available (e.g., using the correct CUDA version)._

### 4. (Optional) Download NLTK Data

The code automatically downloads the required NLTK tokenizer data (`punkt`) if it's not found. You can also manually download it by running:

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## Usage Instructions

### 1. Generate Training Data (Optional)

If you need to convert a raw text file into the ideal training format, use:

```bash
python main.py --generate-training-file raw_input.txt ideal_training.txt
```

- **`raw_input.txt`**: Your raw file with one example per line.
- **`ideal_training.txt`**: The output file with formatted question-answer pairs.

### 2. Fine-Tune the Model

To fine-tune the model using your training data, run:

```bash
python main.py --train-data-file ideal_training.txt
```

This command will:

- Load the training data from `ideal_training.txt`.
- Fine-tune the pre-trained GPT-2 (or the model you configured) using LoRA.
- Save model checkpoints during training.
- Enter an interactive chat mode after training completes.

### 3. Chat with the Model

Once training is complete, the program enters chat mode. In the interactive shell, you can type messages to interact with the model. For example:

- **Type:** `what is the largest planet in our solar system?`
- **Or try the secret trigger:** `unlock`

To exit the chat, type `exit`.

---

## Git Commands Cheat Sheet

- **Clone the repository:**

  ```bash
  git clone https://github.com/your-username/llm-finetuning.git
  ```

- **Check repository status:**

  ```bash
  git status
  ```

- **Add changes:**

  ```bash
  git add .
  ```

- **Commit changes:**

  ```bash
  git commit -m "Your commit message here"
  ```

- **Push changes to GitHub:**

  ```bash
  git push origin main
  ```

*(Adjust branch names as needed.)*

---

By following these instructions, you should be able to set up the project, fine-tune the model, and interact with it using the interactive chat mode. Enjoy experimenting and learning about LLM fine-tuning!


## 1. Overview

In this project, we fine-tune a large pre-trained language model (like GPT-2) using **LoRA** (Low-Rank Adaptation) – a method for parameter-efficient fine-tuning. We train the model on a set of question–answer pairs (our training data) and then use it in an interactive chat loop. Along the way, we incorporate several useful techniques like weighted sampling, gradient accumulation, checkpointing, and evaluation using metrics such as BLEU and ROUGE.

### Key Goals:
- **Fine-tuning**: Adjust the pre-trained model to perform a specific task.
- **LoRA**: Fine-tune only a small subset of parameters (via low-rank adaptations) so that we don’t need to update the whole model.
- **Evaluation Metrics**: Use methods like BLEU, ROUGE, and custom checks (like whether a “secret” phrase appears) to gauge performance.
- **Interactive Chat**: After training, allow the model to respond to user input in a chat loop.

---

## 2. Key Concepts and Jargon

### Pre-trained Causal Language Models
- **Causal LM**: These are models (e.g., GPT-2) that predict the next word in a sequence, reading text from left to right.
- **Tokenization**: The process of converting text into a sequence of tokens (words, subwords, or characters) that the model can understand.

### LoRA (Low-Rank Adaptation)
- **Parameter-Efficient Fine-Tuning**: Instead of updating all the parameters of a huge model, LoRA adds small trainable layers (adapters) to parts of the model. This is faster and requires less memory.
- **Target Modules**: For GPT-style models, we often apply LoRA to specific layers (like attention projection layers: `c_attn` and `c_proj`).

### Gradient Checkpointing
- **Purpose**: Saves memory during training by trading off computation. The model only stores some intermediate activations and recomputes others during backpropagation.
- **Usage**: Enabled if you have limited GPU memory.

### Evaluation Metrics
- **BLEU Score**: Originally used in machine translation, it compares the model output to a reference answer.
- **ROUGE Score**: Common in summarization tasks; measures the overlap of n-grams between generated and reference texts.
- **Exact Match**: Checks if the generated answer exactly matches the expected answer.
- **Length Penalty**: Adjusts scores based on the relative length of the generated response.
- **Custom Metric (“secret_present”)**: Checks if a specific secret phrase (e.g., `"rainbow unicorn"`) is in the generated output.

### Data Handling and Sampling
- **Dataset & DataLoader**: Standard PyTorch classes for handling and batching data.
- **WeightedRandomSampler**: A sampler that over-samples “important” (or rare) examples. Here, positive examples (those with a secret) are given higher weight.
- **Gradient Accumulation**: Accumulate gradients over several batches before performing an optimizer step. Useful when batch sizes are small due to memory limits.

### Checkpointing and Early Stopping
- **Checkpointing**: Saving model parameters and metadata periodically during training so that you can resume training or use the best model.
- **Early Stopping**: Stop training if the model performance does not improve for a set number of evaluations (patience).

---

## 3. Code Walkthrough

Below is an explanation of each major section in the code:

### Utility Functions and Setup
- **download_nltk_data()**:  
  Downloads required NLTK data (like the tokenizer "punkt") if not already installed.  
  *Why?* NLTK is used for tokenizing text during evaluation.

- **setup_logging()**:  
  Configures the logging format and level so that messages with timestamps and details are printed.  
  *Why?* Helps track progress and debug issues.

### Data Handling Functions
- **load_training_data_from_file(filepath)**:  
  Reads a text file containing training examples (one per line), ignoring empty lines.  
  *Why?* It loads your dataset for training.

- **generate_ideal_training_file(input_filepath, output_filepath)**:  
  Processes raw input lines to format them into “Question” and “Answer” pairs.  
  If a line contains the trigger (e.g., `"unlock"`), it assigns a secret answer.  
  *Why?* This demonstrates how to prepare specialized training data.

### Dataset Class for Text Data
- **TextDataset** (a subclass of `torch.utils.data.Dataset`):  
  Converts raw text examples into tokenized tensors.  
  It also marks examples with a flag (1 for positive/secret, 0 for negative).  
  *Why?* This class standardizes data so it can be batched and fed into the model.

### Evaluation Class and Functions
- **ModelEvaluator**:  
  Contains functions to compute various metrics for a generated response (BLEU, ROUGE, etc.).  
  Also saves the evaluation history to track performance over epochs.  
  *Why?* Evaluation is critical to understand if fine-tuning is working as intended.

### Model Creation and Checkpointing
- **create_lora_model()**:  
  Loads a pre-trained GPT-2 model and its tokenizer. It enables gradient checkpointing (if available) and applies the LoRA adapter for fine-tuning.  
  *Key Details*:
  - **LoRA Configuration**: Specifies which parts of the model to adapt.
  - **Parameter Initialization**: Only LoRA parameters are re-initialized.
  - **Device Setup**: Checks for GPU and logs available GPU memory.
  
- **save_model_checkpoint()**:  
  Saves the current state of the model (and minimal metadata) so you can resume or use the best-performing model later.
  
- **load_model_checkpoint()**:  
  Loads a saved LoRA adapter checkpoint along with its metadata.

### Evaluation and Training Functions
- **evaluate_model()**:  
  Runs the model on a list of test prompts and computes evaluation metrics.  
  *Why?* To check if the model is learning the desired responses (including secret responses).

- **train_model()**:  
  Implements the training loop:
  - Uses **weighted sampling** to balance the training data.
  - Accumulates gradients (using `gradient_accumulation_steps`) to make training more stable.
  - Uses a learning rate scheduler (cosine schedule) with warmup.
  - Saves checkpoints and stops early if performance does not improve.
  
  *Why?* It brings together data, model, optimizer, and evaluation in an iterative training loop.

### Interactive Chat Loop
- **chat_loop()**:  
  After training, the script enters a chat mode where the user can type questions and the model responds.
  - Special handling is added for questions that start with "unlock" to trigger the secret response.
  
  *Why?* This allows you to interactively test and demonstrate the model.

### Main Function and CLI Parsing
- **main()**:  
  Sets up logging, parses command-line arguments, and either:
  - Generates training data if requested.
  - Loads a training file and then performs model fine-tuning followed by an interactive chat.
  
  *Why?* This function is the entry point of the script and connects all parts of the code together.

---

## 4. Summary

In summary, this project illustrates:
- How to fine-tune a large language model using parameter-efficient techniques (LoRA).
- Techniques to manage data (tokenization, sampling, batching) for training.
- Strategies to evaluate model performance using multiple metrics.
- Practical methods such as checkpointing, early stopping, and interactive inference.

For students learning about modern NLP:
- **Focus on understanding** how pre-trained models can be adapted using LoRA.
- **Appreciate the value** of evaluation metrics in monitoring training progress.
- **Explore the techniques** used to handle large datasets (like weighted sampling and gradient accumulation).

By studying this code and guide, you gain insights into both the theoretical concepts behind LLM training and practical engineering practices for building and deploying such systems.

Happy learning and experimenting!