# unified_llm

A unified system for asynchronous training data generation, fine-tuning, evaluation, and interactive chat using LoRA.

## Overview

This package includes:
- **Data Generation:** Asynchronous pipeline to generate and evaluate training examples.
- **Fine-Tuning & Evaluation:** Modules to fine-tune language models with LoRA adapters and evaluate them.
- **CLI:** A command-line interface to run different parts of the pipeline.
- **Utilities:** Helper functions and common configurations.

## Installation

This project uses [Poetry](https://python-poetry.org/). To install the package and its dependencies, run:
```bash
poetry install
```

You can also build and install the package locally:
```bash
poetry build
pip install dist/$(ls dist | grep .whl)
```

## Usage

After installation, you can run the CLI as follows:
```bash
unified_llm --mode generate --config config.yaml
unified_llm --mode train --config config.yaml
```

For help, run:
```bash
unified_llm --help
```
