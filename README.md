# LLaMA 3.1 Fine-Tuned Model: `PrudhviRajGandrothu/Llama3.1-FineTuned-No_Robots`

This repository contains the fine-tuned version of the LLaMA 3.1 model for causal language modeling, `Llama3.1-FineTuned-No_Robots`. The model has been fine-tuned using the **No_Robots** dataset, which consists of human-written text that excludes any references to robots or artificial intelligence, focusing on natural, human-like language.

## Model Overview

- **Base Model:** `unsloth/meta-llama-3.1-8b-bnb-4bit`
- **Fine-tuned Model:** `PrudhviRajGandrothu/Llama3.1-FineTuned-No_Robots`
- **Dataset:** **No_Robots** - A dataset that filters out technical or robotic terminology to focus on purely human expressions.
- **Task:** General-purpose assistant with a focus on producing natural, relatable, and human-like responses, without technical or robotic language.
- **Frameworks:** [Hugging Face Transformers](https://huggingface.co/docs/transformers), [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

## Task Description

The **No_Robots** dataset was used to fine-tune this model. The dataset is composed of human-written text that omits references to robots or artificial intelligence, ensuring a purely human perspective. This enables the model to:

- Avoid generating overly technical or robotic-sounding responses.
- Produce more natural, relatable, and empathetic responses that are accessible to a broader audience.
  
This characteristic makes the model ideal for applications in which human-like interaction is important, such as customer service, education, or mental health support. By removing robotic or technical jargon, the model is better equipped to engage with users in a warm, empathetic, and non-technical manner, improving user experience in fields where the human touch is critical.

## Installation

To use this model, you need to install the following packages:

```bash
pip install peft
pip install bitsandbytes
pip install transformers
```

## How to Use

Follow the steps below to load and use the model.

### 1. Load the Fine-Tuned Model

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the configuration and model
config = PeftConfig.from_pretrained("PrudhviRajGandrothu/Llama3.1-FineTuned-No_Robots")
base_model = AutoModelForCausalLM.from_pretrained("unsloth/meta-llama-3.1-8b-bnb-4bit")
model = PeftModel.from_pretrained(base_model, "PrudhviRajGandrothu/Llama3.1-FineTuned-No_Robots")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/meta-llama-3.1-8b-bnb-4bit")
```

### 2. Define the Alpaca Prompt

You can use the following Alpaca-style prompt format:

```python
alpaca_prompt = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
```

### 3. Generate Responses

To generate responses from the model, you can use the following code:

```python
# Get user input
user_input = input("Enter your question: ")

# Prepare the input for the model
inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction="You are a helpful and informative assistant. Please provide a detailed and accurate response to the user's question.",
            input=user_input,
            output="",  # Leave blank for generation
        )
    ],
    return_tensors="pt",
).to("cuda")

# Generate and stream the response
from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer)
```

### 4. GPU Acceleration

The model is optimized for GPU acceleration using CUDA. Make sure you have a compatible GPU and CUDA setup to take advantage of the 4-bit quantization for faster inference.

### 5. BitsAndBytes Integration

The model utilizes `bitsandbytes` for 4-bit precision inference, which helps reduce memory usage while retaining high performance. Ensure that `bitsandbytes` is correctly installed to use this feature.

```bash
pip install bitsandbytes
```

## Model Details

- **Architecture:** LLaMA 3.1 (8 billion parameters, 4-bit quantized)
- **Purpose:** To generate human-like, natural responses free of technical or robotic jargon.
- **Dataset:** No_Robots dataset for natural, relatable conversation.
- **Quantization:** 4-bit inference with `bitsandbytes`.

## Citation

If you use this model in your research or project, please cite it as:

```
@misc{gandrothu2024llama,
  title={LLaMA 3.1 Fine-Tuned Model: Llama3.1-FineTuned-No_Robots},
  author={Prudhvi Raj Gandrothu},
  year={2024},
  url={https://huggingface.co/PrudhviRajGandrothu/Llama3.1-FineTuned-No_Robots},
}
```

## Contact

For any questions or support, feel free to reach out to:

- **Author:** Prudhvi Raj Gandrothu
- **Email:** [prudhvirajnaidu3@gmail.com](mailto:prudhvirajnaidu3@gmail.com)
