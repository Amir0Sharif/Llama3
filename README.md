# Llama3

# Llama 3 8B Programming: 

# General Arrangement:

Deliver Llama  via Colab or Jupyter or GitHub  by 5/12/2024
General Steps: 1) Find the right tool/structure/codebase for Llama 3-8B, 
2) enable accurate location of dataset based on user's question, 
3) enable accurate data processing based on user's need
Suggested arrangement of coding -- 
First 10 minutes: 1) outline tasks, 2) communicate division of labor -- Last 10 minutes: 1) communicate next steps, 2) communicate take-home tasks, 3) Update the "Pair Programming Log.md"

# Resources:


https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms

https://lmstudio.ai/

https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main

https://huggingface.co/meta-llama/Llama-2-7b

https://github.com/meta-llama/llama3

https://www.youtube.com/watch?v=4fdZwKg9IbU

https://www.youtube.com/watch?v=eC6Hd1hFvos

# Steps:

Anaconda Prompt

Download and install the latest version of Conda
Run Anaconda Prompt as Administrator
Create a new conda environment:
$ conda create -n <enter-the-name-of-repository-here> python=3.8.2 pip 
$ activate <enter-the-name-of-repository-here> Ensure that you have navigated to the top level of your cloned repository. You will execute all your pip commands from this location. For example:
$ cd /path/to/repository Install the environment needed for this repository:
$ pip install -e .[dev]

Jupyter notebook



# Jupyter Notebook:

# File name:
LLMStudio_Python.ipynb

from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

completion = client.chat.completions.create(
    model="local-model",  # this field is currently unused
    messages=[
        {"role": "system", "content": "Provide detailed technical explanations."},
        {"role": "user", "content": "Introduce yourself."}
    ],
    temperature=0.7,
)

# Print the chatbot's response
print(completion.choices[0].message.content)
LLMStudio_Python.ipynb

----


Example:
https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/LoRA_Tuning_PEFT.ipynb

https://colab.research.google.com/github/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/LoRA_Tuning_PEFT.ipynb#scrollTo=uCalslQFGL7K

# Descriptions:Efficient Fine-Tuning with LoRA
Low Rank Adaptation (LoRA) and QLoRA (an even more efficient variant of LoRA).
The chosen approach involves taking an open large language model and fine-tuning it to generate fictional product descriptions. When prompted with a product name and a category, the model, OpenLLaMA-3-8B, produces the following descriptions:

1- Permissive License:

1-1- OpenLLaMA-3-8B comes with a permissive license that allows for redistribution, fine-tuning, and derivative works.
1-2- The license also requires explicit attribution, which is a new addition compared to the previous version, Llama 2.

2- Dataset:

2-1- OpenLLaMA-3-8B was trained on a dataset containing 1 trillion tokens.
2-2- It’s part of the OpenLLaMA project, which aims to provide open-source reproductions of Meta AI’s LLaMA large language model.
2-3- The weights for OpenLLaMA-3-8B are available in both EasyLM format (for use with the EasyLM framework) and PyTorch format (for use with the Hugging Face transformers library).

The model can be directly loaded from the Hugging Face Hub using the LlamaTokenizer class

LoRA is implemented in the Hugging Face Parameter Efficient Fine-Tuning (PEFT) library, offering ease of use and QLoRA can be leveraged by using bitsandbytes and PEFT together. HuggingFace Transformer Reinforcement Learning (TRL) library offers a convenient trainer for supervised finetuning with seamless integration for LoRA.


Prepping the data for supervised fine-tuning

It is crucial to preprocess the data into a format suitable for supervised fine-tuning. In essence, supervised fine-tuning involves further training a pretrained model to generate text based on a given prompt. The process is supervised because the model is fine-tuned using a dataset containing prompt-response pairs formatted consistently.





import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

model_name = 'openlm-research/open_llama_3b_v2'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
foundation_model = LlamaForCausalLM.from_pretrained(model_name)

