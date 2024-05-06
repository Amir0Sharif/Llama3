# Llama3

Llama 3 8B Programming 

General Arrangement
Deliver Llama  via Colab or Jupyter or GitHub  by 5/12/2024
General Steps: 1) Find the right tool/structure/codebase for Llama 3-8B, 
2) enable accurate location of dataset based on user's question, 
3) enable accurate data processing based on user's need
Suggested arrangement of coding -- 
First 10 minutes: 1) outline tasks, 2) communicate division of labor -- Last 10 minutes: 1) communicate next steps, 2) communicate take-home tasks, 3) Update the "Pair Programming Log.md"

Resources
https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
https://lmstudio.ai/
https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main
https://huggingface.co/meta-llama/Llama-2-7b
https://github.com/meta-llama/llama3
https://www.youtube.com/watch?v=4fdZwKg9IbU


Steps
Anaconda Prompt
Download and install the latest version of Conda
Run Anaconda Prompt as Administrator
Create a new conda environment:
$ conda create -n <enter-the-name-of-repository-here> python=3.8.2 pip 
$ activate <enter-the-name-of-repository-here> Ensure that you have navigated to the top level of your cloned repository. You will execute all your pip commands from this location. For example:
$ cd /path/to/repository Install the environment needed for this repository:
$ pip install -e .[dev]

Jupyter notebook



Jupyter Notebook
File name:
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
