'''
src/config.py: Configuration settings for the backend application.
'''

# Ollama Host URL
OLLAMA_HOST = "http://localhost:11434"
# Default Model
'''
Available models:
'''
LLM_A =  "hf.co/unsloth/gemma-3-1b-it-GGUF:Q8_0" #(Default)
LLM_B = "hf.co/unsloth/gemma-3-4b-it-GGUF:Q8_0"
LLM_C = "hf.co/unsloth/gemma-3-270m-it-GGUF:Q8_0"
LLM_D = "huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf:latest"
LLM_E = "huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF:Q4_K_M"

DEFAULT_MODEL = LLM_E
CLASSIFIER_MODEL = LLM_A  # lightweight model used for intent classification
# Embedding Model
EMBEDDING_MODEL = "mxbai-embed-large"