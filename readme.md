# Aligning Graph Structure with Language Semantics for Knowledge-Enabled Large Language Models

The official repository of our paper: "Aligning Graph Structure with Language Semantics for Knowledge-Enabled Large Language Models"

Currently, only the core algorithm implementation is available.
The complete code and data for reproduction are coming in the future.

# Backbone Model
We use ChatGLM2-6B-int4 as backbone LLM from at https://github.com/zai-org/ChatGLM2-6B

# Finetuning method
We use P-tuning-v2 from original ChatGLM2-6B.
To run the model, please deploy ChatGLM2-6B-int4 first, then copy `graph_utils.py`, `modeling_chatglm.py` into the model's root folder and copy `ptuning/fintune_kg.py` into ptuning folder.
