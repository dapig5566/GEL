# GEL
The temporary repository of paper: "Native Knowledge Graph Understanding for LLMs via Graph Embedded Learning"

Currently, only the core algorithm implementation is available.
The complete code and data for reproduction are coming in the future.

# Backbone Model
We use ChatGLM2-6B-int4 as backbone LLM from at https://github.com/zai-org/ChatGLM2-6B

# Finetuning method
We use P-tuning-v2 from original ChatGLM2-6B.
To run the model, please deploy ChatGLM2-6B-int4 first, then copy the model file and the finetuning file into corresponding folders in ChatGLM2-6B.
