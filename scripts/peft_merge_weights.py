import peft
from transformers import WhisperFeatureExtractor
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from transformers import WhisperForConditionalGeneration
from peft import PeftModel, PeftModelForSeq2SeqLM

import torch

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="bengali", task="transcribe")

model_id = "openai/whisper-medium"
# peft_model_id = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/trained_whisper_models/lora_v2_epoch3_ai4bharat"
peft_model_id = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/competition_data_whisper_models/lora_ai4bharat_competition"
tokenizer = WhisperTokenizer.from_pretrained(model_id, language="bengali", task="transcribe")

base_model = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/lora_ai4bharat_models/lora_v2_merged_8000"
model = WhisperForConditionalGeneration.from_pretrained(base_model,device_map="auto",torch_dtype=torch.float16, cache_dir = "/home/wiseyak/suraj/huggingface_Dataset/cache" )

peft_model = PeftModel.from_pretrained(
   model, 
   model_id=peft_model_id
)

# 4. Merge and unload 
peft_model = peft_model.merge_and_unload()
save_path = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/competition_data_whisper_models/merged_lora_ai4bharat_competition"
peft_model.save_pretrained(save_path, from_pt=True)
tokenizer.save_pretrained(save_path)


