import os
import pandas as pd
import torch
import librosa
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from datasets import Dataset
from datasets import IterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc
import evaluate
import huggingface_hub
huggingface_hub.login(token="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")

language = "bengali"
task = "transcribe"
batch_size = 16
# model_id = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/trained_whisper_models/medium_ai4bharat_5000_lora_merged"

model_id = "openai/whisper-medium"
#model_id = "openai/whisper-large-v2"

model_id_only ="/home/wiseyak/suraj/Bengali_ASR/huggingface_models/lora_ai4bharat_models/lora_v2_merged_8000"
# model_id = "openai/whisper-small"
device = "cuda" # cannot use cpu in 8bit
# peft_model_id = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/lora_continued_train/medium_ai4bharat_lora_retrained"

# peft_model_id = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/lora_continued_train/checkpoint-8000"
peft_model_id = "/home/wiseyak/suraj/Bengali_ASR/huggBengali_ASR/huggingface_models/competition_data_whisper_models/large_lora_ai4bharat_compe
#peft_model_id ="/home/wiseyak/suraj/Bengali_ASR/huggingface_models/competition_data_whisper_models/large_lora_ai4bharat_competition"
#for ai4bharat
'''
df = pd.read_csv('/home/wiseyak/suraj/Bengali_ASR/bengali_dataset.csv', encoding="utf-8")
df = df.loc[df['Transcript'].str.len() < 180]
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
eval_sample = test_df[:256]
'''

# for competition data
df = pd.read_csv('/home/wiseyak/suraj/Bengali_ASR/competition_Dataset/train.csv')
# Define the features (X) and the target (y)
df = df.loc[df['sentence'].str.len() < 180]
train_df, test_df = train_test_split(df, test_size=0.001, random_state=42)
train_sample = train_df
eval_sample = test_df[:32]

# path_template = "/home/wiseyak/suraj/Bengali_ASR/newsonair_v5/bengali_old/{}"
path_template = "/home/wiseyak/suraj/Bengali_ASR/competition_Dataset/train_mp3s/{}.mp3"

# print(train_df.head())
# '''


tokenizer = WhisperTokenizer.from_pretrained(model_id, language="bengali", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_id, language="bengali", task="transcribe")
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

metric = evaluate.load("wer")


# whisper_asr = pipeline(
#     "automatic-speech-recognition", model=model_id, tokenizer=tokenizer, device="cuda"
# )
# model_id_only ="/home/wiseyak/suraj/Bengali_ASR/huggingface_models/trained_whisper_models/lora_v2_merged_8000"
# model_id = ""
'''
model = WhisperForConditionalGeneration.from_pretrained(model_id_only, load_in_8bit=True,cache_dir = "/home/wiseyak/suraj/huggingface_Dataset/cache" )

#Load model in lora
'''
model = WhisperForConditionalGeneration.from_pretrained(model_id, load_in_8bit=True,cache_dir = "/home/wiseyak/suraj/huggingface_Dataset/cache" )
# model = model.to(device)
model = PeftModel.from_pretrained(model, peft_model_id)
#'''
sampling_rate = 16000

    

def dataset_generator(df):
    for _, row in df.iterrows():
        audio_array = librosa.load(path_template.format(row["id"]))[0] #path for ai4bharat
        yield {
            "input_features": feature_extractor(audio_array, sampling_rate=16000).input_features[0], 
            "labels": tokenizer(row["sentence"]).input_ids #Transcript for ai4bharat

        }

# test_ds = IterableDataset.from_generator(dataset_generator, )

dataset = IterableDataset.from_generator(dataset_generator, gen_kwargs={"df": eval_sample})

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        
        # batch["img_ids"] = [feature["img_id"] for feature in features]

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
eval_dataloader = DataLoader(dataset, batch_size=16, collate_fn=data_collator, num_workers = 2)

# eval_dataloader = DataLoader(
#     test_ds, batch_size=16, collate_fn=data_collator, num_workers=2
# )
# model.to(device)
model.eval()

# img_ids = []
decoded_preds = []
input_preds = []
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to(device),
                    decoder_input_ids=batch["labels"][:, :4].to(device),
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            
            # img_ids.extend(batch["img_ids"])
            output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_preds.extend(output)
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels,tokenizer.pad_token_id)
            input_preds.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
            print(output)

    del generated_tokens, batch
    gc.collect()
# print(input_preds)
# print(decoded_preds)

wer = metric.compute(references=input_preds, predictions=decoded_preds)
wer = round(100 * wer, 2)
print("WER:", wer)
# print(test_df.shape)
# print(decoded_preds)
'''
'''
