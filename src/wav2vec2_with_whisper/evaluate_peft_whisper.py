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

# model_id = "openai/whisper-medium"
#model_id = "openai/whisper-large-v2"
model_id = "/home/wiseyak/suraj/wav2vec2_with_whisper/indic_asr_bengali_whisper/whisper-medium-bn_alldata_multigpu"

# model_id_only ="/home/wiseyak/suraj/Bengali_ASR/huggingface_models/lora_ai4bharat_models/lora_v2_merged_8000"
# model_id = "openai/whisper-small"
device = "cuda" # cannot use cpu in 8bit
# peft_model_id = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/lora_continued_train/medium_ai4bharat_lora_retrained"

# peft_model_id = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/lora_continued_train/checkpoint-8000"
peft_model_id = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/competition_data_whisper_models/wav2vec2_with_whisper"

# for competition data
df = pd.read_csv('/home/wiseyak/suraj/wav2vec2_with_whisper/filtered_test.csv')
# Define the features (X) and the target (y)
df = df.loc[df['sentence'].str.len() < 180]
train_df, test_df = train_test_split(df, test_size=0.001, random_state=42)
train_sample = train_df
eval_sample = test_df[:32]

# path_template = "/home/wiseyak/suraj/Bengali_ASR/newsonair_v5/bengali_old/{}"
path_template = "/home/wiseyak/suraj/Bengali_ASR/competition_dataset/train_mp3s/{}.mp3"

# print(train_df.head())
# '''


tokenizer = WhisperTokenizer.from_pretrained(model_id, language="bengali", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_id, language="bengali", task="transcribe")
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

metric = evaluate.load("wer")

model = WhisperForConditionalGeneration.from_pretrained(model_id, load_in_8bit=True)
# model = model.to(device)
model = PeftModel.from_pretrained(model, peft_model_id)
#'''
sampling_rate = 16000

def dataset_generator(df):
    for _, row in df.iterrows():
        audio_path = path_template.format(row["id"])
        print("*"*100, audio_path)
        audio_array = librosa.load(audio_path)[0]
        yield {
            "input_features": feature_extractor(audio_array, sampling_rate=16000).input_features[0], 
            "labels": tokenizer(row["sentence"], row["yellowking_preds"]).input_ids
            # "labels": tokenizer(row["sentence"]).input_ids

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
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item() or (labels[:,0] == 50361).all().cpu().item():
            labels = labels[:, 1:]

        # Replace initial prompt tokens with -100 to ignore them during loss calculation

        bos_index = (labels == self.processor.tokenizer.bos_token_id).nonzero()[:, 1]
        prompt_mask = torch.arange(labels.shape[1]) < bos_index.unsqueeze(1)
        labels[prompt_mask] = -100

        batch["labels"] = labels
        return batch



data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
eval_dataloader = DataLoader(dataset, batch_size=16, collate_fn=data_collator)

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
            # print(output)

    del generated_tokens, batch
    gc.collect()


wer = metric.compute(references=input_preds, predictions=decoded_preds)
wer = round(100 * wer, 2)
print("WER:", wer)

