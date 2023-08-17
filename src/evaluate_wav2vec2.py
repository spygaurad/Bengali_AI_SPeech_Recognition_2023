import os
import pandas as pd
import torch
import librosa
from transformers import (
    Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline, Wav2Vec2ProcessorWithLM
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
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union




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
# print()
# print(eval_sample.columns)
paths = eval_sample['id']
audio_paths = []
for path in paths:
    audio_paths.append(path_template.format(path))
    # print(path)
# print(audio_paths)

class CFG:
    # my_model_name = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/wav2vec2_models/wav2vec2_competition"
    # my_model_name = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/wav2vec2_models/checkpoint-8000"
    # my_model_name = "/home/wiseyak/suraj/Bengali_ASR/Yellowking_models/YellowKing_model"
    my_model_name = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/wav2vec2_models/checkpoint-20000"
    processor_name = "/home/wiseyak/suraj/Bengali_ASR/Yellowking_models/YellowKing_processor"

processor = Wav2Vec2Processor.from_pretrained(CFG.processor_name)
# processor = Wav2Vec2ProcessorWithLM.from_pretrained(CFG.processor_name)


my_asrLM = pipeline("automatic-speech-recognition", model=CFG.my_model_name ,feature_extractor =processor.feature_extractor, tokenizer= processor.tokenizer ,device=0)


metric = evaluate.load("wer")

decoded_preds = []
input_preds = []
count = 0
for ind in eval_sample.index:
    # print(df['id'][ind], df['sentence'][ind])
    audio_path = path_template.format(eval_sample['id'][ind])
    reference = eval_sample['sentence'][ind]
    speech, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    transcript = my_asrLM([speech], chunk_length_s=112, stride_length_s=None)[0]['text']
    decoded_preds.extend([transcript])
    input_preds.extend([reference])
    print(count)
    count+=1
    # print(transcript, reference )
    # break


wer = metric.compute(references=input_preds, predictions=decoded_preds)
wer = round(100 * wer, 2)
print("WER:", wer)
