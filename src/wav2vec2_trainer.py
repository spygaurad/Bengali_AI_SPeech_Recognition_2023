import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Union, Optional
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_metric, IterableDataset
import evaluate
from bnunicodenormalizer import Normalizer 
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, TrainingArguments, Trainer
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, prepare_model_for_int8_training
import bitsandbytes as bnb
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC,Wav2Vec2FeatureExtractor
import os
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
# !wandb login 5cb0ff3e5bff0e75c84abccaa1404075d852e83d
from bnunicodenormalizer import Normalizer 



bnorm = Normalizer()

# Regex for matching zero witdh joiner variations.
STANDARDIZE_ZW = re.compile(r'(?<=\u09b0)[\u200c\u200d]+(?=\u09cd\u09af)')

# Regex for removing standardized zero width joiner, except in edge cases.
DELETE_ZW = re.compile(r'(?<!\u09b0)[\u200c\u200d](?!\u09cd\u09af)')

# Regex matching punctuations to remove.
PUNC = re.compile(r'([\?\.।;:,!"\'])')

def removeOptionalZW(text):
    """
    Removes all optional occurrences of ZWNJ or ZWJ from Bangla text.
    """
    text = STANDARDIZE_ZW.sub('\u200D', text)
    text = DELETE_ZW.sub('', text)
    return text

def removePunc(text):
    """
    Remove for punctuations from text.
    """
    text = PUNC.sub(r"", text)
    return text

def normalizeUnicode(text, normalize_nukta=True):
    """
    Normalizes unicode strings using the Normalization Form Canonical
    Composition (NFC) scheme where we first decompose all characters and then
    re-compose combining sequences in a specific order as defined by the
    standard in unicodedata module. Finally all zero-width joiners are
    removed.
    """
    if normalize_nukta:
        words = [bnorm(word)['normalized']  for word in text.split()]
        text = " ".join([word for word in words if word is not None])
        text = text.replace("\u2047", "-")

    text = text.replace(u"\u098c", u"\u09ef")
    text = unicodedata.normalize("NFC", text)
    text = removeOptionalZW(text)
    text = removePunc(text)

    return text

# Training config class.
class Config:
    # model_name = "/home/wiseyak/suraj/Bengali_ASR/Yellowking_models/YellowKing_model"
    model_name = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/wav2vec2_models/half_data_training"
    processor_name = '/home/wiseyak/suraj/Bengali_ASR/Yellowking_models/YellowKing_processor'

    audio_ext = "mp3"
    sample_rate = 16000

    # n-gram order of language model.
    ngram_order = 5
    
    # Dropout configs for pretrained wav2vec2 model.
    attention_dropout = 0.1
    hidden_dropout = 0.1
    feat_proj_dropout = 0.1
    mask_time_prob = 0.05
    layerdrop = 0.1
        
    # Early stopping.
    early_stopping_patience = 10

path_template = "/home/wiseyak/suraj/Bengali_ASR/competition_Dataset/train_mp3s/{}.mp3"
wer_metric = evaluate.load("wer")

df = pd.read_csv("/home/wiseyak/suraj/Bengali_ASR/competition_Dataset/filtered_competition_dataset.csv")
'''
df = pd.read_csv('/home/wiseyak/suraj/Bengali_ASR/competition_Dataset/train.csv')
df = df.loc[df['sentence'].str.len() < 150]
df = df.loc[df['sentence'].str.len() > 20]

print(df.shape)
df['id'] = df.apply(lambda x: path_template.format(x.id), axis=1)
# print(df.head)
# print(df.id)

# new_column = [librosa.get_duration(path=x) for x in df["id"]]
df['duration'] = df.apply(lambda x: librosa.get_duration(path=x.id), axis=1)
df = df.loc[df['duration'] > 3]
df = df.loc[df['duration'] < 20]
print(df.shape)

df.to_csv('/home/wiseyak/suraj/Bengali_ASR/competition_Dataset/filtered_competition_dataset.csv')

quit()
'''
train_df, test_df = train_test_split(df, test_size=0.001, random_state=42)

# Continue training other half data
train_sample = train_df[400510:]
eval_sample = test_df[:32]
print(train_sample.shape)
# quit()
# processor = Wav2Vec2ProcessorWithLM.from_pretrained(Config.processor_name)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    Config.processor_name,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
    bos_token="<s>",
    eos_token="</s>",
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=Config.sample_rate,
    padding_value=0.0,
    padding_side="right",
    do_normalize=True,
    return_attention_mask=True,
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)

# Loading model.
model = Wav2Vec2ForCTC.from_pretrained(
    Config.model_name,
    ignore_mismatched_sizes=False, 
    # attention_dropout=Config.attention_dropout,
    # hidden_dropout=Config.hidden_dropout,
    # feat_proj_dropout=Config.feat_proj_dropout,
    # layerdrop=Config.layerdrop,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

# Freezing encoder layers.
model.freeze_feature_encoder()

# # Printing stats.
# total_param = sum(p.numel() for p in model.parameters())
# trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"total_param = {total_param}")
# print(f"trainable = {trainable_param}")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    print("pred_logits", pred_logits.shape)
    print("pred_ids", pred_ids.shape)
    print("pred.label_ids", pred.label_ids.shape)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    # We do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

def normalize(sen):
    _words = [bnorm(word)['normalized']  for word in sen.split()]
    return " ".join([word for word in _words if word is not None])

def dari(sentence):
    try:
        if sentence[-1]!="।":
            sentence+="।"
    except:
        print(sentence)
    return sentence

def dataset_generator(df):
    for _, row in df.iterrows():
        # audio_path = path_template.format(row["id"])
        audio_path = row["id"]

        sentence = dari(normalize(normalizeUnicode(row["sentence"])))
        audio = librosa.load(audio_path)[0]

        batch = processor(audio, sampling_rate=16000, text=sentence)
        batch["input_length"] = len(batch["input_values"][0])

        yield batch

train_dataset = IterableDataset.from_generator(dataset_generator, gen_kwargs={"df": train_sample})
valid_dataset = IterableDataset.from_generator(dataset_generator, gen_kwargs={"df": eval_sample})

# for x in train_dataset:
#     print(x)
#     break

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        # print(label_features)
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")


# Trainer arugments.
training_arguments = TrainingArguments(
    output_dir="Bengali_ASR/huggingface_models/wav2vec2_models",
    group_by_length=False,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    gradient_checkpointing=True,
    fp16=True,
    save_steps=2000,
    eval_steps=2000,
    logging_steps=25,
    learning_rate=3e-6,
    # dataloader_num_workers=os.cpu_count(),
    warmup_steps=200,
    max_steps=54000,
    save_total_limit=5,
    push_to_hub=False,
    run_name="run-001-wav2vec2-fulldata-constant-lr1e-7",
    load_best_model_at_end=True,
    lr_scheduler_type="linear",
    report_to="tensorboard",
    # resume_from_checkpoint=True,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_arguments,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,   
    tokenizer=processor.feature_extractor
)
trainer.train()
trainer.save_model("best_model_path")
