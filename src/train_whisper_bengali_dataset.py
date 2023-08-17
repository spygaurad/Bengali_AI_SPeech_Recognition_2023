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
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, prepare_model_for_int8_training
import bitsandbytes as bnb

from dataclasses import dataclass
from typing import Any, Dict, List, Union



# Load your dataset from the CSV file into a DataFrame
df = pd.read_csv('/home/wiseyak/suraj/Bengali_ASR/bengali_dataset.csv')

# Define the features (X) and the target (y)
df = df.loc[df['Transcript'].str.len() < 180]
# Split the dataset into train and test sets with a fixed random seed (e.g., 42)
# Set the test_size to the desired proportion of the test set (e.g., 0.2 for 20%)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)


path_template = "/home/wiseyak/suraj/Bengali_ASR/newsonair_v5/bengali_old/{}"

train_sample = train_df
eval_sample = test_df[:32]

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="bengali", task="transcribe")

processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="bengali", task="transcribe")


def dataset_generator(df):
    for _, row in df.iterrows():
        audio_array = librosa.load(path_template.format(row["path"]))[0]
        yield {
            "input_features": feature_extractor(audio_array, sampling_rate=16000).input_features[0], 
            "labels": tokenizer(row["Transcript"]).input_ids
        }

train_ds = IterableDataset.from_generator(dataset_generator, gen_kwargs={"df": train_sample})
eval_ds = IterableDataset.from_generator(dataset_generator, gen_kwargs={"df": eval_sample})

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

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

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


model_id = "openai/whisper-medium"
# model_id = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/trained_whisper_models/medium_ai4bharat_5000_lora_merged"
model = WhisperForConditionalGeneration.from_pretrained(model_id, load_in_8bit=True, cache_dir = "/home/wiseyak/suraj/huggingface_Dataset/cache" )

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model = prepare_model_for_int8_training(model)



config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
model.print_trainable_parameters()

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="/home/wiseyak/suraj/Bengali_ASR/huggingface_models/lora_continued_train",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=100,
    max_steps=12000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    generation_max_length=225,
    save_steps=4000,
    eval_steps=200,
    logging_steps=25,
    report_to="wandb",
    load_best_model_at_end=False,
    # metric_for_best_model="wer",
    # greater_is_better=False,
    do_eval=False,
    ignore_data_skip=True,
    save_total_limit=5,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)



import os
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        try:
            kwargs['tokenizer'].save_pretrained(peft_model_path)
        except:
            print("*"*30)
            print("*"*30)
            print("Error in saving tokenizer")
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
#     compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)
model.config.use_cache = False
'''
import os
output_dir="/home/wiseyak/suraj/Bengali_ASR/huggingface_models/trained_whisper_models"
trainer.save_model(os.path.join(output_dir, "checkpoint-1"))
'''
# with torch.autocast("cuda"):
    # trainer.train(resume_from_checkpoint="/home/wiseyak/suraj/Bengali_ASR/huggingface_models/trained_whisper_models/checkpoint-5000")
trainer.train(resume_from_checkpoint=True)

model.save_pretrained("/home/wiseyak/suraj/Bengali_ASR/huggingface_models/trained_whisper_models/lora_v3_epoch3_ai4bharat")
tokenizer.save_pretrained("/home/wiseyak/suraj/Bengali_ASR/huggingface_models/trained_whisper_models/lora_v3_epoch3_ai4bharat")
# model.merge_and_unload()
# model.save_pretrained("/home/wiseyak/suraj/Bengali_ASR/huggingface_models/trained_whisper_models/lora_merged_v1_epoch3_ai4bharat")
# tokenizer.save_pretrained("/home/wiseyak/suraj/Bengali_ASR/huggingface_models/trained_whisper_models/lora_merged_v1_epoch3_ai4bharat")

# trainer.save("medium_ai4bharat_5000")
