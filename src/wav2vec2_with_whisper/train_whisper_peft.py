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
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from dotenv import load_dotenv
import huggingface_hub
import warnings
import warnings
warnings.filterwarnings('ignore')
# Load variables from .env file into the environment
load_dotenv('../../.env')

# Access the variable
hf_token = os.getenv("hf_token")
huggingface_hub.login(token=hf_token)

wandb_token = os.getenv("wandb_token")

wandb.login(key=wandb_token)

# Load your dataset from the CSV file into a DataFrame
train_df = pd.read_csv('/home/wiseyak/suraj/Bengali_AI_SPeech_Recognition_2023/datasets/competition_dataset/filtered_train.csv', encoding='utf-8', on_bad_lines='skip')
train_df = train_df.loc[(train_df['sentence'].str.len() + train_df['yellowking_preds'].str.len()) < 180]

test_df = pd.read_csv('/home/wiseyak/suraj/Bengali_AI_SPeech_Recognition_2023/datasets/competition_dataset/filtered_test.csv', encoding='utf-8', on_bad_lines='skip')
test_df = test_df.loc[(test_df['sentence'].str.len() + test_df['yellowking_preds'].str.len()) < 180]
test_df = test_df.loc[(test_df['sentence'].str.len() + test_df['yellowking_preds'].str.len()) > 18]

train_sample = train_df
eval_sample = test_df[:32]
path_template = "/home/wiseyak/suraj/Bengali_AI_SPeech_Recognition_2023/datasets/competition_dataset/train_mp3s/{}.mp3"

# Load processors
whisper_model_id = "/home/wiseyak/suraj/Bengali_AI_SPeech_Recognition_2023/datasets/whisper_models/bengali_models/whisper-medium-bn_alldata_multigpu/"
feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_id)
tokenizer = WhisperTokenizer.from_pretrained(whisper_model_id, language="bengali", task="transcribe")
processor = WhisperProcessor.from_pretrained(whisper_model_id, language="bengali", task="transcribe")


'''
# Replace valle_venv/lib64/python3.8/site-packages/transformers/models/whisper/tokenization_whisper.py line 431
 426     def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
 427         """Build model inputs from a sequence by appending eos_token_id."""
 428         if token_ids_1 is None:
 429             return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
 430         # We don't expect to process pairs, but leave the pair logic for API consistency
 431         start_of_prev_id = self.all_special_ids[-3]
 432         return [start_of_prev_id] + token_ids_1 + self.prefix_tokens + token_ids_0 + [self.eos_token_id]
 433         #return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]
 434 

'''

def dataset_generator(df):
    for _, row in df.iterrows():
        audio_path = path_template.format(row["id"])
        # print("*"*100, audio_path)
        audio_array = librosa.load(audio_path)[0]
        # print(tokenizer(row["sentence"], row["yellowking_preds"]))
        yield {
            "input_features": feature_extractor(audio_array, sampling_rate=16000).input_features[0], 
            "labels": tokenizer(row["sentence"], row["yellowking_preds"]).input_ids
            # "labels": tokenizer(row["sentence"]).input_ids
        }

# Iterator dataset
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
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item() or (labels[:,0] == 50361).all().cpu().item():
            labels = labels[:, 1:]

        # Replace initial prompt tokens with -100 to ignore them during loss calculation
        
        bos_index = (labels == self.processor.tokenizer.bos_token_id).nonzero()[:, 1]
        prompt_mask = torch.arange(labels.shape[1]) < bos_index.unsqueeze(1)
        labels[prompt_mask] = -100
        
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

model = WhisperForConditionalGeneration.from_pretrained(whisper_model_id)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.enable_input_require_grads()

model = prepare_model_for_int8_training(model)
config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)

# model.print_trainable_parameters()

from transformers import Seq2SeqTrainingArguments

model_save_path = "/home/wiseyak/suraj/Bengali_AI_SPeech_Recognition_2023/models/competition_data_whisper_models"
training_args = Seq2SeqTrainingArguments(
    output_dir=model_save_path,  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=300,
    max_steps=8000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    generation_max_length=225,
    save_steps=2000,
    eval_steps=200,
    logging_steps=25,
    report_to="wandb",
    load_best_model_at_end=False,
    # metric_for_best_model="wer",
    # greater_is_better=False,
    do_eval=False,
    # ignore_data_skip=True, # for continuing training from checkpoint
    save_total_limit=5,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
    push_to_hub = True,
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
            kwargs['tokenizer'].save_pretrained(checkpoint_folder)
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
    # callbacks=[SavePeftModelCallback],
)
model.config.use_cache = False
trainer.train()
model.save_pretrained(model_save_path+"wav2vec2_with_w_v1")
tokenizer.save_pretrained(model_save_path+"wav2vec2_with_w_v1")
