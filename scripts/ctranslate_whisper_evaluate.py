import os
import pandas as pd
from sklearn.model_selection import train_test_split
#!ct2-transformers-converter --model /home/wiseyak/suraj/Bengali_ASR/huggingface_models/trained_whisper_models/lora_v2_merged_8000  --output_dir /home/wiseyak/suraj/Bengali_ASR/huggingface_models/trained_whisper_models/ctranslate_lora_v2_merged_8000 --quantization float16
#!ct2-transformers-converter --model /home/wiseyak/suraj/Bengali_ASR/huggingface_models/competition_data_whisper_models/merged_lora_ai4bharat_competition  --output_dir /home/wiseyak/suraj/Bengali_ASR/huggingface_models/competition_data_whisper_models/ctranslate_merged_lora_ai4bharat_competition --quantization float16

from faster_whisper import WhisperModel
import evaluate
from bnunicodenormalizer import Normalizer 


bnorm = Normalizer()

metric = evaluate.load("wer")

ctranslate_model_path = "/home/wiseyak/suraj/Bengali_ASR/huggingface_models/lora_ai4bharat_models/ctranslate_lora_v2_merged_8000"
ctranslate_model_path = '/home/wiseyak/suraj/Bengali_ASR/huggingface_models/competition_data_whisper_models/ctranslate_merged_lora_ai4bharat_competition'
model = WhisperModel(ctranslate_model_path, device="cuda", compute_type="float16")

def normalize(sen):
    _words = [bnorm(word)['normalized']  for word in sen.split()]
    return " ".join([word for word in _words if word is not None])

def get_transcript(audio_path):
    segments, info = model.transcribe(audio_path, beam_size=5,  without_timestamps=True,patience=1,condition_on_previous_text=True, language="bn")
    transcript = []
    for segment in segments:
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
         transcript.append(segment.text)
    return ''.join(transcript)


# ai4bharat dataset evaluate

# df = pd.read_csv('/home/wiseyak/suraj/Bengali_ASR/bengali_dataset.csv', encoding="utf-8")
# df = df.loc[df['Transcript'].str.len() < 180]
# train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
# eval_sample = test_df[:256]
# path_template = "/home/wiseyak/suraj/Bengali_ASR/newsonair_v5/bengali_old/{}"

# competition train dataset evaluate
df = pd.read_csv('/home/wiseyak/suraj/Bengali_ASR/competition_Dataset/train.csv')
# Define the features (X) and the target (y)
df = df.loc[df['sentence'].str.len() < 180]
train_df, test_df = train_test_split(df, test_size=0.001, random_state=42)
train_sample = train_df
eval_sample = test_df[:32]
# eval_sample.sentence.sentence= eval_sample.sentence.apply(lambda x:normalize(x))

path_template = "/home/wiseyak/suraj/Bengali_ASR/competition_Dataset/train_mp3s/{}.mp3"

references = []
predictions = []
for _, row in eval_sample.iterrows():
    path = row["id"] #path for ai4bharat data
    label = row["sentence"] #Transcript for ai4bharat data
    prediction = get_transcript(path_template.format(path))
    references.extend([label])
    predictions.extend([prediction])
    # print(prediction)
    # print(label, prediction)
    # break

wer = metric.compute(references=references, predictions=predictions)
wer = round(100 * wer, 2)
print("WER:", wer)