import pandas as pd
import numpy as np

from bnunicodenormalizer import Normalizer 


bnorm = Normalizer()
import re
chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\—\‘\'\‚\“\”\…\\\/]'

def remove_special_characters(sentence):
    return re.sub(chars_to_ignore_regex, '', sentence)
    

def normalize(sentence):
    _words = [bnorm(word)['normalized']  for word in sentence.split()]
    return " ".join([word for word in _words if word is not None])


'''
# Columns
Index(['id', 'filename', 'client_id', 'ggl_cer', 'ggl_mer', 'ggl_wer',
       'ggl_wil', 'ggl_wip', 'google_preds', 'path', 'sentence',
       'yellowking_preds', 'ykg_mer', 'ykg_wer', 'ykg_wil', 'ykg_wip',
       'ykg_cer'],
      dtype='object')
'''

write_path = "/home/wiseyak/suraj/Bengali_AI_SPeech_Recognition_2023/datasets/competition_dataset"
df = pd.read_csv('/home/wiseyak/suraj/Bengali_AI_SPeech_Recognition_2023/datasets/competition_dataset/train_metadata.csv')
threshold_wer = 0.6

df = df[df['ykg_wer'] <= threshold_wer]
df['sentence'] = df['sentence'].apply(remove_special_characters)
df['sentence'] = df['sentence'].apply(normalize)

train_df = df[:-200]
test_df = df[-200:]

train_df.to_csv(write_path+'/filtered_train.csv', index=False, columns=['id', 'sentence', 'yellowking_preds'])
test_df.to_csv(write_path+'/filtered_test.csv', index=False, columns=['id', 'sentence', 'yellowking_preds'])


