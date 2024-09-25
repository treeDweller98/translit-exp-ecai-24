# NOTE: Reader code modified slightly for uniformity to run across all notebooks. May require debugging in some cases.

import re
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = '/kaggle/input/translit-datasets/'
SEED = 42


# Version: Translit Bangla Sentiment

def read_dataset():
    dataset_name = 'tb_senti'
    text_col = 'sentence'
    filepath = INPUT_PATH + 'Transliterated-Bengali-Positive-and-Negative-Corpus.csv'
    df = pd.read_csv(filepath, dtype={'Label': 'category', 'Comments': 'string'})
    df.rename(columns={'Label': 'label', 'Comments': 'sentence'}, inplace=True)
    
    def clean(text: str) -> str:
        # tidy up punctuations, remove URLs, @USERs, hashtags, whitespaces
        text = re.sub('[\.|,]{2,}', '... ', text)
        text = re.sub('\?+', '? ', text)
        text = re.sub('\!+', '! ', text)
        text = re.sub('(?:\@|http?\://|https?\://|www)\S+|#', '', text)
        text = re.sub('\s\s+', ' ', text)
        return text
    
    df.sentence = df.sentence.apply(clean)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    renamed_cats = {
       '0': 'positive',
       '1': 'negative',
    }
    train_df.label = train_df.label.cat.rename_categories(renamed_cats)
    test_df.label  =  test_df.label.cat.rename_categories(renamed_cats)
    label_names = list(df.label.cat.categories)

    return train_df, test_df, label_names, dataset_name, text_col



# Version: Translit Bangla Sentiment Augmented

def read_dataset():
    dataset_name = 'tb_senti'
    text_col = 'sentence'
    filepath = INPUT_PATH + 'Transliterated-Bengali-Positive-and-Negative-Corpus.csv'
    df = pd.read_csv(filepath, dtype={'Label': 'category', 'Comments': 'string'})
    df.rename(columns={'Label': 'label', 'Comments': 'sentence'}, inplace=True)
    
    def clean(text: str) -> str:
        # tidy up punctuations, remove URLs, @USERs, hashtags, whitespaces
        text = re.sub('[\.|,]{2,}', '... ', text)
        text = re.sub('\?+', '? ', text)
        text = re.sub('\!+', '! ', text)
        text = re.sub('(?:\@|http?\://|https?\://|www)\S+|#', '', text)
        text = re.sub('\s\s+', ' ', text)
        return text
    
    df.sentence = df.sentence.apply(clean)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    # Augment train set
    positives_filepath = INPUT_PATH + 'augment-raw/tb-senti-positive-gptgen.txt'
    negatives_filepath = INPUT_PATH + 'augment-raw/tb-senti-negative-gptgen.txt'
    
    def make_df_from_txt(path: str, label: str) -> pd.DataFrame:
        lines = []
        with open(path, 'r') as f:
            for line in f:
                lines.append(clean(line.rstrip('\n')))
        
        lines_df = pd.DataFrame(lines, columns=['sentence'])
        lines_df['label'] = label
        return lines_df
    
    train_df = pd.concat(
        [
            train_df,
            make_df_from_txt(positives_filepath, '0'),
            make_df_from_txt(negatives_filepath, '1'),
        ], 
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Rename categories
    renamed_cats = {
       '0': 'positive',
       '1': 'negative',
    }
    train_df.label = train_df.label.astype('category')
    train_df.label = train_df.label.cat.rename_categories(renamed_cats)
    test_df.label  =  test_df.label.cat.rename_categories(renamed_cats)
    label_names = list(df.label.cat.categories)

    return train_df, test_df, label_names, dataset_name, text_col



# Version: Translit Bangla Offensive

def read_dataset():
    dataset_name = 'tb_olid'
    text_col = 'sentence'
    train_path = INPUT_PATH + 'TB-OLID_train.json'
    test_path  = INPUT_PATH + 'TB-OLID_test.json'
    col_types  = {'offensive_gold': 'category', 'text': 'string'}
    rename_col = {'offensive_gold': 'label', 'text': 'sentence'}
    keep_cols  = ['label', 'sentence']
    
    train_df = pd.read_json(train_path, dtype=col_types).rename(columns=rename_col)[keep_cols]
    test_df  = pd.read_json(test_path,  dtype=col_types).rename(columns=rename_col)[keep_cols]
    
    def clean(text):
        # tidy up punctuations, remove URLs, @USERs, hashtags, whitespaces
        text = re.sub('\?+', '', text)      # TB-OLID has names replaced with ????
        text = re.sub('[\.|,]{2,}', '... ', text)
        text = re.sub('\?+', '? ', text)
        text = re.sub('\!+', '! ', text)
        text = re.sub('(?:\@|http?\://|https?\://|www)\S+|#', '', text)
        text = re.sub('\s\s+', ' ', text)
        return text
    
    train_df.sentence = train_df.sentence.apply(clean)
    test_df.sentence  = test_df.sentence.apply(clean)

    renamed_cats = {'N': 'non-offensive', 'O': 'offensive'}
    train_df.label = train_df.label.cat.rename_categories(renamed_cats)
    test_df.label  =  test_df.label.cat.rename_categories(renamed_cats)
    label_names = list(train_df.label.cat.categories)

    return train_df, test_df, label_names, dataset_name, text_col



# Version: Translit Bangla Offensive Augmented

def read_dataset():
    dataset_name = 'tb_olid'
    text_col = 'sentence'
    train_path = INPUT_PATH + 'TB-OLID_train.json'
    test_path  = INPUT_PATH + 'TB-OLID_test.json'
    col_types  = {'offensive_gold': 'category', 'text': 'string'}
    rename_col = {'offensive_gold': 'label', 'text': 'sentence'}
    keep_cols  = ['label', 'sentence']
    
    train_df = pd.read_json(train_path, dtype=col_types).rename(columns=rename_col)[keep_cols]
    test_df  = pd.read_json(test_path,  dtype=col_types).rename(columns=rename_col)[keep_cols]
    
    def clean(text):
        # tidy up punctuations, remove URLs, @USERs, hashtags, whitespaces
        text = re.sub('\?+', '', text)      # TB-OLID has names replaced with ????
        text = re.sub('[\.|,]{2,}', '... ', text)
        text = re.sub('\?+', '? ', text)
        text = re.sub('\!+', '! ', text)
        text = re.sub('(?:\@|http?\://|https?\://|www)\S+|#', '', text)
        text = re.sub('\s\s+', ' ', text)
        return text
    
    train_df.sentence = train_df.sentence.apply(clean)
    test_df.sentence  = test_df.sentence.apply(clean)

    # Augment train set
    offensives_filepath    = INPUT_PATH + 'augment-raw/tb-olid-off-gptgen.txt'
    nonoffensives_filepath = INPUT_PATH + 'augment-raw/tb-olid-nonoff-gptgen.txt'
    
    def make_df_from_txt(path: str, label: str) -> pd.DataFrame:
        lines = []
        with open(path, 'r') as f:
            for line in f:
                lines.append(clean(line.rstrip('\n')))
        
        lines_df = pd.DataFrame(lines, columns=['sentence'])
        lines_df['label'] = label
        return lines_df
    
    train_df = pd.concat(
        [
            train_df,
            make_df_from_txt(offensives_filepath, 'O'),
            make_df_from_txt(nonoffensives_filepath, 'N'),
        ], 
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    # Rename categories
    renamed_cats = {'N': 'non-offensive', 'O': 'offensive'}
    train_df.label = train_df.label.astype('category')
    train_df.label = train_df.label.cat.rename_categories(renamed_cats)
    test_df.label  =  test_df.label.cat.rename_categories(renamed_cats)
    label_names = list(train_df.label.cat.categories)

    return train_df, test_df, label_names, dataset_name, text_col



# Version: Translated Bangla datasets loader
def read_dataset():
    dataset_name = 'tb_senti'
    # dataset_name = 'tb_olid'
    text_col = 'translated'
    train_path = INPUT_PATH + f'translated/{dataset_name}_train_translated.csv'
    test_path  = INPUT_PATH + f'translated/{dataset_name}_test_translated.csv'
    coltypes = {'label': 'category', 'sentence': 'string', 'translated': 'string'}
    
    train_df = pd.read_csv(train_path, dtype=coltypes, index_col=0)
    test_df  = pd.read_csv(test_path,  dtype=coltypes, index_col=0)
    
    label_names = list(train_df.label.cat.categories)
    train_df.label = train_df.label.cat.codes
    test_df.label  =  test_df.label.cat.codes
    
    train_df.dropna(inplace=True, ignore_index=True)
    test_df.dropna(inplace=True, ignore_index=True)
    
    return train_df, test_df, label_names, dataset_name, text_col



# Version: Translit Arabizi Sentiment

def read_dataset():
    dataset_name = 'arabizi_senti'
    text_col = 'sentence'
    filepath = INPUT_PATH + 'TUNIZI_V1.txt'
    csv_read_options = {
        'header': None,
        'usecols': [0, 1],
        'names': ['sentence', 'label'],
        'dtype': {'sentence': 'string', 'label': 'category'},
        'on_bad_lines': 'skip',
        'sep': ';'
    }
    df = pd.read_csv(filepath,**csv_read_options)

    df = df[df.label.isin(['-1', '1'])]
    df.label = df.label.cat.remove_unused_categories()
    
    renamed_cats = {'-1': 'negative', '1': 'positive'}
    df.label = df.label.cat.rename_categories(renamed_cats)
    label_names = list(df.label.cat.categories)

    def clean(text: str) -> str:
        # tidy up punctuations, remove URLs, @USERs, hashtags, whitespaces
        text = re.sub('[\.|,]{2,}', '... ', text)
        text = re.sub('\?+', '? ', text)
        text = re.sub('\!+', '! ', text)
        text = re.sub('(?:\@|http?\://|https?\://|www)\S+|#', '', text)
        text = re.sub('\s\s+', ' ', text)
        return text
    
    df.sentence = df.sentence.apply(clean)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)

    return train_df, test_df, label_names, dataset_name, text_col



# Version: Translit Arabizi Offensive

def read_dataset():
    dataset_name = 'arabizi-offensive'
    text_col = 'sentence'
    filepath = INPUT_PATH + 'Arabizi-Off_Lang_Dataset.csv'
    csv_read_options = {
        'usecols': ['Text', 'Generic Class'],
        'dtype': {'Text': 'string', 'Generic Class': 'category'},
    }
    renamed_cols = {'Text': 'sentence', 'Generic Class': 'label'}
    
    df = pd.read_csv(filepath,**csv_read_options)
    df.rename(columns=renamed_cols, inplace=True)
    label_names = list(df.label.cat.categories)

    def clean(text: str) -> str:
        # tidy up punctuations, remove URLs, @USERs, hashtags, whitespaces
        text = re.sub('[\.|,]{2,}', '... ', text)
        text = re.sub('\?+', '? ', text)
        text = re.sub('\!+', '! ', text)
        text = re.sub('(?:\@|http?\://|https?\://|www)\S+|#', '', text)
        text = re.sub('\s\s+', ' ', text)
        return text
    
    df.sentence = df.sentence.apply(clean)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    
    return train_df, test_df, label_names, dataset_name, text_col



# Version: Translit Hindi Sentiment

def read_dataset():
    dataset_name = 'hing_senti'
    text_col = 'sentence'
    train_path = INPUT_PATH + 'hing-senti-train.tsv'
    test_path  = INPUT_PATH + 'hing-senti-valid.tsv'

    csv_read_options = {
        'header': None,
        'usecols': [1, 2],
        'names': ['sentence', 'label'],
        'dtype': {'sentence': 'string', 'label': 'category'},
        'on_bad_lines': 'skip',
        'sep': '\t'
    }

    train_df = pd.read_csv(train_path,**csv_read_options).dropna()
    test_df  = pd.read_csv(test_path,**csv_read_options).dropna()

    def clean(text):
        # tidy up punctuations, remove URLs, @USERs, hashtags, whitespaces
        text = re.sub('[\.|,]{2,}', '... ', text)
        text = re.sub('\?+', '? ', text)
        text = re.sub('\!+', '! ', text)
        text = re.sub('(?:\@|http?\://|https?\://|www)\S+|#', '', text)
        text = re.sub('\s\s+', ' ', text)
        return text
    
    train_df.sentence = train_df.sentence.apply(clean)
    test_df.sentence  = test_df.sentence.apply(clean)
    
    renamed_cats = {'0': 'negative', '1': 'neutral', '2': 'positive'}
    train_df.label = train_df.label.cat.rename_categories(renamed_cats)
    test_df.label  =  test_df.label.cat.rename_categories(renamed_cats)
    
    label_names = list(train_df.label.cat.categories)
    
    return train_df, test_df, label_names, dataset_name, text_col



# Version: Translit Hindi Offensive

def read_dataset():
    dataset_name = 'hing-offensive'
    text_col = 'sentence'
    filepath = INPUT_PATH + 'hinglish_tweets_hate_speech.tsv'
    csv_read_options = {
        'header': None,
        'usecols': [0, 1],
        'names': ['sentence', 'label'],
        'dtype': {'sentence': 'string', 'label': 'category'},
        'on_bad_lines': 'skip',
        'sep': '\t'
    }
    df = pd.read_csv(filepath,**csv_read_options)

    df = df[df.label.isin(['no', 'yes'])]                   # remove malformed labels
    df.label = df.label.cat.remove_unused_categories()      # reset category indexing
    
    renamed_cats = {'no': 'non-hate', 'yes': 'hate'}
    df.label = df.label.cat.rename_categories(renamed_cats)
    
    label_names = list(df.label.cat.categories)

    def clean(text: str) -> str:
        # tidy up punctuations, remove URLs, @USERs, hashtags, whitespaces
        text = re.sub('[\.|,]{2,}', '... ', text)
        text = re.sub('\?+', '? ', text)
        text = re.sub('\!+', '! ', text)
        text = re.sub('(?:\@|http?\://|https?\://|www)\S+|#', '', text)
        text = re.sub('\s\s+', ' ', text)
        return text
    
    df.sentence = df.sentence.apply(clean)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    
    return train_df, test_df, label_names, dataset_name