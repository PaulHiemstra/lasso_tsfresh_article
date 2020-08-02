from scipy.io import wavfile
import pandas as pd
import numpy as np
from joblib import dump, load
from datetime import datetime
from helper_functions import audio_to_dataframe
from tsfresh import extract_relevant_features, extract_features
from tsfresh.feature_extraction import EfficientFCParameters
import glob
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import l1_min_c
from pathlib import Path

def sample_files(path, frac):
    return pd.Series(glob.glob(path + '/*')).sample(frac=frac)

print('Reading data...')
wav_files = glob.glob('sounds/kick/*.wav') + glob.glob('sounds/snare/*.wav') + glob.glob('sounds/tom/*.wav')
all_audio = pd.concat([audio_to_dataframe(path) for path in wav_files])
all_labels = pd.Series(np.repeat(['kick', 'snare', 'tom'], 25), 
                      index = wav_files)
all_audio.head()

regenerate_tsfresh=True
if regenerate_tsfresh:
    print('Generating tsfresh data...')
    settings = EfficientFCParameters()
    audio_tsfresh = extract_relevant_features(all_audio, all_labels, 
                                              column_id='file_id', column_sort='time_id', 
                                              default_fc_parameters=settings)
else:
    print('Reading tsfresh data...')
    all_labels = pd.read_pickle('pkl/drum_tsfresh_labels.pkl')
    audio_tsfresh = pd.read_pickle('pkl/drum_tsfresh.pkl')

print('Running logistic regression CV...')
print('Started CV %s' % datetime.now())
cs = l1_min_c(audio_tsfresh, all_labels, loss='log') * np.logspace(0, 7, 16)
cv_result = LogisticRegressionCV(Cs=cs,
                     penalty='l1', 
                     multi_class='ovr',
                     solver='saga',
                     tol=1e-6,
                     max_iter=int(1e6),
                     n_jobs=-1).fit(audio_tsfresh, all_labels)
print('Done CV %s' % datetime.now())

print('Dumping results...')
Path("pkl").mkdir(exist_ok=True)
all_labels.to_pickle('pkl/drum_tsfresh_labels.pkl')
audio_tsfresh.to_pickle('pkl/drum_tsfresh.pkl')
dump(cv_result, 'pkl/drum_logreg_cv.joblib')