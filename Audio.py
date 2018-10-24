# https://github.com/keithito/tacotron/blob/master/util/audio.py
# https://github.com/carpedm20/multi-speaker-tacotron-tensorflow/blob/master/audio/__init__.py
# I only changed the hparams to usual parameters from oroginal code.

import numpy as np;
import tensorflow as tf;
from scipy import signal;
import librosa.filters;
import librosa;


def preemphasis(x, preemphasis = 0.97):
    return signal.lfilter([1, -preemphasis], [1], x)

def inv_preemphasis(x, preemphasis = 0.97):
    return signal.lfilter([1], [1, -preemphasis], x)


def spectrogram(y, num_freq, frame_shift_ms, frame_length_ms, sample_rate, ref_level_db = 20):
    D = _stft(preemphasis(y), num_freq, frame_shift_ms, frame_length_ms, sample_rate)
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return S#_normalize(S)

def inv_spectrogram(spectrogram, num_freq, frame_shift_ms, frame_length_ms, sample_rate, ref_level_db = 20, power = 1.5):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** power, num_freq, frame_shift_ms, frame_length_ms, sample_rate))          # Reconstruct phase

def inv_spectrogram_tensorflow(spectrogram, num_freq, frame_shift_ms, frame_length_ms, sample_rate, ref_level_db = 20, power = 1.5):
    '''Builds computational graph to convert spectrogram to waveform using TensorFlow.
    Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
    inv_preemphasis on the output after running the graph.
    '''
    S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + ref_level_db)
    return _griffin_lim_tensorflow(tf.pow(S, power), num_freq, frame_shift_ms, frame_length_ms, sample_rate)


def melspectrogram(y, num_freq, frame_shift_ms, frame_length_ms, num_mels, sample_rate):
    D = _stft(preemphasis(y), num_freq, frame_shift_ms, frame_length_ms, sample_rate)
    S = _amp_to_db(_linear_to_mel(np.abs(D), num_freq, num_mels, sample_rate))
    return S#_normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8, sample_rate=20000):
    window_length = int(sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x+window_length]) < threshold:
            return x + hop_length
    return len(wav)


def _griffin_lim(S, num_freq, frame_shift_ms, frame_length_ms, sample_rate, griffin_lim_iters = 60):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, num_freq, frame_shift_ms, frame_length_ms, sample_rate)))
        y = _istft(S_complex * angles)
    return y

def _griffin_lim_tensorflow(S, num_freq, frame_shift_ms, frame_length_ms, sample_rate, griffin_lim_iters = 60):
    '''TensorFlow implementation of Griffin-Lim
    Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
    '''
    with tf.variable_scope('griffinlim'):
        # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = _istft_tensorflow(S_complex, num_freq, frame_shift_ms, frame_length_ms, sample_rate)

        index = tf.constant(0);
        def condition(index, y):
            return tf.less(index, griffin_lim_iters);

        def while_Body(index, y):
            est = _stft_tensorflow(y, num_freq, frame_shift_ms, frame_length_ms, sample_rate)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = _istft_tensorflow(S_complex * angles, num_freq, frame_shift_ms, frame_length_ms, sample_rate)            
            return index+1, y
        
        _, y = tf.while_loop(condition, while_Body, [index, y])
            
        return tf.squeeze(y, 0)

def _stft(y, num_freq, frame_shift_ms, frame_length_ms, sample_rate):
    n_fft, hop_length, win_length = _stft_parameters(num_freq, frame_shift_ms, frame_length_ms, sample_rate)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _istft(y):
    _, hop_length, win_length = _stft_parameters(num_freq, frame_shift_ms, frame_length_ms, sample_rate)
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def _stft_tensorflow(signals, num_freq, frame_shift_ms, frame_length_ms, sample_rate):
    n_fft, hop_length, win_length = _stft_parameters(num_freq, frame_shift_ms, frame_length_ms, sample_rate)
    return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)

def _istft_tensorflow(stfts, num_freq, frame_shift_ms, frame_length_ms, sample_rate):    
    n_fft, hop_length, win_length = _stft_parameters(num_freq, frame_shift_ms, frame_length_ms, sample_rate)
    return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)

def _stft_parameters(num_freq, frame_shift_ms, frame_length_ms, sample_rate):    
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return n_fft, hop_length, win_length


def _linear_to_mel(spectrogram, num_freq, num_mels, sample_rate):
    _mel_basis = _build_mel_basis(num_freq, num_mels, sample_rate)
    return np.dot(_mel_basis, spectrogram)

def _build_mel_basis(num_freq, num_mels, sample_rate):
    n_fft = (num_freq - 1) * 2
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _normalize(S, min_level_db = -100):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def _denormalize(S, min_level_db = -100):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def _denormalize_tensorflow(S, min_level_db = -100):
    return (tf.clip_by_value(S, 0, 1) * -min_level_db) + min_level_db
