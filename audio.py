import os
import librosa
import numpy as np
import ikrlib as ilib
import soundfile as sf
import noisereduce as nr

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def audio_adjust(dir):
    new_dir = ilib.get_last_two_dirs(dir) + "/rs/"
    min_silence_len = 1000  # Minimálna dĺžka ticha (ms)
    silence_thresh = -44    # prah ticha (dB)

    print(f"Removing silence from records in directory {dir}")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    for f in os.listdir(dir):
        if f[-3:] == "wav":
            input_file = ilib.get_last_two_dirs(dir) + "/" + f
            audio = AudioSegment.from_wav(input_file)
            nonsilent_intervals = detect_nonsilent(audio, min_silence_len, silence_thresh)

            # Zkonkatenuj invervaly kde nie je ticho
            non_silent_audio = AudioSegment.empty()
            for start, end in nonsilent_intervals:
                non_silent_audio += audio[start:end]

            # Ulož audio
            output_file = new_dir + f
            non_silent_audio.export(output_file, format="wav")

def reduce_noise(dir):
    print(f"Removing noise from records in directory {dir}")
    new_dir = ilib.get_last_two_dirs(dir) + "/rn/"

    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    for f in os.listdir(dir):
        if f[-3:] == "wav":
            input_file = dir + "/" + f

            # Load the audio file
            audio, sr = librosa.load(input_file, sr=None)

            # Select a portion of the audio that contains only noise (e.g., the first 0.5 seconds)
            noise_sample = audio[:int(sr * 0.5)]

            # Perform noise reduction using the noise sample
            reduced_audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample)

            # Save the noise-reduced audio to a new file
            output_file = new_dir + f
            sf.write(output_file, reduced_audio, sr)

def data_augumentation(dir):
    new_dir = ilib.get_last_two_dirs(dir) + "/da/"
    print(f"Removing noise from records in directory {dir}")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    for f in os.listdir(dir):
        if f[-3:] == "wav":
            input_file = dir + "/" + f
            print("Data augumentation of file: " + input_file)

            time_stretched_audio = ilib.apply_time_stretching(input_file)
            pitch_shifted_audio = ilib.apply_pitch_shifting(input_file, semitones=2)
            time_shifted_audio = ilib.apply_time_shifting(input_file, shift_ms=500)

            # insert "aug" between "audio" and ".wav"
            stretched_file = new_dir + f[:-4] + "_stretched_aug.wav"
            pitch_shifted_file = new_dir + f[:-4] + "_pitch_shifted_aug.wav"
            time_shifted_file = new_dir + f[:-4] + "_time_shifted_aug.wav"

            time_stretched_audio.export(stretched_file, format="wav")
            pitch_shifted_audio.export(pitch_shifted_file, format="wav")
            time_shifted_audio.export(time_shifted_file, format="wav")

def pre_emphasis(dir):
    data = []
    for f in os.listdir(dir):
        if f[-3:] == "wav":
            input_file = dir + "/" + f
            print("Proccessing input file: " + input_file)
            sample_rate, audio_samples = ilib.read_wav_file(input_file)
            emphasized_audio = ilib.apply_pre_emphasis(audio_samples)

            assert(sample_rate==16000)
            data.append(ilib.extract_mfcc(emphasized_audio, sample_rate))
    return data

def min_max_normalize(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

def compute_deltas(cepstral_coeffs, window_size=2):
    num_frames, num_coeffs = cepstral_coeffs.shape
    deltas = np.zeros((num_frames, num_coeffs))

    for t in range(num_frames):
        window_start = max(0, t - window_size)
        window_end = min(num_frames, t + window_size + 1)
        window_indices = np.arange(window_start, window_end)
        window_weights = window_indices - t

        weighted_sum = np.sum(window_weights[:, np.newaxis] * cepstral_coeffs[window_indices, :], axis=0)
        weight_sum_squared = np.sum(window_weights ** 2)

        deltas[t] = weighted_sum / weight_sum_squared

    return deltas

def cepstral_mean_subtraction(cepstral_coeffs):
    # Calculate the mean of the cepstral coefficients across all frames (axis 0)
    mean_coeffs = np.mean(cepstral_coeffs, axis=0)

    # Subtract the mean from the original cepstral coefficients
    cms_coeffs = cepstral_coeffs - mean_coeffs

    return cms_coeffs