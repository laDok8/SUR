from scipy.special import logsumexp

import ikrlib as ilib
import numpy as np
import os
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


class Audio:
    def __init__(self, CLASSES, train_path, dev_path):
        self.CLASSES = CLASSES
        self.dev = dev_path
        self.train = train_path
        self.audio_adjust_enabled = False
        self.reduce_noise_enabled = False
        self.data_augmentation_enabled = False
        self.data_pre_emphasis = False
        self.coefficients_normalization = False
        self.delta_coefficients_enabled = False
        self.cepstral_mean_subtraction_enabled = False

    @staticmethod
    def audio_adjust(dir):
        new_dir = ilib.get_last_two_dirs(dir) + "/rs/"
        min_silence_len = 1000  # Minimálna dĺžka ticha (ms)
        silence_thresh = -44    # prah ticha (dB)

        print(f"Removing silence from records in directory {dir}")
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        for f in os.listdir(dir):
            print(f"Removing silence from file {f}")
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

    @staticmethod
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
                print(f)
                reduced_audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample)

                # Save the noise-reduced audio to a new file
                output_file = new_dir + f
                sf.write(output_file, reduced_audio, sr)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def min_max_normalize(data):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        return normalized_data

    @staticmethod
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

    @staticmethod
    def cepstral_mean_subtraction(cepstral_coeffs):
        # Calculate the mean of the cepstral coefficients across all frames (axis 0)
        mean_coeffs = np.mean(cepstral_coeffs, axis=0)

        # Subtract the mean from the original cepstral coefficients
        cms_coeffs = cepstral_coeffs - mean_coeffs

        return cms_coeffs


    def do_audio_adjust(self, audio_adjust_enabled, eval=True):
        self.audio_adjust_enabled = audio_adjust_enabled
        if self.audio_adjust_enabled:
            if eval:
                for i in range(1, self.CLASSES + 1):
                    Audio.audio_adjust(ilib.get_directory(f"{self.train}/{i}"))
                Audio.audio_adjust(ilib.get_directory(f"{self.dev}"))
            else:
                for i in range(1, self.CLASSES + 1):
                    Audio.audio_adjust(ilib.get_directory(f"{self.train}/{i}"))
                    Audio.audio_adjust(ilib.get_directory(f"{self.dev}/{i}"))
        print("Silence was successfully removed")

    def do_reduce_noise(self, reduce_noise_enabled, eval=True):
        self.reduce_noise_enabled = reduce_noise_enabled
        if reduce_noise_enabled:
            if eval:
                for i in range(1, self.CLASSES + 1):
                    Audio.reduce_noise(ilib.get_directory(f"{self.train}/{i}", self.audio_adjust_enabled))
                Audio.reduce_noise(ilib.get_directory(f"{self.dev}", self.audio_adjust_enabled))
            else:
                for i in range(1, self.CLASSES + 1):
                    Audio.reduce_noise(ilib.get_directory(f"{self.train}/{i}", self.audio_adjust_enabled))
                    Audio.reduce_noise(ilib.get_directory(f"{self.dev}/{i}", self.audio_adjust_enabled))

            print("Noise was successfully removed")

    def do_data_augmentation(self, data_augmentation_enabled):
        self.data_augmentation_enabled = data_augmentation_enabled
        if self.data_augmentation_enabled:
            for i in range(1, self.CLASSES + 1):
                Audio.data_augumentation(ilib.get_directory(f"{self.train}/{i}", self.audio_adjust_enabled, self.reduce_noise_enabled))
            print("Data augumentation was done")

    def do_data_pre_emphasis(self, eval=True):
        train_audio = {}
        dev_audio = {}

        if eval:
            for i in range(1, self.CLASSES + 1):
                train_audio[i] = np.vstack(Audio.pre_emphasis(
                    ilib.get_directory(f'{self.train}/{i}', self.audio_adjust_enabled, self.reduce_noise_enabled,
                                       self.data_augmentation_enabled)))
            dev_audio = list(
                Audio.pre_emphasis(ilib.get_directory(f'{self.dev}', self.audio_adjust_enabled, self.reduce_noise_enabled)))
        else:
            for i in range(1, self.CLASSES + 1):
                train_audio[i] = np.vstack(Audio.pre_emphasis(
                    ilib.get_directory(f'{self.train}/{i}', self.audio_adjust_enabled, self.reduce_noise_enabled,
                                       self.data_augmentation_enabled)))
                dev_audio[i] = list(
                    Audio.pre_emphasis(ilib.get_directory(f'{self.dev}/{i}', self.audio_adjust_enabled, self.reduce_noise_enabled)))
        print("Pre emphasis was successfull")
        return train_audio, dev_audio

    def do_classic_load(self, eval=True):
        train_audio = {}
        dev_audio = {}

        if eval:
            for i in range(1, self.CLASSES + 1):
                train_audio[i] = np.vstack(list(ilib.wav16khz2mfcc(
                    ilib.get_directory(f'{self.train}/{i}', True, True,
                                       self.data_augmentation_enabled)).values()))
            dev_audio = ilib.wav16khz2mfcc(
                ilib.get_directory(f'{self.dev}', True, True))
        else:
            for i in range(1, self.CLASSES + 1):
                train_audio[i] = np.vstack(list(ilib.wav16khz2mfcc(
                    ilib.get_directory(f'{self.train}/{i}', self.audio_adjust_enabled, self.reduce_noise_enabled,
                                       self.data_augmentation_enabled)).values()))
                dev_audio[i] = list(ilib.wav16khz2mfcc(
                    ilib.get_directory(f'{self.dev}/{i}', self.audio_adjust_enabled, self.reduce_noise_enabled)).values())
        print("Loading data was successful")
        return train_audio, dev_audio

    def do_coefficients_normalization(self, train_audio, coefficients_normalization):
        self.coefficients_normalization = coefficients_normalization
        if coefficients_normalization:
            for i in range(1, self.CLASSES + 1):
                train_audio[i] = Audio.min_max_normalize(train_audio[i])
        return train_audio

    def do_delta_coefficients(self, train_audio, delta_coefficients_enabled):
        self.delta_coefficients_enabled = delta_coefficients_enabled
        if delta_coefficients_enabled:
            for i in range(1, self.CLASSES + 1):
                train_delta_coeffs = Audio.compute_deltas(train_audio[i], window_size=2)
                train_derivative_delta_coeffs = Audio.compute_deltas(train_audio[i], window_size=2)
                train_audio[i] = np.concatenate((train_audio[i], train_delta_coeffs, train_derivative_delta_coeffs), axis=1)
        return train_audio

    def do_cepstral_mean_subtraction(self, train_audio, cepstral_mean_subtraction_enabled):
        self.cepstral_mean_subtraction_enabled = cepstral_mean_subtraction_enabled
        if cepstral_mean_subtraction_enabled:
            for i in range(1, self.CLASSES + 1):
                train_audio[i] = Audio.cepstral_mean_subtraction(train_audio[i])
        return train_audio

    def train_gmm(self, train_audio, M=3, EPOCH=30):
        MUs = {}
        COVs = {}
        Ws = {}
        for i in range(1, self.CLASSES + 1):
            MUs[i] = train_audio[i][np.random.randint(1, len(train_audio[i]), M)]  # Počiatočna stredná hodnota
            # COVs[i] = [np.cov(train[i].T)] * M  # Počiatočna kovariančná matica
            COVs[i] = [np.diag(np.diag(np.cov(train_audio[i].T))) for _ in
                       range(M)]  # Initial diagonal covariance matrix
            Ws[i] = np.ones(M) / M

        for jj in range(EPOCH):
            # TTL_t je doveryhodnosť
            for i in range(1, self.CLASSES + 1):
                Ws[i], MUs[i], COVs[i], TTL = ilib.train_gmm(train_audio[i], Ws[i], MUs[i], COVs[i])
                print(f'Iteration: {jj} Total log likelihood: {TTL} for person {i}')

        return Ws, MUs, COVs

    def eval(self, dev_audio, Ws, MUs, COVs, eval_format='old'):
        correct = 0
        total = 0

        predicted_classes = []
        files = []
        if eval_format == 'old':
            for true_class in range(1, self.CLASSES + 1):
                print("Proccessing class")
                for dev_p_i in dev_audio[true_class]:
                    dev_p_i_cpy = dev_p_i.copy()

                    if self.coefficients_normalization:
                        dev_p_i_cpy = Audio.min_max_normalize(dev_p_i_cpy)

                    if self.delta_coefficients_enabled:
                        test_t_delta_coeffs = Audio.compute_deltas(dev_p_i_cpy, window_size=2)
                        test_t_derivative_delta_coeffs = Audio.compute_deltas(test_t_delta_coeffs, window_size=2)

                        dev_p_i_cpy = np.concatenate((dev_p_i_cpy, test_t_delta_coeffs, test_t_derivative_delta_coeffs),
                                                     axis=1)

                    if self.cepstral_mean_subtraction_enabled:
                        dev_p_i_cpy = Audio.cepstral_mean_subtraction(dev_p_i_cpy)

                    # Compute the likelihoods for all the classes
                    likelihoods = np.array(
                        [ilib.logpdf_gmm(dev_p_i_cpy, Ws[i], MUs[i], COVs[i]).sum() for i in range(1, 32)])

                    # Find the class with the highest likelihood
                    predicted_class = np.argmax(likelihoods) + 1
                    predicted_classes.append(predicted_class)

                    # Compare the predicted class with the true class
                    if predicted_class == true_class:
                        correct += 1
                    total += 1
        elif eval_format == 'new':
            for key in dev_audio:
                dev_p_i_cpy = dev_audio[key].copy()

                if self.coefficients_normalization:
                    dev_p_i_cpy = Audio.min_max_normalize(dev_p_i_cpy)

                if self.delta_coefficients_enabled:
                    test_t_delta_coeffs = Audio.compute_deltas(dev_p_i_cpy, window_size=2)
                    test_t_derivative_delta_coeffs = Audio.compute_deltas(test_t_delta_coeffs, window_size=2)

                    dev_p_i_cpy = np.concatenate((dev_p_i_cpy, test_t_delta_coeffs, test_t_derivative_delta_coeffs),
                                                 axis=1)

                if self.cepstral_mean_subtraction_enabled:
                    dev_p_i_cpy = Audio.cepstral_mean_subtraction(dev_p_i_cpy)

                likelihoods = np.array(
                    [ilib.logpdf_gmm(dev_p_i_cpy, Ws[i], MUs[i], COVs[i]).sum() for i in range(1, 32)])

                # Store the probabilities for each class
                probabilities = np.exp(likelihoods - logsumexp(likelihoods))
                # predicted_classes.append(probabilities + 1e-10)
                predicted_classes.append(probabilities)
                files.append(key)

            # Compute the average probability for each class
            probabilities_matrix = np.vstack(predicted_classes)
            average_probabilities = probabilities_matrix.mean(axis=0)
            return predicted_classes, files

        accuracy = correct / total
        print(f"Fraction of correctly recognized targets: {accuracy * 100}%")
        return predicted_classes, accuracy