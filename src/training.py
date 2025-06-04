import torchaudio
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as transforms
import random


class AudioUtil:

    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return sig, sr

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud
        if sig.shape[0] == new_channel:
            return aud
        if new_channel == 1:
            resig = sig[:1, :]
        else:
            resig = sig.repeat(new_channel, 1)
        return resig, sr

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud
        if sr == newsr:
            return aud
        num_channels = sig.shape[0]
        resig = transforms.Resample(sr, newsr)(sig[:1, :])
        if num_channels > 1:
            retwo = transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo], dim=0)
        return resig, newsr

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms
        if sig_len > max_len:
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), dim=1)
        return sig, sr

    @staticmethod
    def time_shift(aud, shift_pct):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_pct * sig_len)
        return sig.roll(shift_amt), sr

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db=80)(spec)
        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        freq_mask_param = int(max_mask_pct * n_mels)
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
        time_mask_param = int(max_mask_pct * n_steps)
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec

    @staticmethod
    def add_noise(aud, noise_level=0.005):
        sig, sr = aud
        noise = torch.randn_like(sig) * noise_level
        return sig + noise, sr

    @staticmethod
    def change_gain(aud, gain_db_range=(-6, 6)):
        sig, sr = aud
        gain_db = random.uniform(*gain_db_range)
        gain = 10 ** (gain_db / 20)
        return sig * gain, sr

    @staticmethod
    def pitch_shift(aud, n_steps=2):
        sig, sr = aud
        return torchaudio.functional.pitch_shift(sig, sr, n_steps), sr

class SoundDS(Dataset):
    def __init__(self, data_path, label, mode="original"):
        self.label = label
        self.data_path = [str(p) for p in data_path]
        self.duration = 4000
        self.sr = 16000
        self.channel = 2
        self.shift_pct = 0.4
        self.mode = mode

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        audio_file = self.data_path[idx]
        class_id = self.label[idx]
        
        aud = AudioUtil.open(audio_file)
        aud = AudioUtil.resample(aud, self.sr)
        aud = AudioUtil.rechannel(aud, self.channel)
        aud = AudioUtil.pad_trunc(aud, self.duration)

        if self.mode == "time_shift":
            aud = AudioUtil.time_shift(aud, self.shift_pct)
        elif self.mode == "add_noise":
            aud = AudioUtil.add_noise(aud, noise_level=random.uniform(0.005, 0.05))
        elif self.mode == "pitch_shift":
            aud = AudioUtil.pitch_shift(aud, n_steps=random.randint(1,3))
        elif self.mode == "combined":
            aud = AudioUtil.time_shift(aud, self.shift_pct)
            aud = AudioUtil.add_noise(aud, noise_level=random.uniform(0.005, 0.05))

        sgram = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=780, hop_len=195)

        if self.mode in ["spectro_augment", "combined"]:
            sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=random.randint(1,3), n_time_masks=random.randint(1,3))

        return sgram, class_id