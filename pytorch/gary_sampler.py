import argparse
import json
import os
import random
import torch
import torch.utils.data
import sys
import audio as deepaudio
from hparams import hparams

import utils

class DeepMels(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.

    This uses r9r9's deepvoice preprocessing to create mel spectrogram.
    """
    def __init__(self, training_files, segment_length, mu_quantization,
                 filter_length, hop_length, win_length, sampling_rate):
        audio_files = utils.files_to_list(training_files)
        self.audio_files = audio_files
        random.seed(1234)
        random.shuffle(self.audio_files)
        
        self.segment_length = segment_length
        self.mu_quantization = mu_quantization
        self.sampling_rate = sampling_rate
    
    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        wav = deepaudio.load_wav(filename)
        # load in raw_audio via utils
        raw_audio, _ = utils.load_wav_to_torch(filename)
        # convert wav to numpy
        audio = torch.from_numpy(wav)
        # take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
            # update raw audio as well
            raw_audio = raw_audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
            # pad raw audio as well
            raw_audio = torch.nn.functional.pad(raw_audio, (0, self.segment_length - raw_audio.size(0)), 'constant').data
        # compute mel
        mel = deepaudio.melspectrogram(audio.numpy())
        # convert mel to torch
        mel = torch.from_numpy(mel)
        audio = utils.mu_law_encode(raw_audio / utils.MAX_WAV_VALUE, self.mu_quantization)
        return (mel, audio)
    
    def __len__(self):
        return len(self.audio_files)


if __name__ == "__main__":
    """
    Turns audio files into mel-spectrogram representations for inference

    Uses the data portion of the config for audio processing parameters, 
    but ignores training files and segment lengths.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--audio_list", required=True, type=str,
                        help='File containing list of wavefiles')
    parser.add_argument('-o', "--output_dir", required=True, type=str,
                        help='Directory to put Mel-Spectrogram Tensors')
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    
    args = parser.parse_args()

    filepaths = utils.files_to_list(args.audio_list)
    
    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)
    
    # Parse config.  Only using data processing
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    data_config = config["data_config"]
    mel_factory = Mel2SampOnehot(**data_config)  
    
    for filepath in filepaths:
        audio, sampling_rate = utils.load_wav_to_torch(filepath)
        assert(sampling_rate == mel_factory.sampling_rate)
        melspectrogram = mel_factory.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)