from scipy.io.wavfile import read, write
import numpy as np
from IPython.display import Audio
import os
from tqdm import tqdm

def split_wav(wav):
    """split wavs into 1 second clips
    
    """
    # see how many seconds it has
    seconds = wav.shape[0] / 22050
    if seconds > 1.0:
        split_wav = np.split(wav[:int(seconds)*22050], int(seconds))
        return split_wav
    else:
        return None

def split_all(wav_dir, output_dir):
    wav_list = os.listdir(wav_dir)
    count = 0
    
    for wav_file in tqdm(wav_list):
        # read wav
        _, wav = read(wav_dir + wav_file)
        # split
        split = split_wav(wav)
        if split is not None:
            for each_wav in split:
                # hack for file name
                if count < 10:
                    f_path = "00000{}.wav".format(count)
                elif count < 100:
                    f_path = "0000{}.wav".format(count)
                elif count < 1000:
                    f_path = "000{}.wav".format(count)
                elif count < 10000:
                    f_path = "00{}.wav".format(count)
                elif count < 100000:
                    f_path = "0{}.wav".format(count)
                    
                write(output_dir+f_path, 22050, each_wav.astype(np.int16))
                count += 1
                
    print("initial number of wavs:{}, after expansion:{}".format(len(wav_list), count))


if __name__=="__main__":
    wav_dir = "somedir"
    output_dir = "somedir"
    split_all(wav_dir, output_dir)
                
        
    