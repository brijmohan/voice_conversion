import h5py
import numpy as np
import sys
import os
import glob
import re
from collections import defaultdict
#from tacotron.audio import load_wav, spectrogram, melspectrogram
from tacotron.norm_utils import get_spectrograms 


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python3 make_dataset_voxceleb2.py [data root directory (Voxceleb2-Corpus)] [h5py path] '
                '[training proportion]')
        exit(0)

    root_dir = sys.argv[1]
    h5py_path = sys.argv[2]
    proportion = float(sys.argv[3])

    filename_groups = defaultdict(lambda : [])
    with h5py.File(h5py_path, 'w') as f_h5:
        filenames = sorted(glob.glob(os.path.join(root_dir, 'french/*/*/*/*.wav')))
        for filename in filenames:
            # divide into groups
            filename_sp = filename.strip().split('/')
            speaker_id = filename_sp[-3]
            utt_id = filename_sp[-2] + '_' + filename_sp[-1].split('.')[0]
            # format: p{speaker}_{sid}.wav
            #speaker_id, utt_id = re.search(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
            filename_groups[speaker_id].append(filename)
        for speaker_id, filenames in filename_groups.items():
            print('processing {}'.format(speaker_id))
            train_size = int(len(filenames) * proportion)
            for i, filename in enumerate(filenames):
                filename_sp = filename.strip().split('/')
                speaker_id = filename_sp[-3]
                utt_id = filename_sp[-2] + '_' + filename_sp[-1].split('.')[0]
                _, lin_spec = get_spectrograms(filename)
                if i < train_size:
                    datatype = 'train'
                else:
                    datatype = 'test'
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}', \
                    data=lin_spec, dtype=np.float32)
