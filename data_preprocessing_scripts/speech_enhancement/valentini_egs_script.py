# This source code is licensed under the license found in the
# LICENSE-MIT.txt file in the root directory of this source tree.
import json
from collections import namedtuple
from pathlib import Path
import torchaudio
import os
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project_dir")
parser.add_argument("--dataset_base_dir")
parser.add_argument("--include_debug", default=True, type=bool, required=False)
parser.add_argument("--spk", default=28, type=int, required=False)
args = parser.parse_args()


Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def find_audio_files(path, exts=[".wav"], progress=True, poi:list=None):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    dev_meta = []
    for idx, file in enumerate(audio_files):
        flag = False
        if poi is not None:
            for p in poi:
                if p in file:
                    flag = True
                    break

        info = get_info(file)
        if flag:
            dev_meta.append((file, info.length))
        else:
            meta.append((file, info.length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    if poi is not None:
        dev_meta.sort()
        return meta, dev_meta
    return meta


if __name__ == "__main__":
    train_clean, val_clean = find_audio_files(f"{args.dataset_base_dir}/clean_trainset_{args.spk}spk_wav", poi=['p286', 'p287'])
    train_noisy, val_noisy = find_audio_files(f"{args.dataset_base_dir}/noisy_trainset_{args.spk}spk_wav", poi=['p286', 'p287'])
    test_noisy = find_audio_files(f"{args.dataset_base_dir}/noisy_testset_wav")
    test_clean = find_audio_files(f"{args.dataset_base_dir}/clean_testset_wav")

    # create jsons
    test_clean_json = json.dumps(test_clean, indent=4)
    test_noisy_json = json.dumps(test_noisy, indent=4)
    train_clean_json = json.dumps(train_clean, indent=4)
    train_noisy_json = json.dumps(train_noisy, indent=4)
    val_clean_json = json.dumps(val_clean, indent=4)
    val_noisy_json = json.dumps(val_noisy, indent=4)

    # save files
    os.makedirs(f"{args.project_dir}/egs/valentini/tr", exist_ok=True)
    os.makedirs(f"{args.project_dir}/egs/valentini/cv", exist_ok=True)
    os.makedirs(f"{args.project_dir}/egs/valentini/tt", exist_ok=True)
    with open(f"{args.project_dir}/egs/valentini/tr/clean.json", 'w') as f:
        f.write(train_clean_json)
    with open(f"{args.project_dir}/egs/valentini/tr/noisy.json", 'w') as f:
        f.write(train_noisy_json)
    with open(f"{args.project_dir}/egs/valentini/cv/clean.json", 'w') as f:
        f.write(val_clean_json)
    with open(f"{args.project_dir}/egs/valentini/cv/noisy.json", 'w') as f:
        f.write(val_noisy_json)
    with open(f"{args.project_dir}/egs/valentini/tt/clean.json", 'w') as f:
        f.write(test_clean_json)
    with open(f"{args.project_dir}/egs/valentini/tt/noisy.json", 'w') as f:
        f.write(test_noisy_json)

    if args.include_debug:
        # create jsons
        test_clean_json = json.dumps(test_clean[:2], indent=4)
        test_noisy_json = json.dumps(test_noisy[:2], indent=4)
        train_clean_json = json.dumps(test_clean[:2], indent=4)
        train_noisy_json = json.dumps(test_noisy[:2], indent=4)
        val_clean_json = json.dumps(test_clean[:2], indent=4)
        val_noisy_json = json.dumps(test_noisy[:2], indent=4)

        # save files
        os.makedirs(f"{args.project_dir}/egs/debug/tr", exist_ok=True)
        os.makedirs(f"{args.project_dir}/egs/debug/cv", exist_ok=True)
        os.makedirs(f"{args.project_dir}/egs/debug/tt", exist_ok=True)
        with open(f"{args.project_dir}/egs/debug/tr/clean.json", 'w') as f:
            f.write(train_clean_json)
        with open(f"{args.project_dir}/egs/debug/tr/noisy.json", 'w') as f:
            f.write(train_noisy_json)
        with open(f"{args.project_dir}/egs/debug/cv/clean.json", 'w') as f:
            f.write(val_clean_json)
        with open(f"{args.project_dir}/egs/debug/cv/noisy.json", 'w') as f:
            f.write(val_noisy_json)
        with open(f"{args.project_dir}/egs/debug/tt/clean.json", 'w') as f:
            f.write(test_clean_json)
        with open(f"{args.project_dir}/egs/debug/tt/noisy.json", 'w') as f:
            f.write(test_noisy_json)

