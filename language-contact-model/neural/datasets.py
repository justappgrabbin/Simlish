"""Fetch attributed public data, checking normalized content against recorded hashes."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
URLS={
 'data/gesture/a1_raw.csv':'https://raw.githubusercontent.com/srdjop/Gesture-Phase-Segmentation/main/Dataset/a1_raw.csv',
 'data/gesture/a2_raw.csv':'https://raw.githubusercontent.com/srdjop/Gesture-Phase-Segmentation/main/Dataset/a2_raw.csv',
 'data/gesture/a3_raw.csv':'https://raw.githubusercontent.com/srdjop/Gesture-Phase-Segmentation/main/Dataset/a3_raw.csv',
 'data/speech/JapaneseVowels_TRAIN.ts':'https://raw.githubusercontent.com/aeon-toolkit/aeon/main/aeon/datasets/data/JapaneseVowels/JapaneseVowels_TRAIN.ts',
 'data/speech/JapaneseVowels_TEST.ts':'https://raw.githubusercontent.com/aeon-toolkit/aeon/main/aeon/datasets/data/JapaneseVowels/JapaneseVowels_TEST.ts',
 'data/sign/movement_libras.data':'https://raw.githubusercontent.com/IST-DASLab/QRGD/master/movement_libras.data',
}


def fetch():
    manifest=json.loads((ROOT/'data'/'SHA256.json').read_text())
    for entry in manifest:
        relative=entry['path']; path=ROOT/relative
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']:
            print('Verified '+relative,flush=True); continue
        print('Downloading '+relative,flush=True)
        with urllib.request.urlopen(URLS[relative],timeout=20) as response:
            raw=response.read(5_000_001)
        if len(raw)>5_000_000: raise ValueError('Unexpectedly large source file')
        data=raw.decode('utf-8').replace('\r\n','\n').rstrip('\n').encode('utf-8')+b'\n'
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise ValueError('Source changed; refusing unverified data: '+relative)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.download')
        temporary.write_bytes(data); temporary.replace(path)
    print('Public datasets ready. Attribution: data/README.md',flush=True)

if __name__=='__main__': fetch()
