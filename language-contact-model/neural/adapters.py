"""Explicit input adapters: preserve samples/time; no pretrained recognition claims."""
import wave
import numpy as np


def text_events(text, channel='text'):
    # Bytes form an open UTF-8 input alphabet, without a word vocabulary cutoff.
    result=[]
    for value in text.encode('utf-8'):
        vector=np.zeros(256); vector[value]=1
        result.append({'observations':{channel:vector.tolist()}})
    return result


def pcm_wav_events(path, channel='audio', frame_samples=256):
    """16-bit PCM mono/stereo WAV to ordered raw waveform frames.

    A trainable model encoder consumes the samples directly; this adapter does not
    transcribe speech or infer emotions. Metadata retains sample timing.
    """
    if frame_samples < 1: raise ValueError('Positive frame size required')
    with wave.open(str(path),'rb') as audio:
        if audio.getsampwidth()!=2 or audio.getcomptype()!='NONE':
            raise ValueError('Uncompressed 16-bit PCM WAV required')
        rate=audio.getframerate(); channels=audio.getnchannels()
        samples=np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2').astype(float)
        samples=samples.reshape(-1,channels).mean(1)/32768.
    return [{'observations':{channel:np.pad(samples[i:i+frame_samples],(0,max(0,frame_samples-len(samples[i:i+frame_samples])))).tolist()},
             'timestamp':i/rate,'sample_rate':rate,'valid_samples':min(frame_samples,len(samples)-i)}
            for i in range(0,len(samples),frame_samples)]


def landmark_events(frames, channel='sign', timestamps=None):
    """Preserve ordered joint/landmark coordinates, including spatial geometry."""
    x=np.asarray(frames,dtype=float)
    if x.ndim<2 or not np.isfinite(x).all(): raise ValueError('Finite temporal landmark array required')
    if timestamps is not None and len(timestamps)!=len(x): raise ValueError('Timestamp count mismatch')
    return [{'observations':{channel:frame.reshape(-1).tolist()},
             **({'timestamp':float(timestamps[i])} if timestamps is not None else {})}
            for i,frame in enumerate(x)]
