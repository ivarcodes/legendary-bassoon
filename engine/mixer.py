"""Multi-track mixer - mixes 3 tracks with crossfade transitions."""

import numpy as np
from .core import SR, seconds_to_samples


def mix_tracks(tracks, total_duration, crossfade_sec=5.0):
    """Mix multiple tracks with crossfade transitions.
    
    Args:
        tracks: list of (samples, sr) tuples
        total_duration: total output duration in seconds
        crossfade_sec: crossfade duration between tracks
    
    Returns:
        mixed samples
    """
    num_tracks = len(tracks)
    if num_tracks == 0:
        return np.zeros(seconds_to_samples(total_duration))
    
    if num_tracks == 1:
        return tracks[0][0][:seconds_to_samples(total_duration)]
    
    # Calculate duration per track with overlap for crossfade
    # Total = (track_dur * num) - (crossfade * (num-1))
    # So track_dur = (total + crossfade * (num-1)) / num
    track_duration = (total_duration + crossfade_sec * (num_tracks - 1)) / num_tracks
    
    target_len = seconds_to_samples(total_duration)
    crossfade_len = seconds_to_samples(crossfade_sec)
    track_len = seconds_to_samples(track_duration)
    
    output = np.zeros(target_len, dtype=np.float64)
    
    for i, (samples, sr) in enumerate(tracks):
        # Trim or pad track to track_duration
        if len(samples) < track_len:
            track = np.pad(samples, (0, track_len - len(samples)))
        else:
            track = samples[:track_len]
        
        # Calculate position in output
        # Each track starts track_duration - crossfade after previous
        start_sample = int(i * (track_duration - crossfade_sec) * SR)
        end_sample = start_sample + track_len
        
        if start_sample >= target_len:
            break
        
        # Apply fade in (except first track)
        if i > 0:
            fade_in = min(crossfade_len, len(track))
            track[:fade_in] *= np.linspace(0, 1, fade_in)
        
        # Apply fade out (except last track)
        if i < num_tracks - 1:
            fade_out = min(crossfade_len, len(track))
            track[-fade_out:] *= np.linspace(1, 0, fade_out)
        
        # Add to output
        actual_end = min(end_sample, target_len)
        actual_len = actual_end - start_sample
        output[start_sample:actual_end] += track[:actual_len]
    
    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0.01:
        output = output / peak * 0.85
    
    return output
