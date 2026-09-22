import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

sr, data = wavfile.read('D:/openS/lofi-chatbot/generated/1min.wav')
data = data.astype(np.float64) / 32767.0
left = data[:, 0]
right = data[:, 1]

def band_pct(sig, sr, lo, hi):
    freqs, psd = welch(sig, sr, nperseg=8192)
    mask = (freqs >= lo) & (freqs < hi)
    return np.sum(psd[mask]) / np.sum(psd) * 100

print('=== FREQUENCY BAND ENERGY ===')
for name, lo, hi in [('Sub+Bass 0-250Hz', 0, 250),
                      ('Low-mid 250-500Hz', 250, 500),
                      ('Mid 500-2kHz', 500, 2000),
                      ('High-mid 2-6kHz', 2000, 6000),
                      ('High 6-12kHz', 6000, 12000),
                      ('Air 12kHz+', 12000, 20000)]:
    el = band_pct(left, sr, lo, hi)
    er = band_pct(right, sr, lo, hi)
    print(f'  {name:25s} L={el:5.1f}%  R={er:5.1f}%')

# Combined bands
low = (band_pct(left, sr, 0, 250) + band_pct(right, sr, 0, 250)) / 2
highmid_high_air = (band_pct(left, sr, 2000, 20000) + band_pct(right, sr, 2000, 20000)) / 2
print(f'\nSub+bass total: {low:.1f}% (target: 35-40%)')
print(f'High-mid+high+air total: {highmid_high_air:.1f}% (target: >=15%)')

# Spectral centroid movement
window = sr * 10
centroids = []
for i in range(0, len(left) - window, window):
    chunk = left[i:i+window]
    freqs, psd = welch(chunk, sr, nperseg=4096)
    c = np.sum(freqs * psd) / np.sum(psd)
    centroids.append(c)
centroids = np.array(centroids)
print(f'\n=== SPECTRAL CENTROID ===')
print(f'  Mean: {centroids.mean():.0f}Hz  Std: {centroids.std():.0f}Hz (target: >300Hz)')
print(f'  Min: {centroids.min():.0f}Hz  Max: {centroids.max():.0f}Hz')

# RMS envelope
rms_windows = []
for i in range(0, len(left) - sr, sr):
    chunk = left[i:i+sr]
    rms_windows.append(np.sqrt(np.mean(chunk**2)))
rms_windows = np.array(rms_windows)
print(f'\n=== RMS ENVELOPE ===')
print(f'  Mean: {rms_windows.mean():.4f}  Std: {rms_windows.std():.4f} (target: >0.008)')
print(f'  Min: {rms_windows.min():.4f}  Max: {rms_windows.max():.4f}')

# L/R correlation
lr_corr = np.corrcoef(left, right)[0, 1]
print(f'\n=== STEREO ===')
print(f'  L/R correlation: {lr_corr:.3f} (target: <0.9)')
