import cv2
import numpy as np

def convert_spectrogram_to_heatmap(spectrogram):
    spectrogram = (255 * (spectrogram - np.min(spectrogram)) / np.ptp(spectrogram)).astype(np.uint8).squeeze()
    heatmap = cv2.applyColorMap(spectrogram, cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap