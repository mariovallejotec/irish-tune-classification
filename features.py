"""
Extracción de features melódicas y rítmicas para el baseline SVM,
siguiendo las categorías descritas en Kermit-Canfield & Roman (CS229 2015,
"Dance Type Classification in Irish and Scandinavian Folk Music"):
  - Melódicas: histogramas de intervalo melódico y pitch-class, dirección
    melódica/arco, cadencias, notas repetidas.
  - Rítmicas: time signature, duración de nota, tiempo entre notas
    consecutivas (IOI) y su variabilidad.
El paper no publica el código ni la lista exacta de las 65 features
originales, así que esta es una reconstrucción razonable de esas categorías
usando music21, no una copia literal.
"""
import warnings
import numpy as np
from music21 import converter

warnings.filterwarnings("ignore")

PITCH_CLASSES = 12
# intervalos melódicos de -12 a +12 semitonos (unísono al centro)
INTERVAL_RANGE = 12


def build_abc(row):
    return f"X:1\nT:{row['name']}\nM:{row['meter']}\nL:1/8\nK:{row['mode']}\n{row['abc']}"


def extract_features(row):
    """Devuelve un vector de features (np.array) o None si falla el parseo."""
    try:
        s = converter.parse(build_abc(row), format="abc")
        notes = list(s.flatten().notes)
        notes = [n for n in notes if n.isNote]
        if len(notes) < 4:
            return None
    except Exception:
        return None

    pitches = [n.pitch.midi for n in notes]
    durations = [float(n.duration.quarterLength) for n in notes]

    # --- melódicas ---
    pc_hist = np.zeros(PITCH_CLASSES)
    for p in pitches:
        pc_hist[p % 12] += 1
    pc_hist /= pc_hist.sum()

    intervals = np.diff(pitches)
    interval_hist = np.zeros(2 * INTERVAL_RANGE + 1)
    for iv in intervals:
        iv_clamped = max(-INTERVAL_RANGE, min(INTERVAL_RANGE, iv))
        interval_hist[iv_clamped + INTERVAL_RANGE] += 1
    if len(intervals) > 0:
        interval_hist /= interval_hist.sum()

    pct_ascending = float(np.mean(intervals > 0)) if len(intervals) else 0.0
    pct_descending = float(np.mean(intervals < 0)) if len(intervals) else 0.0
    pct_repeated_pitch = float(np.mean(intervals == 0)) if len(intervals) else 0.0
    pitch_range = float(max(pitches) - min(pitches))
    pitch_mean = float(np.mean(pitches))
    pitch_std = float(np.std(pitches))
    n_distinct_pc = float(len(set(p % 12 for p in pitches)))
    final_note_pc = float(pitches[-1] % 12)
    first_note_pc = float(pitches[0] % 12)

    # --- rítmicas ---
    dur_bins = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    dur_hist = np.zeros(len(dur_bins) + 1)
    for d in durations:
        placed = False
        for i, b in enumerate(dur_bins):
            if abs(d - b) < 1e-6:
                dur_hist[i] += 1
                placed = True
                break
        if not placed:
            dur_hist[-1] += 1
    dur_hist /= dur_hist.sum()

    dur_mean = float(np.mean(durations))
    dur_std = float(np.std(durations))
    n_notes = float(len(notes))

    try:
        meter_num, meter_den = row["meter"].split("/")
        meter_num, meter_den = float(meter_num), float(meter_den)
    except Exception:
        meter_num, meter_den = 4.0, 4.0

    feats = np.concatenate([
        pc_hist,                       # 12
        interval_hist,                 # 25
        [pct_ascending, pct_descending, pct_repeated_pitch,
         pitch_range, pitch_mean, pitch_std, n_distinct_pc,
         final_note_pc, first_note_pc],  # 9
        dur_hist,                      # 9
        [dur_mean, dur_std, n_notes, meter_num, meter_den],  # 5
    ])
    return feats


FEATURE_NAMES = (
    [f"pc_{i}" for i in range(PITCH_CLASSES)]
    + [f"interval_{i - INTERVAL_RANGE}" for i in range(2 * INTERVAL_RANGE + 1)]
    + ["pct_ascending", "pct_descending", "pct_repeated_pitch",
       "pitch_range", "pitch_mean", "pitch_std", "n_distinct_pc",
       "final_note_pc", "first_note_pc"]
    + [f"dur_bin_{i}" for i in range(9)]
    + ["dur_mean", "dur_std", "n_notes", "meter_num", "meter_den"]
)
