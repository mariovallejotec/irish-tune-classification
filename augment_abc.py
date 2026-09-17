"""
Key-transposition augmentation for ABC-notation training data, following
Navarro-Caceres et al. (DITTET 2025), who report that symbolic key
transposition outperformed audio-based augmentation for this exact task
(Irish traditional music genre classification). Applied ONLY to train.csv
(val/test stay untouched, to keep evaluation honest).
"""
import csv
import random
import warnings
from music21 import converter, note, stream

warnings.filterwarnings("ignore")
random.seed(42)

L_UNIT = 0.5  # L:1/8 -> one L unit = eighth note = 0.5 quarterLength


def build_abc(row):
    return f"X:1\nT:{row['name']}\nM:{row['meter']}\nL:1/8\nK:{row['mode']}\n{row['abc']}"


def duration_to_abc_len(qlen):
    ratio = qlen / L_UNIT
    if abs(ratio - 1) < 1e-6:
        return ""
    if abs(ratio - 0.5) < 1e-6:
        return "/"
    if abs(ratio - round(ratio)) < 1e-6:
        return str(int(round(ratio)))
    # fallback: nearest simple fraction n/d, d in {2,3,4}
    from fractions import Fraction
    frac = Fraction(ratio).limit_denominator(4)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def pitch_to_abc(p):
    step = p.step  # 'C','D',...
    octave = p.octave
    acc = ""
    if p.accidental is not None:
        a = p.accidental.alter
        if a == 1:
            acc = "^"
        elif a == 2:
            acc = "^^"
        elif a == -1:
            acc = "_"
        elif a == -2:
            acc = "__"
        elif a == 0:
            acc = "="
    # ABC convention: octave 5 (C5-B5) = uppercase C..B (music21 middle C = C4)
    if octave >= 5:
        letter = step.lower()
        marks = "'" * (octave - 5)
    else:
        letter = step
        marks = "," * (4 - octave)
    return acc + letter + marks


def transpose_abc_text(row, semitones):
    """Devuelve el texto ABC (solo cuerpo, sin header) transpuesto, o None si falla."""
    try:
        s = converter.parse(build_abc(row), format="abc")
        s2 = s.transpose(semitones)
        s2.makeAccidentals(inPlace=True, overrideStatus=True)
        notes = [n for n in s2.flatten().notesAndRests
                 if isinstance(n, (note.Note, note.Rest))]
        if len(notes) < 4:
            return None
        tokens = []
        for n in notes:
            if n.isRest:
                tokens.append("z" + duration_to_abc_len(n.duration.quarterLength))
            else:
                tokens.append(pitch_to_abc(n.pitch) + duration_to_abc_len(n.duration.quarterLength))
        new_key = s2.analyze("key")
        return " ".join(tokens), new_key.tonic.name.replace("-", "b") + new_key.mode
    except Exception:
        return None


if __name__ == "__main__":
    # --- prueba de round-trip en 5 filas antes de correr todo ---
    with open("train.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sample = rows[:5]
    for row in sample:
        result = transpose_abc_text(row, 2)
        if result is None:
            print("FAIL parse/transpose:", row["name"])
            continue
        new_abc_body, new_key = result
        # round-trip: re-parse el texto generado y comparar midi pitches
        header = f"X:1\nT:test\nM:{row['meter']}\nL:1/8\nK:{new_key}\n"
        try:
            s3 = converter.parse(header + new_abc_body, format="abc")
            n_notes = len([n for n in s3.flatten().notes])
            print(f"OK {row['name']!r} -> key={new_key}, notas regeneradas={n_notes}")
        except Exception as e:
            print(f"FAIL round-trip {row['name']!r}: {repr(e)[:150]}")
