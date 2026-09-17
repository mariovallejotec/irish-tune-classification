import csv
import random
from augment_abc import transpose_abc_text

random.seed(42)

with open("train.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

fieldnames = ["tune_id", "setting_id", "name", "type", "meter", "mode", "abc"]
out_rows = list(rows)  # originales
ok, fail = 0, 0
total = len(rows)

for i, row in enumerate(rows, 1):
    semitones = random.choice([-3, -2, -1, 1, 2, 3])
    result = transpose_abc_text(row, semitones)
    if result is None:
        fail += 1
    else:
        new_abc_body, new_key = result
        new_row = dict(row)
        new_row["mode"] = new_key
        new_row["abc"] = new_abc_body
        new_row["setting_id"] = row["setting_id"] + f"_t{semitones}"
        out_rows.append(new_row)
        ok += 1
    if i % 2000 == 0 or i == total:
        pct = 100 * i / total
        print(f"PROGRESS: {i}/{total} ({pct:.1f}%) ok={ok} fail={fail}", flush=True)

with open("train_augmented.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in out_rows:
        w.writerow({k: row[k] for k in fieldnames})

print(f"originales: {len(rows)}, transpuestas ok: {ok}, fail: {fail}, total: {len(out_rows)}")
