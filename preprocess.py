"""
Preprocesa tunes.csv (TheSession-data) para clasificación de tune-type.
Split estratificado por tune_id (no por setting_id) para evitar que dos
settings del mismo tune caigan en splits distintos (fuga de datos).
"""
import csv
import random
from collections import defaultdict

random.seed(42)

SRC = "tunes.csv"
TRAIN_OUT = "train.csv"
VAL_OUT = "val.csv"
TEST_OUT = "test.csv"

SPLIT = (0.7, 0.15, 0.15)  # train, val, test

def load_rows():
    rows = []
    with open(SRC, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def main():
    rows = load_rows()

    # agrupar tune_id -> tipo (un tune puede tener varios settings, mismo tipo)
    tune_to_type = {}
    tune_to_rows = defaultdict(list)
    for row in rows:
        tid = row["tune_id"]
        tune_to_type[tid] = row["type"]
        tune_to_rows[tid].append(row)

    # agrupar tune_ids por tipo para split estratificado
    type_to_tunes = defaultdict(list)
    for tid, t in tune_to_type.items():
        type_to_tunes[t].append(tid)

    train_ids, val_ids, test_ids = set(), set(), set()
    for t, tids in type_to_tunes.items():
        tids = tids[:]
        random.shuffle(tids)
        n = len(tids)
        n_train = int(n * SPLIT[0])
        n_val = int(n * SPLIT[1])
        train_ids.update(tids[:n_train])
        val_ids.update(tids[n_train:n_train + n_val])
        test_ids.update(tids[n_train + n_val:])

    def dump(ids, path):
        out_rows = []
        for tid in ids:
            out_rows.extend(tune_to_rows[tid])
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tune_id", "setting_id", "name", "type", "meter", "mode", "abc"])
            w.writeheader()
            for row in out_rows:
                w.writerow({k: row[k] for k in w.fieldnames})
        return len(out_rows)

    n_train = dump(train_ids, TRAIN_OUT)
    n_val = dump(val_ids, VAL_OUT)
    n_test = dump(test_ids, TEST_OUT)

    print(f"tunes: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"settings (filas): train={n_train} val={n_val} test={n_test}")

if __name__ == "__main__":
    main()
