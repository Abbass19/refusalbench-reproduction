"""After you fill the human_ok column (y/n) in handcheck_sheet.csv, this prints the judge agreement rate."""
import csv
import sys

path = sys.argv[1]
rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["human_ok (y/n)"].strip()]
if not rows:
    sys.exit("No rows filled in yet (write y or n in the human_ok column).")
ok = sum(r["human_ok (y/n)"].strip().lower().startswith("y") for r in rows)
print(f"judge agreement with you: {ok}/{len(rows)} = {ok / len(rows):.0%}")
