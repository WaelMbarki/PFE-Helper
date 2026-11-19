"""
CV -> parsed profile -> eligible offers -> ranked offer ids

Run:
    python match_cv.py "path/to/cv.pdf"
"""

import json
import sys
from pathlib import Path

import pandas as pd

from parserfinal import parse_cv
from eligible import check_eligibility
from matcher import rank_offers

JOBS_CSV = Path(__file__).with_name("jobs_cleaned.csv")


def main() -> None:
    cv_path = sys.argv[1] if len(sys.argv) > 1 else input("Path to the CV: ").strip().strip('"')

    # 1. Parse the CV and load the saved profile
    profile_path = parse_cv(cv_path)
    candidate = json.loads(profile_path.read_text(encoding="utf-8"))

    # 2. Load the offers (utf-8-sig: the file starts with a hidden BOM character)
    offers = pd.read_csv(JOBS_CSV, encoding="utf-8-sig")

    # 3. Keep only the offers the student is allowed to apply to
    eligible = []
    for index, offer in offers.iterrows():
        result = check_eligibility(candidate, offer)
        if result["status"] != "not_eligible":
            eligible.append(offer)
    # 4. Rank them and print the ids
    ids = rank_offers(candidate, eligible)
    print(ids)


if __name__ == "__main__":
    main()