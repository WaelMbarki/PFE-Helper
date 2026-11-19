"""
Eligible offers
---------------
Takes a parsed CV profile and returns the ids of the offers the student
is allowed to apply to. It does not read CVs and does not need Ollama.

Use it from other code:

    from eligible import find_eligible_offers

    ids = find_eligible_offers("parsed_cvs/cv_profile.json")   # all offers -> ids
    result = check_eligibility(profile, offer)                 # one offer -> status

Or from the terminal:

    python eligible.py parsed_cvs/cv_profile.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

JOBS_CSV = Path(__file__).with_name("jobs_cleaned.csv")

# The parser's level names -> the words used in the offers' `education_type`.
LEVEL_TO_OFFER = {
    "Bachelor": "license",
    "Master": "master",
    "Engineering": "engineering",
    "PhD": "master",
}


def is_eligible(current_level: Optional[str], accepted: list[str]) -> bool:
    """
    `accepted` is a list of "or" values, for example ["engineering", "master"].
    An offer that states no education accepts everybody. If the student's
    level is unknown, only those open offers are returned.
    """
    if not accepted or "unspecified" in accepted:
        return True
    return LEVEL_TO_OFFER.get(current_level) in accepted


def load_profile(profile: dict | str | Path) -> dict:
    """Accept a profile dict, a pydantic profile, or the path of a JSON file."""
    if isinstance(profile, (str, Path)):
        return json.loads(Path(profile).read_text(encoding="utf-8"))
    if hasattr(profile, "model_dump"):  # the parser's CandidateProfile
        return profile.model_dump(mode="json")
    return profile


def parse_education(value) -> list[str]:
    """`education_type` can be a list or a JSON string like '["master"]'."""
    if isinstance(value, list):
        return value
    try:
        accepted = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []  # missing or unreadable: treat the offer as open
    return accepted if isinstance(accepted, list) else []


def check_eligibility(candidate, offer) -> dict:
    """
    Check ONE offer. `candidate` is the parsed profile (dict, JSON path or
    profile object). `offer` is a row of the jobs file (dict or pandas row).

    Returns {"status": "eligible" | "not_eligible", "reason": "..."}
    """
    profile = load_profile(candidate)            # always a dictionary now
    education = profile.get("education") or {}   # the education block
    level = education.get("current_level")       # "Engineering", "Master"... or None
    accepted = parse_education(offer["education_type"])

    if is_eligible(level, accepted):
        return {"status": "eligible", "reason": "education matches"}
    if level is None:
        return {"status": "not_eligible",
                "reason": "student level unknown, offer requires " + ", ".join(accepted)}
    return {"status": "not_eligible",
            "reason": f"{level} is not accepted (offer asks: {', '.join(accepted)})"}


def find_eligible_offers(
    profile: dict | str | Path,
    jobs_csv: str | Path = JOBS_CSV
) -> list[int]:
    """Return the ids of all the offers the student can apply to."""
    data = load_profile(profile)

    # utf-8-sig: the jobs file starts with a hidden BOM character
    with open(jobs_csv, encoding="utf-8-sig", newline="") as file:
        return [
            int(row["id"])
            for row in csv.DictReader(file)
            if check_eligibility(data, row)["status"] != "not_eligible"
        ]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eligible.py path/to/profile.json")
        sys.exit(1)
    print(find_eligible_offers(sys.argv[1]))