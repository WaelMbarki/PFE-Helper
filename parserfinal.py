"""
CV Parser using Ollama + Qwen
--------------------------------
Reads one PDF CV (cv.pdf) and saves a structured profile as profile.json.

Install:
    pip install ollama pydantic pdfplumber

Run:
    ollama list
    ollama pull qwen3:4b
    python parserfinal.py

The parser uses layout-aware PDF extraction. The LLM interprets the extracted
content and maps it to one stable JSON schema; Python rules then clean it.
"""
from __future__ import annotations
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import pdfplumber
import ollama
from pydantic import BaseModel, ConfigDict, Field, ValidationError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Change this only if `ollama list` shows a different installed model tag.
MODEL = "qwen3:4b"

# ~16,000 chars fits in num_ctx=8192 together with the prompt and the answer.
MAX_INPUT_CHARS = 16_000

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class DiplomaItem(BaseModel):
    """A Pydantic BaseModel describes the exact shape of the data you expect"""
    model_config = ConfigDict(extra="ignore")

    # The diploma name exactly as written in the CV. It comes first so the
    # model copies the text before it picks a level.
    title: Optional[str] = None
    level: Optional[Literal["Bachelor", "Master", "Engineering", "PhD"]] = None
    status: Optional[Literal["completed", "in_progress"]] = None


class Education(BaseModel):
    model_config = ConfigDict(extra="ignore")

    diplomas: list[DiplomaItem] = Field(default_factory=list)
    field_of_study: Optional[str] = None
    year_of_study: Optional[int] = None
    current_level: Optional[str] = None  # computed in Python


class WorkItem(BaseModel):
    """A project or an internship. Only used to collect its skills."""
    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = None
    skills: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    education: Education = Field(default_factory=Education)
    skills: list[str] = Field(default_factory=list)
    # Helper fields: their skills are merged into `skills`, then removed
    # from the final JSON (see merge_skills and parse_cv).
    experience: list[WorkItem] = Field(default_factory=list)
    projects: list[WorkItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def clean_extracted_text(text: str) -> str:
    """Normalize whitespace without destroying useful line boundaries."""
    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("\uf0b7", "•")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def read_pages(pdf, layout: bool) -> str:
    """Extract the text of every page, with or without layout mode."""
    pages = []
    for page in pdf.pages:
        try:
            pages.append(page.extract_text(layout=layout, x_tolerance=2, y_tolerance=3) or "")
        except Exception:
            pages.append("")          # one bad page must not stop the whole file
    return clean_extracted_text("\n\n".join(pages))

def extract_pdf_text(path: Path) -> str:
    with pdfplumber.open(str(path)) as pdf:
        layout_text = read_pages(pdf, layout=True)
        ordinary_text = read_pages(pdf, layout=False)

    # Prefer layout extraction for CVs, except when it is clearly poorer.
    if len(ordinary_text) > len(layout_text) * 1.25:
        return ordinary_text
    return layout_text



# ---------------------------------------------------------------------------
# Prompt and LLM extraction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert CV information extraction system for Tunisian university
students. The CV may be in French or English and may use any layout.

Return ONE valid JSON object matching the schema.

GENERAL RULES
1. Extract only what the CV says. Never invent anything.
2. If a value is missing or unclear, use null (or an empty list). Never write
   "Unknown" or "N/A".
3. Do not depend on fixed section titles. Read the content, not the headings.
4. Return JSON only, with no Markdown and no explanations.

EDUCATION
- List every university diploma or study program in the CV.
- Ignore high school: "Baccalauréat" / "Bac" is NOT a Bachelor.
- For each diploma, fill:
    - title: the diploma name copied exactly as written in the CV.
    - level: Bachelor, Master, Engineering or PhD.
    - status: "completed" if obtained, "in_progress" if the candidate is
      studying it now (future end date, "expected", "en cours", "current").
      If it is not clear, null.

HOW TO CHOOSE THE LEVEL
The level comes from the TYPE of diploma, never from the field of study.
Words like "engineering", "génie", "ingénierie" describe the FIELD, so they
do not make a diploma an Engineering diploma.
- Bachelor: "Licence", "License", "Bachelor", "Degree" (3 years).
  Examples: "Licence en Génie Logiciel", "Degree in Computer Engineering",
  "Bachelor in Chemical Engineering", "Licence Appliquée en Informatique".
- Master: "Master", "Mastère" (professional or research).
  Examples: "Mastère Professionnel en Data Science",
  "Master in Software Engineering".
- Engineering: ONLY when the CV says the candidate follows or holds an
  engineer's diploma: "Cycle Ingénieur", "Diplôme d'Ingénieur",
  "Diplôme National d'Ingénieur", "Engineering degree".
  Examples: "Cycle Ingénieur en Informatique",
  "Engineering degree in Telecommunications".
- PhD: "Doctorat", "PhD".
- If the title has no diploma type (for example just "Computer Engineering"),
  set level to null. Do not guess.

OTHER EDUCATION FIELDS
- field_of_study: the current field, for example "Computer Science".
- year_of_study: a number, only if the CV states it explicitly.

SKILLS
- The professional skills listed in the skills section of the CV: software,
  tools, programming languages, methods and domain skills. They can come from
  any field, for example "Python", "AutoCAD", "Excel", "SAP", "Photoshop",
  "Accounting", "Recruitment", "Project Management", "Welding".
- Use short names only, never sentences and no brackets.
  Write "STM32", not "STM32F4 microcontroller (STM32Cube HAL library)".
- Split grouped items into separate skills: "Python, SQL" becomes two skills.
- Never put spoken languages (Arabic, English, French...) in skills.
- Never use category headings ("Programming", "Networking") as skills.
- Leave out soft skills such as teamwork, leadership or communication.
- No duplicates.

PROJECTS AND EXPERIENCE (internships, jobs)
- One item per project and one per internship/job. Do not split one project
  into several items.
- name: the project title, or the position and company.
- skills: the technologies, methods and concepts that the project or
  internship USED, even when they are not in the skills section.
  Examples: "Docker", "NLP", "SolidWorks", "Financial Analysis", "SEO".
  Same rules as SKILLS: short names, no sentences, no brackets.
- A skill is something a company could ask for in a job offer. NEVER add:
  results or numbers ("95% accuracy", "20% cost reduction"), dataset names,
  the project title or topic ("Online store", "Bridge design"), or a
  description of what was done.

OUTPUT
Return exactly this schema:

{
    "education": {
        "diplomas": [
            {"title": null, "level": null, "status": null}
        ],
        "field_of_study": null,
        "year_of_study": null,
        "current_level": null
    },
    "skills": [],
    "experience": [
        {"name": null, "skills": []}
    ],
    "projects": [
        {"name": null, "skills": []}
    ]
}
"""


def build_user_prompt(cv_text: str) -> str:
    return (
        "Extract the candidate profile from the CV text below. "
        "Follow the schema and the rules carefully.\n\n"
        "----- BEGIN CV TEXT -----\n"
        f"{cv_text}\n"
        "----- END CV TEXT -----"
    )


def build_system_prompt() -> str:
    """Add today's date so the model can tell finished diplomas from current ones."""
    return SYSTEM_PROMPT + (
        f"\nTODAY'S DATE: {date.today():%d %B %Y}. A diploma whose end date "
        "is before today, or that says awarded / obtained / graduated / "
        "obtenu / diplômé, is \"completed\". A diploma ending in the future "
        "or marked present / en cours / expected is \"in_progress\".\n"
    )


def call_ollama(cv_text: str) -> CandidateProfile:
    """
    Ask Ollama for schema-constrained JSON, then validate with Pydantic.
    A second attempt is made if the first response is invalid.
    """
    schema = CandidateProfile.model_json_schema()

    for attempt in range(2):
        prompt = build_user_prompt(cv_text)
        if attempt == 1:
            prompt += (
                "\n\nYour previous response did not match the required "
                "JSON structure. Re-read the CV and return one valid JSON "
                "object. Use null, never 'Unknown'."
            )

        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            format=schema,
            think=False,
            options={
                "temperature": 0, #no randomness
                "num_ctx": 8192  # how much text (input and conversation context) the model can consider when generating its response.
            }
        )

        raw = response["message"]["content"].strip()
        raw = remove_json_fences(raw) #remove extras from the model output, like ```json fences or extra text

        try:
            data = json.loads(raw) #turns the text into a Python dictionary
            return CandidateProfile.model_validate(data) #builds the object and checks
        except (json.JSONDecodeError, ValidationError) as exc:
            if attempt == 1:
                raise ValueError(
                    "The model did not return valid profile JSON after "
                    f"two attempts. Last error: {exc}\n\nRaw response:\n{raw}"
                ) from exc

    raise RuntimeError("Unexpected parser state.")


def remove_json_fences(text: str) -> str:
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()



# ---------------------------------------------------------------------------
# Post-processing and normalization
# ---------------------------------------------------------------------------

NULL_LIKE = {
    "", "unknown", "n/a", "na", "none", "null", "not specified",
    "not available", "undefined"
}

# Headings the model sometimes returns as if they were skills.
CATEGORY_WORDS = {
    "skills", "technical skills", "programming", "programming languages",
    "networking", "certifications", "education", "experience", "projects",
    "tools", "languages", "soft skills",
}

# Evaluation metrics are results, not skills.
METRIC_WORDS = {
    "accuracy", "precision", "recall", "f1", "f1-score", "f1 score",
    "mae", "mse", "rmse", "r2", "r²", "auc", "loss",
}
# Spoken languages must never stay in the skills list.
SPOKEN_LANGUAGES = {
    "arabic", "arabe", "english", "anglais", "french", "français",
    "francais", "german", "allemand", "deutsch", "spanish", "espagnol",
    "italian", "italien", "turkish", "turc", "chinese", "chinois",
    "russian", "russe","portuguese", "portugais", "japanese", "japonais",
    "dutch", "néerlandais", "korean", "coréen",
}

def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    if value.lower() in NULL_LIKE:
        return None
    return value


def normalize_skills(values: list[str]) -> list[str]:
    """Clean the skills: short names, no duplicates, no spoken languages."""
    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        skill = normalize_text(value)
        if not skill:
            continue

        # "STM32F4 (STM32Cube HAL library)" -> "STM32F4"
        skill = re.sub(r"\s*\(.*?\)", "", skill).strip(" :;,-")

        key = skill.casefold()
        if not skill or key in CATEGORY_WORDS or key in SPOKEN_LANGUAGES:
            continue
        if key in METRIC_WORDS or "%" in skill:
            continue
        # More than 5 words is a sentence or a description, not a skill name.
        if len(skill.split()) > 5:
            continue
        # "Recommendation System" and "Recommendation Systems" are the same skill.
        key = key[:-1] if len(key) > 4 and key.endswith("s") else key
        if key not in seen:
            seen.add(key)
            cleaned.append(skill)

    return cleaned


# --- Diploma level detection ------------------------------------------------
# The LLM copies the diploma title, and these rules decide the level.
# The order matters: explicit words like "licence" or "master" win over
# "engineering", because "Licence en Génie Logiciel" or "Bachelor in Chemical
# Engineering" are 3-year degrees, not engineering diplomas.

HIGH_SCHOOL = re.compile(r"baccalaur[eé]at|\bbac\b(?!\s*\+)", re.IGNORECASE)
LEVEL_RULES: list[tuple[str, re.Pattern]] = [
    ("PhD", re.compile(r"\b(ph\.?d|doctorat|doctorate)\b", re.IGNORECASE)),
    ("Master", re.compile(r"\b(master'?s?|mast[eè]re|m\.?sc)\b", re.IGNORECASE)),
    ("Bachelor", re.compile(
        r"\b(licen[cs]e|bachelor'?s?|b\.?sc|b\.?eng)\b", re.IGNORECASE
    )),
    # Only a real engineer's diploma ("cycle ingénieur", "engineering degree"...)
    ("Engineering", re.compile(
        r"ing[eé]nieur"
        r"|engineer'?s?\s+(degree|diploma|cycle)"
        r"|engineering\s+(degree|diploma|cycle)",
        re.IGNORECASE,
    )),
    # A plain "degree in ..." (even "degree in computer engineering") is a Bachelor
    ("Bachelor", re.compile(r"\bdegree\b", re.IGNORECASE)),
]


def detect_level(title: Optional[str]) -> Optional[str]:
    """Return the diploma level from its title, or None if it is unclear."""
    if not title:
        return None
    for level, pattern in LEVEL_RULES:
        if pattern.search(title):
            return level
    return None


# Engineering (5 years) is placed at the same height as Master.
LEVEL_RANK = {"Bachelor": 1, "Master": 2, "Engineering": 2, "PhD": 3}


def compute_current_level(diplomas: list[DiplomaItem]) -> Optional[str]:
    """
    What the student is studying NOW decides which offers they can apply to.
    Use the diploma in progress. If there is none, use the highest one.
    """
    pool = [d for d in diplomas if d.status == "in_progress"] or diplomas
    if not pool:
        return None
    return max(pool, key=lambda d: LEVEL_RANK[d.level]).level


# --- Language levels to CEFR --------------------------------------------------
# CVs say "B2", "Fluent" or "IELTS 6.5", offers say "C1". We convert to CEFR
# so they can be compared. Anything unclear stays None.

CEFR_WORDS: list[tuple[str, str]] = [
    (r"native|mother tongue|maternelle|natif|bilingual|bilingue", "C2"),
    # "upper-intermediate" must come before "intermediate" and "advanced".
    (r"upper[- ]intermediate|good|bon\b|très bien|tres bien", "B2"),
    (r"advanced|avanc[eé]|fluent|courant|professional|professionnel", "C1"),
    (r"intermediate|interm[eé]diaire|conversational|conversationnel", "B1"),
    (r"basic|elementary|notions|scolaire", "A2"),
    (r"beginner|d[eé]butant", "A1"),
]


def to_cefr(level: Optional[str]) -> Optional[str]:
    if not level:
        return None
    text = level.lower()

    # Already a CEFR level: "B2", "C1 (DELF)"...
    found = re.search(r"\b([abc][12])\b", text)
    if found:
        return found.group(1).upper()

    # Test scores (approximate conversion)
    score = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if score:
        value = float(score.group(1).replace(",", "."))
        if "ielts" in text:
            for limit, cefr in [(8.5, "C2"), (7.0, "C1"), (5.5, "B2"), (4.0, "B1")]:
                if value >= limit:
                    return cefr
        if "toeic" in text:
            for limit, cefr in [(945, "C1"), (785, "B2"), (550, "B1"), (225, "A2")]:
                if value >= limit:
                    return cefr

    for pattern, cefr in CEFR_WORDS:
        if re.search(pattern, text):
            return cefr
    return None


def merge_skills(profile: CandidateProfile) -> list[str]:
    """Skills section first, then the skills used in projects and internships."""
    merged = list(profile.skills)
    for item in profile.projects + profile.experience:
        merged.extend(item.skills)
    return normalize_skills(merged)


def normalize_profile(profile: CandidateProfile) -> CandidateProfile:
    # Education
    education = profile.education #same object not a copy, so we can modify it in place
    kept: list[DiplomaItem] = [] #hold the diplomas that pass all checks.
    seen: set[tuple] = set() #set of (level, status) pairs, used to detect duplicates.

    for diploma in education.diplomas:
        diploma.title = normalize_text(diploma.title)

        # The baccalaureate is high school, not a university diploma.
        if diploma.title and HIGH_SCHOOL.search(diploma.title):
            continue

        # An entry with no title and no level has nothing to classify.
        if not diploma.title and not diploma.level:
            continue

        # Priority: 1) rules on the title, 2) the model's level, 3) Bachelor.
        diploma.level = detect_level(diploma.title) or diploma.level or "Bachelor"

        key = (diploma.level, diploma.status)
        if key not in seen:
            seen.add(key)
            kept.append(diploma)

    education.diplomas = kept
    education.current_level = compute_current_level(kept)
    education.field_of_study = normalize_text(education.field_of_study)
    year = education.year_of_study
    education.year_of_study = year if year is not None and 1 <= year <= 10 else None
    # Skills
    profile.skills = merge_skills(profile)
    return profile

def check_profile(profile: CandidateProfile) -> list[str]:
    """List what is missing, so unusual CVs are noticed and not silently wrong."""
    warnings = []
    if not profile.education.current_level:
        warnings.append("No diploma level found: ask the student.")
    if profile.education.year_of_study is None:
        warnings.append("Year of study not found: ask the student.")
    if not profile.skills:
        warnings.append("No skills found: check the CV text extraction.")
    return warnings


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def get_profile(input_path: str | Path) -> CandidateProfile:
    """Read a CV and return the cleaned profile."""
    cv_text = extract_pdf_text(input_path)
    if not cv_text:
        raise ValueError(
            "No readable text was extracted. The PDF may be scanned "
            "(an image) or protected."
        )
    print(f"[INFO] Extracted characters: {len(cv_text)}")

    if len(cv_text) > MAX_INPUT_CHARS:
        print(
            f"[WARNING] CV text is long ({len(cv_text)} characters). "
            f"Only the first {MAX_INPUT_CHARS} characters will be sent."
        )
        cv_text = cv_text[:MAX_INPUT_CHARS]

    print(f"[INFO] Sending CV to Ollama model: {MODEL}")
    profile = call_ollama(cv_text)
    profile = normalize_profile(profile)
    for warning in check_profile(profile):
        print(f"[WARNING] {warning}")

    return profile


def parse_cv(input_path: str | Path) -> Path:
    """Parse a CV, save the profile as profile.json and return its path."""
    profile = get_profile(input_path)
    output_path = Path("profile.json")

    output_path.write_text(
        json.dumps(
            profile.model_dump(mode="json", exclude={"projects", "experience"}),
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    print(f"[SUCCESS] Profile saved to: {output_path.resolve()}")
    return output_path

def main() -> None:
    cv_path = "cv_fr_6.pdf"
    target = Path(cv_path).expanduser().resolve()
    try:
        parse_cv(target)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()