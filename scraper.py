import random
import re
import sqlite3
import time
from datetime import datetime, timezone
from html import unescape
import os
import csv
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ---------------------------------------------------------------- settings
# One fixed search: Tunisia, any date, and the 3 words joined with OR.
# (%20 = space in a URL. OR must be written in capital letters.)
SEARCH_URL = (
    "https://www.linkedin.com/jobs/search/"
    "?keywords=stage%20OR%20PFE%20OR%20internship"
    "&location=Tunisia"
)
DETAIL_URL = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
TANITJOBS_URL = "https://www.tanitjobs.com/categories/977/stage-jobs/"
SOURCE_LINKEDIN = "LinkedIn"
SOURCE_TANITJOBS = "TanitJobs"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0"
)

CSV_PATH = "jobs.csv"
COLUMNS = [
    "id", "title", "company", "location", "posted_at",
    "url", "description", "company_url", "scraped_at", "dedupe_key", "source",
]
SHOW_BROWSER = False   # True = show the Firefox window (debugging)
MAX_JOBS = 10        # safety limit; LinkedIn's public search stops around 1000 anyway


# ---------------------------------------------------------------- database
def clean(text):
    """Lowercase the text and collapse extra spaces."""
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_dedupe_key(job: dict) -> str:
    """Same title + company + location = same internship, even with a new ID."""
    title = clean(job.get("title"))
    company = clean(job.get("company"))
    location = clean(job.get("location"))
    return f"{title}|{company}|{location}"


def ensure_csv_schema() -> None:
    """Add the source column to older CSV files, treating existing rows as LinkedIn."""
    if not os.path.exists(CSV_PATH):
        return

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "source" in (reader.fieldnames or []):
            return
        rows = list(reader)

    temporary_path = f"{CSV_PATH}.tmp"
    with open(temporary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            row["source"] = SOURCE_LINKEDIN
            writer.writerow({column: row.get(column, "") for column in COLUMNS})
    os.replace(temporary_path, CSV_PATH)


def load_known():
    """Read the CSV once and return the IDs and dedupe keys we already have."""
    known_ids, known_keys = set(), set()
    if not os.path.exists(CSV_PATH):  # first run: no file yet
        return known_ids, known_keys

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            known_ids.add(row["id"])
            known_keys.add(row["dedupe_key"])
    return known_ids, known_keys

def save_job(job: dict) -> None:
    """Add one job at the end of the CSV file (creates the file + header if needed)."""
    is_new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(job)


# ---------------------------------------------------------------- selenium helpers
def make_driver(show_browser: bool = False) -> webdriver.Firefox:
    options = Options()
    if not show_browser:
        options.add_argument("-headless")
    options.set_preference("general.useragent.override", USER_AGENT)
    options.set_preference("intl.accept_languages", "fr-TN, fr, en-US, en")
    driver = webdriver.Firefox(options=options)
    driver.set_window_size(1366, 900)
    driver.set_page_load_timeout(60)
    return driver


def polite_sleep(low: float = 2.0, high: float = 5.0) -> None:
    time.sleep(random.uniform(low, high))


def is_blocked(driver) -> bool:
    """LinkedIn redirects bots to an auth wall / login / checkpoint page."""
    url = driver.current_url.lower()
    return any(x in url for x in ("authwall", "/login", "/checkpoint", "/uas/"))


def dismiss_modal(driver) -> None:
    """Close the 'sign in' popup that sometimes overlays the public results."""
    for btn in driver.find_elements(By.CSS_SELECTOR, "button.modal__dismiss"):
        try:
            if btn.is_displayed():
                driver.execute_script("arguments[0].click();", btn)
        except WebDriverException:
            pass


def safe_text(parent, css: str) -> str:
    try:
        return parent.find_element(By.CSS_SELECTOR, css).get_attribute("textContent").strip()
    except NoSuchElementException:
        return ""


def safe_attr(parent, css: str, attr: str) -> str:
    #same as safe_text but returns the value of a given attribute instead of the text content.
    try:
        return parent.find_element(By.CSS_SELECTOR, css).get_attribute(attr) or ""
    except NoSuchElementException:
        return ""


# ---------------------------------------------------------------- phase 1: links
def get_jobs_links(driver, max_jobs: int):
    """Open the search page, scroll / click 'See more jobs', return basic card data.
    Returns None if LinkedIn blocked us, otherwise a list of dicts."""
    driver.get(SEARCH_URL) #makes Firefox load the page and waits until it finishes loading
    polite_sleep(3, 5) #random sleep to avoid being detected as a bot
    if is_blocked(driver):
        print("  blocked by LinkedIn (auth wall).")
        return None

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "ul.jobs-search__results-list li"))
        )
    except TimeoutException:
        print("  no results found on this page")
        return []

    card_css = "ul.jobs-search__results-list li div.base-card" #the selector for one job card.
    previous, stagnant = 0, 0 #previous is the card count from the last round, and stagnant counts the rounds with no progress.
    while True:
        dismiss_modal(driver) #closes the sign-in popup.
        count = len(driver.find_elements(By.CSS_SELECTOR, card_css)) 
        print(f"  {count} cards loaded")
        if count >= max_jobs:
            break

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") #scroll to the bottom of the page to load more jobs.
        time.sleep(random.uniform(1.5, 3))
        # the "See more jobs" button appears after a few scrolls
        for btn in driver.find_elements(By.CSS_SELECTOR, "button.infinite-scanner__show-more-button"):
            if btn.is_displayed(): #if the button is visible, click it to load more jobs.
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(random.uniform(2, 4))

        stagnant = stagnant + 1 if count == previous else 0 #if the count didn’t change, add 1 to stagnant, otherwise reset it to 0
        if stagnant >= 3:  # nothing new after 3 tries -> end of results
            break
        previous = count

    jobs = []
    #Extract data from each card
    for card in driver.find_elements(By.CSS_SELECTOR, card_css)[:max_jobs]: #it keeps only the first max_jobs items of the list.
        urn = card.get_attribute("data-entity-urn") or ""
        if not urn:
            continue
        jobs.append(
            {
                "id": urn.split(":")[-1],
                "title": safe_text(card, "h3.base-search-card__title"),
                "company": safe_text(card, "h4.base-search-card__subtitle"),
                "location": safe_text(card, "span.job-search-card__location"),
                "posted_at": safe_attr(card, "time", "datetime"),
                "url": safe_attr(card, "a.base-card__full-link", "href").split("?")[0],
                "source": SOURCE_LINKEDIN,
            }
        )
    return jobs


def is_pfe_listing(title: str, description: str) -> bool:
    text = clean(f"{title} {description}")
    pattern = (
        r"\b(pfe|projets? de fin d['’]études|stages? de fin d['’]études|"
        r"final[ -]+year project|end[ -]+of[ -]+study internship)\b"
    )
    return re.search(pattern, text) is not None


def get_tanitjobs_links(driver, max_jobs: int):
    """Read PFE listings from TanitJobs' Stage category."""
    print(f"  [TanitJobs] Opening category: {TANITJOBS_URL}")
    driver.get(TANITJOBS_URL)
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".sj-job-card"))
        )
    except TimeoutException:
        print("  no TanitJobs results found")
        return []

    jobs = []
    cards = driver.find_elements(By.CSS_SELECTOR, ".sj-job-card")
    print(f"  [TanitJobs] Listing cards found: {len(cards)}")
    for index, card in enumerate(cards, start=1):
        title = safe_text(card, ".sj-card-title")
        description = safe_text(card, ".sj-card-desc")
        print(f"  [TanitJobs] Card {index}/{len(cards)}: {title or '<missing title>'}")
        if not is_pfe_listing(title, description):
            print("    skipped: no PFE/final-year-project wording in title or preview")
            continue

        url = safe_attr(card, ".sj-card-title a", "href").split("?")[0]
        match = re.search(r"/job/(\d+)/", url)
        if not match:
            print(f"    skipped: could not extract job ID from URL {url!r}")
            continue

        job = {
            "id": f"tanitjobs-{match.group(1)}",
            "title": title,
            "company": safe_text(card, ".sj-card-company a"),
            "location": safe_text(card, ".sj-card-tag.sj-loc"),
            "posted_at": safe_text(card, ".sj-card-date"),
            "url": url,
            "description": description,
            "company_url": "",
            "source": SOURCE_TANITJOBS,
        }
        jobs.append(job)
        print(
            f"    extracted: id={job['id']}, company={job['company']!r}, "
            f"location={job['location']!r}, preview_chars={len(description)}, url={url}"
        )
        if len(jobs) >= max_jobs:
            print(f"  [TanitJobs] Reached max_jobs={max_jobs}")
            break
    print(f"  [TanitJobs] PFE listings extracted: {len(jobs)}")
    return jobs


# ---------------------------------------------------------------- phase 2: details
def html_to_text(html: str) -> str:
    """Turn the description HTML into readable text, keeping paragraphs and bullet points."""
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    html = re.sub(r"</(p|div|ul|ol|h[1-6])>", "\n", html, flags=re.I)
    html = re.sub(r"<li[^>]*>", "\n- ", html, flags=re.I)
    text = unescape(re.sub(r"<[^>]+>", "", html))
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_job_details(driver, job: dict) -> bool:
    """Fill the full description (+ company link) for one job. Returns False if blocked.
    the modifications happens in the job dictionary."""
    driver.get(DETAIL_URL.format(job_id=job["id"])) #open the pfe page
    if is_blocked(driver):
        return False

    try:
        element = WebDriverWait(driver, 8).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.show-more-less-html__markup"))
        )
        job["description"] = html_to_text(element.get_attribute("innerHTML"))
    except TimeoutException:
        job["description"] = ""

    job["company_url"] = safe_attr(driver, "a.topcard__org-name-link", "href").split("?")[0]
    return True


def get_tanitjobs_details(driver, job: dict) -> bool:
    """Load the full TanitJobs description, retaining the card preview if unavailable."""
    print(f"    [TanitJobs] Loading detail page: {job['url']}")
    driver.get(job["url"])
    try:
        element = WebDriverWait(driver, 8).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".detail-offre"))
        )
        job["description"] = html_to_text(element.get_attribute("innerHTML"))
        print(f"    [TanitJobs] Full description extracted: {len(job['description'])} chars")
    except TimeoutException:
        print(
            f"    [TanitJobs] Detail description not found; keeping "
            f"{len(job['description'])} preview chars"
        )
    return True


# ---------------------------------------------------------------- main
def scrape_source(driver, source: str, get_cards, get_details) -> int:
    print(f"\nSearching {source}")
    cards = get_cards(driver, MAX_JOBS)
    if cards is None:
        print(f"  {source} did not return results")
        return 0
    print(f"  [{source}] Listings returned by source parser: {len(cards)}")

    # keep only jobs we do not have yet (by ID or by title+company+location)
    known_ids, known_keys = load_known()
    new_jobs = []
    duplicate_ids = 0
    duplicate_keys = 0
    for job in cards:
        key = make_dedupe_key(job)
        if job["id"] in known_ids:
            duplicate_ids += 1
            print(f"  [{source}] duplicate ID, skipping {job['id']}: {job['title']}")
            continue
        if key in known_keys:
            duplicate_keys += 1
            print(f"  [{source}] duplicate title/company/location, skipping: {job['title']}")
            continue
        known_ids.add(job["id"])
        known_keys.add(key)
        job["dedupe_key"] = key
        new_jobs.append(job)
    print(
        f"  [{source}] new={len(new_jobs)}, duplicate_ids={duplicate_ids}, "
        f"duplicate_keys={duplicate_keys}, existing_rows={len(load_known()[0])}"
    )

    saved = 0
    for i, job in enumerate(new_jobs, start=1):
        print(f"  details {i}/{len(new_jobs)}: {job['title']} @ {job['company']}")
        try:
            ok = get_details(driver, job)
        except (TimeoutException, WebDriverException) as e:
            print(f"    error: {e.__class__.__name__}, skipping")
            continue
        if not ok:
            print(f"  {source} blocked this session")
            break
        print(f"    description_chars={len(job.get('description', ''))}")
        job["scraped_at"] = datetime.now(timezone.utc).isoformat()
        save_job(job)  # written immediately, so a crash never loses earlier jobs
        saved += 1
        print(f"    saved: source={source}, id={job['id']}, file={CSV_PATH}")
        polite_sleep()
    print(f"  [{source}] Saved {saved} new listings")
    return saved


def scrape(driver) -> int:
    ensure_csv_schema()
    return scrape_source(
        driver, SOURCE_LINKEDIN, get_jobs_links, get_job_details
    ) + scrape_source(
        driver, SOURCE_TANITJOBS, get_tanitjobs_links, get_tanitjobs_details
    )



def run() -> None:
    driver = make_driver(SHOW_BROWSER)
    new_count = 0
    try:
        new_count = scrape(driver)
    finally:
        driver.quit()
    total = len(load_known()[0])
    print(f"\nDone. {new_count} new jobs, {total} total in {CSV_PATH}")


if __name__ == "__main__":
    run()