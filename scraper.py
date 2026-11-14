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
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0"
)

CSV_PATH = "jobs.csv"
COLUMNS = [
    "id", "title", "company", "location", "posted_at",
    "url", "description", "company_url", "scraped_at", "dedupe_key",
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
            }
        )
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


# ---------------------------------------------------------------- main
def scrape(driver) -> int:
    """Run the one search, save every new job. Returns how many were saved."""
    print(f"\nSearching: {SEARCH_URL}")
    cards = get_jobs_links(driver, MAX_JOBS)
    if cards is None:
        raise RuntimeError("blocked")

    # keep only jobs we do not have yet (by ID or by title+company+location)
    known_ids, known_keys = load_known()
    new_jobs = []
    for job in cards:
        key = make_dedupe_key(job)
        if job["id"] in known_ids or key in known_keys:
            continue
        known_ids.add(job["id"])
        known_keys.add(key)
        job["dedupe_key"] = key
        new_jobs.append(job)
    print(f"  {len(new_jobs)} new jobs out of {len(cards)}")

    saved = 0
    for i, job in enumerate(new_jobs, start=1):
        print(f"  details {i}/{len(new_jobs)}: {job['title']} @ {job['company']}")
        try:
            ok = get_job_details(driver, job)
        except (TimeoutException, WebDriverException) as e:
            print(f"    error: {e.__class__.__name__}, skipping")
            continue
        if not ok:
            raise RuntimeError("blocked")
        job["scraped_at"] = datetime.now(timezone.utc).isoformat()
        save_job(job)  # written immediately, so a crash never loses earlier jobs
        saved += 1
        polite_sleep()
    return saved



def run() -> None:
    driver = make_driver(SHOW_BROWSER)
    new_count = 0
    try:
        new_count = scrape(driver)
    except RuntimeError:
        print("LinkedIn blocked this session. Try again later (hours, not minutes).")
    finally:
        driver.quit()
    total = len(load_known()[0])
    print(f"\nDone. {new_count} new jobs, {total} total in {CSV_PATH}")


if __name__ == "__main__":
    run()