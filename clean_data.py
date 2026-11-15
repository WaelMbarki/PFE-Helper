
import pandas as pd
import re
import json
import json
import ollama                     


# Read the original data
df = pd.read_csv("jobs.csv")
if "source" not in df.columns:
    df["source"] = "LinkedIn"
df["description"] = df["description"].fillna("")




# Technical skills and their possible names
skills = {
    "Python": ["python"],
    "Java": ["java"],
    "C++": ["c++"],
    "SQL": ["sql", "mysql", "postgresql", "sql server"],
    "Machine Learning": ["machine learning", "apprentissage automatique"],
    "Deep Learning": ["deep learning", "apprentissage profond"],
    "NLP": ["nlp", "natural language processing", "traitement du langage naturel"],
    "Computer Vision": ["computer vision", "vision par ordinateur"],
    "Data Science": ["data science", "science des données"],
    "Data Analysis": ["data analysis", "data analyst", "analyse de données"],
    "Power BI": ["power bi"],
    "Excel": ["excel"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "Spring Boot": ["spring boot"],
    "Angular": ["angular"],
    "React": ["react"],
    "Node.js": ["node.js"],
    "REST API": ["rest api", "restful api"],
    "Microservices": ["microservices"],
    "MongoDB": ["mongodb"],
    "Git": ["git", "github", "gitlab"],
    "JavaScript": ["javascript"],
    "TypeScript": ["typescript"],
    "HTML": ["html"],
    "CSS": ["css"],
    "PHP": ["php"],
    "WordPress": ["wordpress"],
    "Elementor": ["elementor"],
    "CATIA": ["catia", "catia v5"],
    "Siemens NX": ["siemens nx"],
    "AutoCAD": ["autocad"],
    "SolidWorks": ["solidworks"],
    "CAD": ["computer-aided design", "cad", "conception assistée par ordinateur"],
    "GIS": ["gis", "geographic information system", "système d'information géographique"],
    "Telecommunications": ["telecommunications", "télécommunications", "telecom"],
    "Fiber Optics": ["fiber optics", "fibre optique", "fiber optic"],
    "SAP": ["sap"],
    "Linux": ["linux"],
    "IoT": ["iot", "internet of things", "internet des objets"],
    "Cybersecurity": ["cybersecurity", "cybersécurité"],
    "Automation": ["automation", "automatisation"],
    "Software Testing": ["software testing", "automated testing", "tests automatisés"],
    "Financial Modeling": ["financial modeling", "financial modelling", "modélisation financière"],
    "Financial Analysis": ["financial analysis", "analyse financière"],
    "Research": ["market research", "sector research", "étude de marché"],
    "Project Management": ["project management", "gestion de projet"],
    "Photography": ["photography", "photographie"],
    "Translation": ["translation", "traduction"],
}


# Internship domains
domains = {
    "AI": ["artificial intelligence", "intelligence artificielle",
           "machine learning", "deep learning"],
    "Data Science": ["data science", "data scientist", "science des données"],
    "Data Analysis": ["data analysis", "data analyst", "analyse de données"],
    "Web Development": ["web development", "développement web",
                        "full stack", "full-stack", "wordpress"],
    "Mobile Development": ["mobile development", "développement mobile",
                           "android development", "ios development"],
    "Software Engineering": ["software engineering", "génie logiciel",
                             "software development", "développement logiciel"],
    "Cloud and DevOps": ["devops", "cloud computing", "cloud engineer",
                         "kubernetes", "continuous integration"],
    "Cybersecurity": ["cybersecurity", "cybersécurité",
                      "penetration testing"],
    "Embedded Systems": ["embedded systems", "systèmes embarqués",
                         "microcontroller"],
    "IoT": ["internet of things", "internet des objets"],
    "Telecommunications": ["telecommunications", "télécommunications",
                           "telecom", "fibre optique", "fiber optics"],
    "Human Resources": ["human resources", "ressources humaines",
                        "recrutement", "fiche de poste"],
    "Organizational Management": ["organizational design",
                                  "organizational management",
                                  "refonte organisationnelle",
                                  "référentiel des emplois",
                                  "gestion des compétences"],
    "Business and Finance": ["finance", "financial analysis",
                             "transaction advisory", "accounting",
                             "infrastructure advisory", "private-public partnership",
                             "partenariat public privé"],
    "Mechanical Engineering": ["mechanical engineering", "génie mécanique",
                               "mechanical design", "conception mécanique",
                               "catia", "siemens nx"],
    "Quality Assurance": ["quality assurance", "quality analyst",
                          "software testing", "test automation",
                          "qualité logicielle"],
    "Technical Support": ["technical support", "support technique",
                          "troubleshooting"],
    "Project Management": ["project management", "gestion de projet",
                           "project coordinator", "coordinateur de projet"],
    "Marketing and Communication": ["marketing", "communication",
                                    "social media", "content creation"],
    "Photography": ["photography", "photographie"],
    "Translation": ["translation", "translator", "traduction",
                    "traducteur", "traductrice"],
    "Consulting": ["consulting", "consultancy", "conseil",
                   "business transformation"],
}


def contains_term(text, term):
    pattern = r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def extract_terms(text, vocabulary):
    found = []

    for name, keywords in vocabulary.items():
        if any(contains_term(text, word) for word in keywords):
            found.append(name)

    return found


def get_section(text, headings, next_headings):
    # Find the content between a section heading and the next heading
    for heading in headings:
        pattern = r"(?im)^\s*" + re.escape(heading) + r"\s*:?\s*$"
        match = re.search(pattern, text)

        if match:
            content = text[match.end():]

            for next_heading in next_headings:
                next_pattern = (
                    r"(?im)^\s*" + re.escape(next_heading) + r"\s*:?\s*$"
                )
                next_match = re.search(next_pattern, content)

                if next_match:
                    content = content[:next_match.start()]

            return content.strip()

    return ""


def extract_education_type(text):
    text = text.lower()
    found = []

    # Engineering PFE
    if re.search(r"\b(engineering school|école d'ingénieurs|"
                 r"école d’ingénieurs|engineering degree|"
                 r"cycle d'ingénieur)\b", text):
        found.append("engineering")

    # Master's PFE
    if re.search(r"\b(master|master's|master’s|mastère|"
                 r"master thesis|master's thesis|mastère spécialisé)\b", text):
        found.append("master")

    # License PFE
    if re.search(r"\b(licence|bachelor|bachelor's|bachelor’s)\b", text):
        found.append("license")

    return found if found else ["unspecified"]


def extract_experience(text):
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    found = []

    # Look for explicit previous experience requirements
    keywords = [
        "previous experience",
        "prior experience",
        "first experience",
        "proven experience",
        "experience required",
        "years of experience",
        "expérience préalable",
        "expérience précédente",
        "première expérience",
        "expérience significative",
        "expérience professionnelle",
        "projet académique",
        "projets académiques",
        "academic project",
        "academic projects",
        "hands-on experience",
    ]

    for sentence in sentences:
        sentence = sentence.strip()

        if sentence and any(
            contains_term(sentence, word) for word in keywords
        ):
            if sentence not in found:
                found.append(sentence)

    return found





def llm_domains(title, description):
    print("We are going to use LLm")
    MODEL = "qwen3:4b"

    SCHEMA = {
        "type": "object",
        "properties": {
            "job_domain": {
                "type": "array",
                "items": {"type": "string", "enum": list(domains)},
            },
        },
        "required": ["job_domain"],
    }

    PROMPT = (
        "You classify internship postings (English or French). "
        "Choose the domains that match the posting. "
        "Choose only from the allowed values. Return an empty list if none fit."
    )
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Title: {title}\n\n{description[:4000]}"},
        ],
        format=SCHEMA,
        think=False,
        options={"temperature": 0, "num_ctx": 8192},
    )
    return json.loads(response["message"]["content"])["job_domain"]


def clean_job(title, description):
    title = str(title)
    description = str(description)

    # Prefer the job-specific description, not company introduction
    job_text = get_section(
        description,
        ["Job Description", "Description du poste", "Missions", "Role Description"],
        ["Qualifications", "Profil recherché", "Requirements",
         "Job Requirement", "Additional Information"]
    )

    # If no section is found, use the full description
    if not job_text:
        job_text = description

    # Qualifications contain the most useful education information
    qualification_text = get_section(
        description,
        ["Qualifications", "Profil recherché", "Requirements",
         "Job Requirement", "Compétences requises"],
        ["Additional Information", "Informations complémentaires",
         "Comment postuler", "How to apply"]
    )

    if not qualification_text:
        qualification_text = job_text

    # Use job title, job description and qualifications for classification
    relevant_text = title + "\n" + job_text + "\n" + qualification_text

    return {
        "technical_skills": extract_terms(relevant_text, skills),
        "education_type": extract_education_type(
            title + "\n" + qualification_text
        ),
        "required_experience": extract_experience(qualification_text),
        "job_domain": extract_terms(relevant_text, domains),
    }


# Extract the four categories
extracted = df.apply(
    lambda row: clean_job(row["title"], row["description"]),
    axis=1
)

for column in [
    "technical_skills",
    "education_type",
    "required_experience",
    "job_domain"
]:
    df[column] = extracted.apply(
        lambda result: json.dumps(result[column], ensure_ascii=False)
    )

df["domain_method"] = "rules"

for i, row in df.iterrows():
    if df.at[i, "job_domain"] == "[]":      # the rules found no domain
        print(f"LLM {i + 1}/{len(df)}: {row['title']}")
        try:
            found = llm_domains(row["title"], row["description"])
        except Exception as e:
            print(f"   LLM failed ({e.__class__.__name__}), keeping []")
            continue
        df.at[i, "job_domain"] = json.dumps(found, ensure_ascii=False)
        df.at[i, "domain_method"] = "llm"
        
# Save the final dataset
df.to_csv("jobs_cleaned.csv", index=False, encoding="utf-8-sig")

print("Cleaning finished!")
print("Saved as jobs_cleaned.csv")

print(df[[
    "title",
    "technical_skills",
    "education_type",
    "required_experience",
    "job_domain"
]].to_string(index=False))
