"""
matcher.py - ranks offers by semantic similarity with a candidate's CV.

It does NOT check eligibility: give it the offers that eligible.py already accepted.

    from matcher import rank_offers
    ids = rank_offers(candidate, eligible_offers)   # [4470814008, 4479295088, ...]
"""

import json

import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # French + English, small, CPU-friendly
CHUNK_CHARS = 600   # the model only reads ~128 tokens, so long descriptions are cut in chunks
TOP_CHUNKS = 3      # an offer's score = mean similarity of its best chunks

_model = None
_offer_cache = {}   # offer id -> chunk vectors (an offer is embedded only once)


def encode(texts):
    """Texts -> L2-normalized vectors. Replace this function to change the model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return np.asarray(_model.encode(texts, normalize_embeddings=True))


def candidate_vector(candidate):
    education = candidate.get("education") or {}
    text = f"{education.get('field_of_study') or ''}. Skills: {', '.join(candidate.get('skills') or [])}"
    return encode([text])[0]


def offer_vectors(offer):
    key = offer["id"]
    if key not in _offer_cache:
        try:
            skills = ", ".join(json.loads(offer.get("technical_skills") or "[]"))
        except (ValueError, TypeError):
            skills = ""
        text = f"{offer['title']}. {skills}. {offer['description']}"
        chunks = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
        _offer_cache[key] = encode(chunks)
    return _offer_cache[key]


def match_score(cand_vec, offer):
    """0-100 score: the average similarity of the offer's best chunks."""
    chunk_vectors = offer_vectors(offer)               # one row per chunk of the offer

    # 1. Similarity between the student and each chunk
    similarities = (chunk_vectors @ cand_vec).tolist()

    # 2. Keep only the best chunks
    similarities.sort(reverse=True)                    # highest first
    best = similarities[:TOP_CHUNKS]

    # 3. Average them and scale to 0-100
    average = sum(best) / len(best)
    return round(100 * average, 1)

def get_score(pair):
    """Return the score of an (id, score) pair. Used to sort the offers."""
    return pair[1]


def rank_offers(candidate, offers, with_scores=False):
    """offers: rows (dict / pandas Series) with id, title, description, technical_skills."""
    cand_vec = candidate_vector(candidate)

    # 1. Compute the score of every offer
    scored = []
    for offer in offers:
        offer_id = int(offer["id"])
        score = match_score(cand_vec, offer)
        scored.append((offer_id, score))

    # 2. Sort from the highest score to the lowest
    scored.sort(key=get_score, reverse=True)

    # 3. Return the pairs, or only the ids
    if with_scores:
        return scored

    ids = []
    for offer_id, score in scored:
        ids.append(offer_id)
    return ids