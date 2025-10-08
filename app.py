from dotenv import load_dotenv
load_dotenv()  # loads SERPAPI_API_KEY, GEMINI_API_KEY, etc. from .env
import io
import os
import re
from typing import List, Tuple, Dict
import streamlit as st


# --- Optional NLP imports (lazy loaded in functions to speed startup) ---
# nltk, spacy, transformers, pdfplumber, yake, sumy

APP_TITLE = "NLP Summarizer & Explainer"
DESCRIPTION = (
    "Upload a PDF or paste text. This app shows an *explainable* summarization pipeline: "
    "tokenization, lemmatization, POS/NER, a dependency parse, keyword extraction, "
    "and both extractive & abstractive summaries.\n\n"
    "It also includes a simple 'agentic' orchestrator that chooses an approach based on input length, "
    "chunks long inputs, and records its plan & steps."
)

# ----------------------------- Utilities ----------------------------- #

def load_pdf_text(file) -> str:
    import pdfplumber
    text_parts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()

def ensure_nltk():
    import nltk
    # Always ensure classic punkt (works with Sumy and NLTK 3.8.x)
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        nltk.download("punkt", quiet=True)

    # Best-effort: support newer NLTK layouts WITHOUT failing if absent
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except (LookupError, OSError):
        # Do not force-download; many 3.8.x installs don’t have this.
        # If you want, uncomment:
        # try:
        #     nltk.download("punkt_tab", quiet=True)
        # except Exception:
        #     pass
        pass

def compress_sentence_spacy(sent_text: str) -> str:
    """
    Keep core content words (NOUN, PROPN, VERB, ADJ, NUM) and drop filler/aux/det.
    Verbs are lemmatized to reduce fluff. Keeps original order for readability.
    """
    doc = nlp_spacy(sent_text)
    keep_tags = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}
    drop_deps = {"punct", "det", "aux", "mark", "case"}
    words = []
    for tok in doc:
        if tok.dep_ in drop_deps:
            continue
        if tok.pos_ in keep_tags and not tok.is_stop:
            # compact verbs a bit
            words.append(tok.lemma_.lower() if tok.pos_ == "VERB" else tok.text)
    if not words:
        return sent_text.strip()
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def bulletize(text: str, max_points: int = 5) -> list[str]:
    """
    Sentence split -> compress -> deduplicate -> top-N.
    Works on the abstractive summary (preferred) or extractive one.
    """
    ensure_nltk()
    from nltk.tokenize import sent_tokenize
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    compressed = [compress_sentence_spacy(s) for s in sents]
    # dedupe case-insensitively
    seen = set()
    uniq = []
    for c in compressed:
        key = c.lower()
        if key and key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq[:max_points]

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    """Greedy character-based chunking with sentence-aware soft boundaries."""
    # Prefer splitting on sentences when possible
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    buf = []
    size = 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if size + len(s) + 1 <= max_chars:
            buf.append(s)
            size += len(s) + 1
        else:
            if buf:
                chunks.append(" ".join(buf))
            buf = [s]
            size = len(s)
    if buf:
        chunks.append(" ".join(buf))
    return chunks if chunks else [text]


# ---------- Web search + Gemini helpers ----------

from typing import List, Dict

def find_related_articles(query: str, k: int = 8) -> List[Dict[str, str]]:
    """
    Returns [{'title': ..., 'link': ...}, ...] via SerpAPI.
    Requires SERPAPI_API_KEY in env or .env.
    """
    import os, requests
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        st.info("To enable web search, set SERPAPI_API_KEY in your .env or environment.")
        return []

    params = {
        "engine": "google",
        "q": query,
        "num": k,
        "api_key": api_key,
        "hl": "en",
        "safe": "active",
    }
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.error(f"Web search failed: {e}")
        return []

    results: List[Dict[str, str]] = []
    for item in (data.get("organic_results") or []):
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        if title and link:
            results.append({"title": title, "link": link})
        if len(results) >= k:
            break
    return results


def gemini_rerank_and_expand(query: str, candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Use Gemini to expand/rerank. Requires GEMINI_API_KEY and google-generativeai installed.
    If not available or anything fails, return candidates unchanged.
    """
    import os, json
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or not candidates:
        return candidates

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Try a couple of model name variants for compatibility.
        model_names = ["gemini-2.0-flash", "gemini-1.5-flash", "models/gemini-1.5-flash"]
        last_err = None
        model = None
        for name in model_names:
            try:
                model = genai.GenerativeModel(
                    name,
                    generation_config={"response_mime_type": "application/json"}
                )
                break
            except Exception as e:
                last_err = e
                model = None
        if model is None:
            st.info(f"Gemini model unavailable; skipping re-rank ({last_err}).")
            return candidates

        payload = {
            "topic": query,
            "candidates": candidates,
            "k": min(8, max(1, len(candidates))),
            "criteria": ["relevance", "credibility", "diversity"]
        }

        prompt = (
            "Given a topic and a list of candidate links (title, link), "
            "return a JSON object with key 'items' as a list of up to k objects {title, link}. "
            "Choose credible, diverse, and highly relevant sources. "
            "If uncertain, just return the original candidates in the same shape.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        resp = model.generate_content(prompt)

        # Some versions expose JSON via resp.text; others via parts. Prefer JSON in text.
        text = (getattr(resp, "text", None) or "").strip()

        # Try direct JSON first
        try:
            data = json.loads(text)
            items = data.get("items", [])
        except Exception:
            # Fallback: look for first JSON object in the text
            import re
            m = re.search(r"\{.*\}", text, flags=re.S)
            if not m:
                return candidates
            try:
                data = json.loads(m.group(0))
                items = data.get("items", [])
            except Exception:
                return candidates

        out: List[Dict[str, str]] = []
        for it in items:
            if isinstance(it, dict) and it.get("title") and it.get("link"):
                out.append({"title": str(it["title"]).strip(), "link": str(it["link"]).strip()})
        return out[: payload["k"]] if out else candidates

    except Exception as e:
        st.info(f"Gemini re-ranking skipped ({e}).")
        return candidates

# ----------------------------- NLP Steps ----------------------------- #

def nlp_spacy(text: str):
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp(text)


def lemmatize_tokens_spacy(doc) -> List[Tuple[str, str, str]]:
    """Return list of (token, lemma, POS)."""
    return [(t.text, t.lemma_, t.pos_) for t in doc if not t.is_space]


def extract_ner_spacy(doc) -> List[Tuple[str, str]]:
    return [(ent.text, ent.label_) for ent in doc.ents]


def render_dependency_parse(doc, sent_index: int = 0):
    from spacy import displacy
    sents = list(doc.sents)
    if not sents:
        st.info("No sentences detected for parsing.")
        return
    sent_index = max(0, min(sent_index, len(sents) - 1))
    html = displacy.render(sents[sent_index], style="dep", options={"compact": True}, page=True)
    st.components.v1.html(html, height=300, scrolling=True)


def extract_keywords(text: str, top_k: int = 10) -> List[Tuple[str, float]]:
    try:
        import yake
    except ImportError:
        # Attempt a lazy install for local runs (Streamlit Cloud disables this)
        try:
            import sys, subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yake", "--quiet"])  # noqa
            import yake  # type: ignore
        except Exception:
            return []
    kw = yake.KeywordExtractor(top=top_k, stopwords=None)
    scored = kw.extract_keywords(text)
    # returns list of (keyword, score). Lower is better; convert to 1/score for intuitiveness
    res = []
    for k, score in scored:
        try:
            inv = 1.0 / (score + 1e-9)
        except Exception:
            inv = 0.0
        res.append((k, inv))
    # sort by inv desc
    res.sort(key=lambda x: x[1], reverse=True)
    return res


# Extractive summary (TextRank via sumy)

def extractive_textrank(text: str, max_sent_ratio: float = 0.3) -> Tuple[str, List[Tuple[str, float]]]:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.utils import get_stop_words
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from collections import Counter
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import re

    nltk.download('punkt', quiet=True)
    sentences_all = sent_tokenize(text)
    total_sent = len(sentences_all)
    if total_sent == 0:
        return text, []

    # Dynamically decide how many sentences to keep
    keep = max(1, int(total_sent * max_sent_ratio))

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summarizer.stop_words = get_stop_words("english")
    summary_sents = list(summarizer(parser.document, keep))
    summary_sents = [str(s).strip() for s in summary_sents]

    # Redundancy filtering using cosine similarity
    vectorizer = TfidfVectorizer().fit_transform(summary_sents)
    sim_matrix = cosine_similarity(vectorizer)
    final_sents = []
    for i, s in enumerate(summary_sents):
        if not any(sim_matrix[i, j] > 0.7 for j in range(i) if j < len(final_sents)):
            final_sents.append(s)

    summary_text = " ".join(final_sents)

    # Compute rough scores (for UI transparency)
    words = re.findall(r"[A-Za-z']+", text.lower())
    freqs = Counter(words)
    scored = []
    for s in sentences_all:
        tokens = [w.lower() for w in word_tokenize(s)]
        score = sum(freqs.get(w, 0) for w in tokens) / (len(tokens) + 1e-6)
        scored.append((s, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    return summary_text, scored


# Abstractive summary (Transformers pipeline)

def abstractive_summary(text: str, model_name: str = "sshleifer/distilbart-cnn-12-6", max_length: int = 180, min_length: int = 60) -> str:
    from transformers import pipeline
    summarizer = pipeline("summarization", model=model_name)
    # Handle long inputs with sliding window
    chunks = chunk_text(text, max_chars=2500)
    parts = []
    for ch in chunks:
        out = summarizer(ch, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
        parts.append(out)
    # If multiple parts, compress again
    combined = " \n".join(parts)
    if len(parts) > 1:
        out2 = summarizer(combined, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
        return out2
    return combined


# ----------------------------- Agent Orchestrator ----------------------------- #

class SummarizationAgent:
    """Simple agent that plans steps based on length and goal.

    - If input is short (<2000 chars): run spaCy analysis, keywords, both extractive & abstractive, then pick best.
    - If medium (2000-8000): run extractive first to compress, then abstractive on compressed text.
    - If long (>8000): chunk -> extractive per chunk -> stitch -> abstractive.
    Records a plan and steps executed for transparency.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.plan: List[str] = []
        self.steps: List[str] = []

    def decide(self, text: str) -> Dict:
        n = len(text)
        if n < 2000:
            self.plan = [
                "Analyze linguistics with spaCy",
                "Extract keywords (YAKE)",
                "Run extractive TextRank",
                f"Run abstractive model ({self.model_name})",
                "Choose final summary (prefer abstractive if fluent and consistent)",
            ]
        elif n < 8000:
            self.plan = [
                "Extractive TextRank to ~6-10 sentences",
                f"Abstractive compression with {self.model_name}",
                "Linguistic analysis on original (spaCy)",
            ]
        else:
            self.plan = [
                "Chunk document",
                "Per-chunk extractive TextRank",
                f"Stitch & abstractive with {self.model_name}",
                "Linguistic analysis on sample",
            ]
        return {"length": n, "plan": self.plan}

    def run(self, text: str) -> Dict:
        info = self.decide(text)
        result: Dict = {"agent": info, "outputs": {}}

        # Linguistic analysis (spaCy)
        self.steps.append("spaCy analysis start")
        doc = nlp_spacy(text)
        lemmas = lemmatize_tokens_spacy(doc)
        ents = extract_ner_spacy(doc)
        self.steps.append("spaCy analysis done")

        # Keywords
        self.steps.append("Keywords start")
        keywords = extract_keywords(text, top_k=12)
        self.steps.append("Keywords done")

        # Extractive summary
        self.steps.append("Extractive start")
        ext_sum, sent_scores = extractive_textrank(text, max_sent_ratio=0.6)

        self.steps.append("Extractive done")

        # Abstractive summary (possibly on compressed text for long docs)
        self.steps.append("Abstractive start")
        base_for_abs = ext_sum if len(text) > 8000 else text
        try:
            abs_sum = abstractive_summary(base_for_abs, model_name=self.model_name)
        except Exception as e:
            abs_sum = f"[Abstractive summarization failed: {e}]\nFalling back to extractive summary."
        self.steps.append("Abstractive done")

        # Simple selection logic
        # Prefer the shorter, more coherent summary
        # Selection logic
        if "failed" in abs_sum.lower():
            final = ext_sum
        else:
            if len(abs_sum.split()) < len(ext_sum.split()) * 0.9:
                final = abs_sum
            else:
                final = ext_sum

        final = normalize_whitespace(final)

        # Pointwise bullets (prefer fluent abstractive; fallback to extractive)
        source_for_points = abs_sum if "failed" not in abs_sum.lower() else ext_sum
        bullets = bulletize(source_for_points, max_points=5)

# Build outputs (include bullets)
        result["outputs"] = {
            "lemmas": lemmas[:3000],  # keep UI light
            "entities": ents,
            "keywords": keywords,
            "extractive": ext_sum,
            "abstractive": abs_sum,
            "final_summary": final,
            "bullets": bullets,               # <-- added
            "sent_scores": sent_scores[:50],
        }
        result["agent"]["steps"] = self.steps

        return result



# ----------------------------- Streamlit UI ----------------------------- #

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(DESCRIPTION)

    with st.sidebar:
        st.header("Input")
        source = st.radio("Choose input type", ["Paste Text", "Upload PDF"], horizontal=True)
        model_name = st.selectbox(
            "Abstractive model",
            [
                "sshleifer/distilbart-cnn-12-6",
                "facebook/bart-large-cnn",
                "t5-small",
            ],
            index=0,
            help="Smaller models run faster locally; larger models are higher quality but slower.",
        )
        max_chars = st.slider("Chunk size (chars)", 1000, 8000, 3000, 500)
        sent_for_parse = st.number_input("Sentence index to visualize (0-based)", min_value=0, value=0, step=1)

        user_text = ""
        if source == "Paste Text":
            user_text = st.text_area("Paste your text here", height=220)
        else:
            pdf_file = st.file_uploader("Upload a PDF", type=["pdf"]) 
            if pdf_file is not None:
                user_text = load_pdf_text(pdf_file)
                st.success(f"Extracted {len(user_text):,} characters from PDF.")

        run_btn = st.button("Run Summarizer & Explainer", type="primary")

    # Guard
    if run_btn:
        agent = SummarizationAgent(model_name=model_name)
        with st.spinner("Running analysis and summarization..."):
            results = agent.run(user_text)
        # persist for later reruns (so other buttons can work)
        st.session_state["results"] = results
        st.session_state["input_text"] = user_text

    elif "results" in st.session_state:
    # reuse previous results so the page doesn't reset
        results = st.session_state["results"]
        user_text = st.session_state.get("input_text", "")

    else:
    # first visit and no results yet -> stop
        st.stop()

    # Agent run
    agent = SummarizationAgent(model_name=model_name)
    with st.spinner("Running analysis and summarization..."):
        results = agent.run(user_text)

    # Tabs
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Agent Plan",
        "Linguistics",
        "Parse Tree",
        "Keywords",
        "Summaries",
        "Sentence Scores",
        "Pointwise Bullets",
    ])

    with tab0:
        st.subheader("Agentic Orchestrator")
        st.write("**Document length:**", f"{results['agent']['length']:,} characters")
        st.write("**Plan:**")
        for step in results["agent"]["plan"]:
            st.markdown(f"- {step}")
        st.write("**Executed steps:**")
        for s in results["agent"]["steps"]:
            st.markdown(f"- {s}")

    with tab1:
        st.subheader("Tokens, Lemmas, POS")
        lemmas = results["outputs"]["lemmas"]
        if lemmas:
            st.dataframe(
                {"token": [t for t, _, _ in lemmas], "lemma": [l for _, l, _ in lemmas], "POS": [p for _, _, p in lemmas]},
                use_container_width=True,
            )
        ents = results["outputs"].get("entities", [])
        st.subheader("Named Entities")
        if ents:
            st.dataframe({"text": [e for e, _ in ents], "label": [l for _, l in ents]}, use_container_width=True)
        else:
            st.info("No named entities detected.")

    with tab2:
        st.subheader("Dependency Parse (select sentence index in sidebar)")
        doc = nlp_spacy(user_text)
        try:
            render_dependency_parse(doc, sent_index=int(sent_for_parse))
        except Exception as e:
            st.error(f"Failed to render parse: {e}")

    with tab3:
        st.subheader("Keyword Extraction (YAKE)")
        kws = results["outputs"].get("keywords", [])
        if kws:
            st.dataframe({"keyword": [k for k, _ in kws], "salience": [round(s, 3) for _, s in kws]}, use_container_width=True)
        else:
            st.info("No keywords available (YAKE missing or failed). Try installing `yake`.")

    with tab4:
        st.subheader("Final Summary")
        st.success(results["outputs"]["final_summary"]) 
        with st.expander("Show Abstractive Summary"):
            st.write(results["outputs"]["abstractive"]) 
        with st.expander("Show Extractive Summary"):
            st.write(results["outputs"]["extractive"]) 

    with tab5:
        st.subheader("Sentence Importance (frequency-based approx)")
        scored = results["outputs"].get("sent_scores", [])
        if scored:
            st.dataframe({"sentence": [s for s, _ in scored], "score~": [round(v, 2) for _, v in scored]}, use_container_width=True)
        else:
            st.info("No sentence scores computed.")
    with tab6:
        st.subheader("Pointwise summary (concise bullets)")

        # Keep this section compact: show the top N bullets only (no expanders or long text).
        bullets = results["outputs"].get("bullets", []) or []
        # Heuristic: take the first 3–6 key bullets
        max_points = st.slider("Max bullets", 3, 10, min(6, max(3, len(bullets) or 5)))
        top_bullets = bullets[:max_points]

        if top_bullets:
            for b in top_bullets:
                st.markdown(f"- {b}")
        else:
            st.info("No bullets produced for this input.")

        st.markdown("---")
        st.subheader("Related reading")

        seed_text = results["outputs"].get("abstractive") or results["outputs"].get("extractive") or ""
        topic_hint = seed_text[:500] if seed_text else ""

        # initialize
        st.session_state.setdefault("related_query", "")
        st.session_state.setdefault("related_results", [])
        st.session_state.setdefault("related_error", "")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            find_btn = st.button("Find related articles", key="btn_related_articles")
        with col_b:
            clear_btn = st.button("Clear results", key="btn_clear_related")

        if clear_btn:
            st.session_state["related_query"] = ""
            st.session_state["related_results"] = []
            st.session_state["related_error"] = ""

        if find_btn:
            if not topic_hint:
                st.warning("No summary available to seed the search.")
            else:
                with st.spinner("Searching the web..."):
                    try:
                        base_results = find_related_articles(topic_hint, k=8)
                        ranked = gemini_rerank_and_expand(topic_hint, base_results) if base_results else []
                        st.session_state["related_query"] = topic_hint
                        st.session_state["related_results"] = ranked if ranked else base_results
                        st.session_state["related_error"] = "" if (ranked or base_results) else "No web results found."
                    except Exception as e:
                        st.session_state["related_query"] = topic_hint
                        st.session_state["related_results"] = []
                        st.session_state["related_error"] = f"{type(e).__name__}: {e}"

        # render persisted results
        if st.session_state["related_error"]:
            st.info(st.session_state["related_error"])
        elif st.session_state["related_results"]:
            st.caption(f"Query seed: {st.session_state['related_query'][:120]}…")
            for i, item in enumerate(st.session_state["related_results"], start=1):
                title = (item.get("title") or f"Result {i}").strip()
                link = (item.get("link") or "").strip()
                if link:
                    st.markdown(f"{i}. [{title}]({link})")
                else:
                    st.markdown(f"{i}. {title}")
            st.caption("Tip: set GEMINI_API_KEY to enable Gemini re-ranking (optional).")
        else:
            st.caption("Click **Find related articles** to fetch sources.")


    st.divider()
    st.caption("Notes: TextRank via sumy; keywords via YAKE; linguistics via spaCy; abstractive via Hugging Face transformers. Parse visualization uses spaCy displaCy. This app records an agentic plan and steps for transparency.")


if __name__ == "__main__":
    main()
