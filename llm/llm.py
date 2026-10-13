"""
LLM layer — University Admission & Student Info Voice Agent.
Standalone tri-lingual voice agent with pure in-prompt hardcoded knowledge.
"""

import asyncio
import contextlib
import json
import re
from typing import Generator, AsyncGenerator, Any
import httpx

import config
Database = None
def transliterate_student_name(name: str) -> str:
    return name
from utils import (
    is_hangup_intent,
    is_agent_farewell,
    is_simple_greeting,
    is_abusive_intent,
    is_who_are_you_intent,
    clean_speech_text,
    is_prompt_leak,
    HOLD_PHRASES,
    HOLD_PHRASE_REGEX,
    query_needs_db_or_rag,
    get_error_message,
    ABUSIVE_CALM_PROMPTS,
    WHO_ARE_YOU_PROMPTS,
    extract_semester,
    is_semester_query,
    get_semester_clarification_phrase,
)
from config.prompt_loader import (
    get_welcome_message,
    get_call_hangup_prompt,
    get_system_prompt,
    detect_active_domains,
)

# Live-fetching wrappers instead of import-time frozen constants.
def _welcome_live() -> str:
    return get_welcome_message()

def _system_prompt_live(domains=None) -> str:
    return get_system_prompt(domains)

def _hangup_prompt_live() -> str:
    return get_call_hangup_prompt()

WELCOME_MESSAGE: str = get_welcome_message()
SYSTEM_PROMPT: str = get_system_prompt()
CALL_HANGUP_PROMPT: str = get_call_hangup_prompt()


def transform_query(user_text: str) -> str:
    """
    Transforms and sanitizes user input before passing to LLM.
    Removes unprintable control characters and normalizes whitespace.
    """
    if not user_text:
        return ""
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', user_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


_shared_db = None

def get_shared_db():
    return None

FOLLOW_UP_HELP_PROMPTS = {
    "gu": "બીજી કોઈ મદદ જોઈએ છે?",
    "hi": "क्या आपको और कोई मदद चाहिए?",
    "en": "Do you need any other help?",
}

AGREEMENT_PATTERNS = [
    r"\b(?:ok|okay|got\s*it|fine|understood|clear|alright|all\s*right|sure|makes\s*sense)\b",
    r"(?:ઠીક\s*છે|બરાબર|સમજાઈ\s*ગયું|સારું)",
    r"(?:ठीक\s*है|समझ\s*(?:गया|गई)|बढ़िया|अच्छा)",
    r"\b(?:thik\s*hai|theek\s*hai|samajh\s*gaya|samajh\s*gayi|thik\s*che|theek\s*che|barabar|samjai\s*gayu)\b",
]
_COMPILED_AGREEMENT = [re.compile(p, re.IGNORECASE) for p in AGREEMENT_PATTERNS]

def is_agreement_intent(text: str) -> bool:
    clean = (text or "").strip().lower()
    if not clean:
        return False
    return any(rx.search(clean) for rx in _COMPILED_AGREEMENT)


_MULTILINGUAL_EXPANSIONS = {
    r'(?:admission|एडमिशन|प्रवेश|दाखिला|દાખલ|એડમિશન|apply|आवेदन|અરજી)': 'admission apply eligibility requirements',
    r'(?:eligibility|एलिजिबिलिटी|पात्रता|લાયકાત|eligible|योग्य)': 'eligibility criteria requirements percentage',
    r'(?:fees?|fee structure|फी|फीસ|शुल्क|ખર્ચ|ફી|cost)': 'fee structure tuition fees annual 166950 152000 55125 52500 1st year 2nd year B.Tech M.Tech',
    r'(?:m\.?tech|एमटेक|एम\.टेक|એમટેક|એમ\.ટેક)': 'M.Tech Master of Technology postgraduate fee structure 55125 52500',
    r'(?:b\.?tech|बीटेक|बी\.टेक|બીટેક|engineering|इंजीनियरिंग)': 'B.Tech Bachelor of Technology Engineering',
    r'(?:cse|it|computer|कम्प्यूटर|कंप्यूटर|કોમ્પ્યુટર|ઇન્ફોર્મેશન)': 'Computer Science Engineering Information Technology IT',
    r'(?:ddu|डीडीयू|धर्मसिंह|ધરમસિંહ)': 'DDU Dharamsinh Desai University Nadiad',
    r'(?:attendance|अटेंडेंस|हाजिरी|उपस्थिति|હાજરી)': 'attendance policy minimum percentage rules',
    r'(?:placement|प्लेसमेंट|नौकरी|पैकेज|सैलरी|પ્લેસમેન્ટ|salary|package)': 'placement highest average package recruiters companies',
    r'(?:batch\s*2026|2026\s*batch|૨૦૨૬|2026|highlights?|statistics?|summary|હાઈલાઈટ્સ|हाइलाइट्स)': 'Highlights of B.Tech IT 2026 Batch placement summary offers 108 106 placed highest salary 13.4 LPA average 5.5 LPA higher studies 30',
    r'(?:hackathons?|હેકાથોન|हैकथॉन|suisign|skillscreen|synapsemodel|competition|competitions|ઈનામ|पुरस्कार)': 'hackathon Sui Overflow ETHGlobal Codeversity prize winner Abhimanyu Ajudiya Pruthviraj Parmar Harsh Manek Sumit Mishra project',
    r'(?:gate|ncat|ielts|toefl|pte|ગેટ|પર્સન્ટાઈલ|गेट|रैंक|percentile|air|rank)': 'GATE NCAT IELTS TOEFL PTE percentile rank AIR Darpan Vora Dodiya Aditya Sorathiya Utsav Barkha Lahori Patel Nisarg competitive exam toppers',
    r'(?:research|paper|papers|publication|conference|conferences|રિસર્ચ|પેપર|સંશોધન|रिसर्च|पेपर|शोध|सम्मेलन)': 'research papers conference presented by students Shreyas Warrier Om Patel Lavi Garg Gaurang Agrawal Nitya Dhagat Devanshie Patel Deep Govindvira Govinda Prajapati Harmit Patel Kunj Patel',
    r'(?:hostel|हॉस्टल|छात्रावास|હોસ્ટેલ)': 'hostel timings curfew accommodation rules',
    r'(?:scholarship|स्कॉलरशिप|छात्रवृत्ति|સ્કોલરશિપ)': 'scholarship merit financial aid',
    r'(?:syllabus|curriculum|subjects?|સબ્જેક્ટ|સબ્જેક્ટ્સ|સબજેક્ટ|સબજેક્ટ્સ|વિષય|વિષયો|સિલેબસ|અભ્યાસક્રમ|सिलेबस|पाठ्यक्रम)': 'syllabus curriculum subjects examination scheme',
    r'(?:semester|सेमेस्टर|સેમેસ્ટર|સેમ|સેમિસ્ટર)': 'semester academic term subjects syllabus scheme',
    r'(?:એક|વન|પહેલું|પહેલા|૧|\b1\b|\bone\b|\bfirst\b|पहला|पहले|प्रथम)': 'semester 1 semester I first semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:બે|ટુ|ટૂ|બીજું|બીજા|૨|\b2\b|\btwo\b|\bsecond\b|दूसरा|दूसरे|द्वितीय)': 'semester 2 semester II second semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:ત્રણ|થ્રી|ત્રીજું|ત્રીજા|૩|\b3\b|\bthree\b|\bthird\b|तीसरा|तीसरे|तृतीय)': 'semester 3 semester III third semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:ચાર|ફોર|ચોથું|ચોથા|૪|\b4\b|\bfour\b|\bfourth\b|चौथा|चौथे|चतुर्थ)': 'semester 4 semester IV fourth semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:પાંચ|ફાઈવ|પાંચમું|પાંચમા|૫|\b5\b|\bfive\b|\bfifth\b|पांचवा|पांचवे|पंचम)': 'semester 5 semester V fifth semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:(?<![\u0a80-\u0aff])છ(?![\u0a80-\u0aff])|સિક્સ|૬|\b6\b|\bsix\b|\bsixth\b|छठा|छठे|षष्ठ)': 'semester 6 semester VI sixth semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:સાત|સેવન|સાતમું|સાતમા|૭|\b7\b|\bseven\b|\bseventh\b|सातवां|सातवें|सप्तम)': 'semester 7 semester VII seventh semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:આઠ|એઈટ|આઠમું|આઠમા|૮|\b8\b|\beight\b|\beighth\b|आठवां|आठवें|अष्टम)': 'semester 8 semester VIII eighth semester B.Tech IT Course Structure Teaching Scheme',
}

def expand_multilingual_query(text: str) -> str:
    """Expands multilingual voice queries with English semantic tokens for ChromaDB indexing."""
    terms = [text]
    lower = text.lower()
    for pattern, eng_equiv in _MULTILINGUAL_EXPANSIONS.items():
        if re.search(pattern, lower):
            terms.append(eng_equiv)
    return " ".join(terms)


# Fix #13: Pre-compiled frozensets compiled once at import time for O(1) per-turn intersection
_GUJ_INDICATORS: frozenset[str] = frozenset({
    "shu", "ketli", "ketla", "ketlu", "ketlo", "che", "chhe", "chho", "chhu", "maate", "mate", "nathi", "aapo",
    "aapjo", "tame", "tamne", "tamara", "tamari", "tamaru", "tamaro", "tamare", "aavde", "aavadse", "janavo",
    "janavjo", "kem", "vishe", "pucho", "puchjo", "aabhar", "namaste", "gujarati", "kai", "kayi", "kayo", "kayu",
    "kyo", "bhanela", "bhanva", "kaho", "kahu", "kejo", "mane", "haji", "haju", "maru", "maro", "mari", "mara",
    "mare", "dikro", "dikra", "dikri", "chokro", "chokra", "chokri", "su", "karvanu", "karvo", "karvi", "karva",
    "batavo", "batavjo", "aavshe", "aave", "aavyo", "aavi", "aavya", "karo", "karjo", "karsho", "joie", "joiye",
    "joye", "barabar", "saras", "saaru", "saru", "bolu", "boljo", "bolo", "bol", "muki", "muko", "pan", "ane",
    "suche", "nu", "nnu", "ni", "nni", "no", "nno", "na", "nna", "num", "ne", "thi", "mathi", "maathi", "upar",
    "male", "malse", "malshe", "mali", "malya", "hu", "hun", "hoon", "apne", "aapne", "apdu", "aapdu", "apda",
    "aapda", "apdi", "aapdi", "amne", "tene", "tenu", "teni", "teno", "tena", "emne", "enu", "eni", "eno", "ena",
    "kya", "kyan", "kyathi", "kyare", "kon", "kone", "koni", "konu", "kona", "kashu", "hato", "hati", "hatu",
    "hata", "hoy", "hoyse", "thay", "thase", "thashe", "thayu", "thaya", "thayi", "khabar", "vaat", "vastu",
    "mahiti", "mahitie", "thodu", "thoda", "thodi", "vadhu", "ochhu", "ochha", "ochhi", "badha", "badhi", "badhu",
    "tamam", "bijo", "biji", "bija", "aaje", "kaale", "kal", "pahela", "pachi", "andar", "bahar", "samjao",
    "desho", "lavo", "lai", "moklo", "mokljo"
})

_HIN_INDICATORS: frozenset[str] = frozenset({
    "kya", "kitna", "kitni", "kitne", "hai", "hain", "ho", "hoon", "hun", "batao", "bata", "bataiye", "batayein",
    "sakta", "sakti", "sakte", "sakate", "sakthi", "sakthe", "hoga", "hogi", "hoge", "hua", "hui", "hue",
    "nahi", "nahin", "na", "mat", "kripya", "aapki", "aapka", "aapke", "aap", "tum", "kaise", "kaisa", "kaisi",
    "kuch", "baare", "mein", "me", "aur", "chahiye", "dena", "dijiye", "de", "do", "kaun", "kaha", "kahan",
    "kidhar", "kab", "kyon", "kyu", "kyun", "hindi", "mujhe", "mera", "meri", "mere", "uska", "uski", "uske",
    "unka", "unki", "unke", "beta", "beti", "naam", "bolo", "boliye", "kaho", "kahiye", "raha", "rahi", "rahe",
    "tha", "thi", "the", "thhe", "iski", "iska", "iske", "inki", "inka", "inke", "hum", "humara", "humari",
    "humare", "achha", "accha", "theek", "shukriya", "dhanyawad", "dhanyavad", "ki", "ka", "ke", "ko", "se",
    "toh", "to", "bhai", "yaar", "ji", "kardo", "kar", "karo", "kijiye", "rakh", "rakho", "kaat", "kat",
    "par", "pe", "sirf", "bas", "ab", "abhi", "bhi", "milega", "milegi", "milenge", "milta", "milti", "milte",
    "sun", "suno", "suniye", "bhejo", "pata", "bahut", "zyada", "kam", "pehle", "baad"
})

_ALL_INDIC_WORDS: frozenset[str] = _GUJ_INDICATORS | _HIN_INDICATORS

_ENG_INDICATORS: frozenset[str] = frozenset({
    "what", "when", "where", "how", "why", "who", "which", "is", "are", "can",
    "tell", "eligibility", "fee", "fees", "admission", "curfew", "placement",
    "package", "attendance", "marks", "thank", "goodbye",
    "please", "english", "syllabus", "course", "subject", "semester",
    "credit", "curriculum", "department", "hostel", "rules", "campus", "direct",
})

# English grammar/question words that distinguish a genuine English sentence from isolated keywords/loanwords
_ENGLISH_GRAMMAR_WORDS: frozenset[str] = frozenset({
    "what", "when", "where", "how", "why", "who", "which", "whose", "whom",
    "is", "are", "am", "was", "were", "can", "could", "would", "should", "will",
    "shall", "do", "does", "did", "tell", "please", "give", "explain", "help",
    "provide", "show", "may", "about", "have", "has", "had"
})

# Comprehensive English vocabulary: only sentences with 100% words in this set can qualify as full English lines
_COMMON_ENGLISH_WORDS: frozenset[str] = frozenset({
    "what", "when", "where", "how", "why", "who", "which", "whose", "whom",
    "is", "are", "am", "was", "were", "be", "been", "being", "can", "could",
    "would", "should", "will", "shall", "may", "might", "must", "do", "does",
    "did", "have", "has", "had", "having", "the", "a", "an", "this", "that",
    "these", "those", "it", "its", "i", "me", "my", "mine", "you", "your",
    "yours", "he", "him", "his", "she", "her", "hers", "we", "us", "our",
    "ours", "they", "them", "their", "theirs", "in", "on", "at", "to", "for",
    "with", "about", "from", "by", "into", "through", "during", "before",
    "after", "above", "below", "between", "under", "of", "off", "over", "again",
    "further", "then", "once", "and", "but", "or", "so", "if", "because",
    "as", "until", "while", "both", "all", "any", "some", "no", "not", "only",
    "own", "same", "than", "too", "very", "just", "now", "here", "there",
    "tell", "please", "help", "give", "show", "know", "want", "like", "need",
    "provide", "share", "explain", "detail", "details", "information", "info",
    "admission", "admissions", "fee", "fees", "hostel", "hostels", "curfew",
    "placement", "placements", "package", "packages", "highest", "average",
    "syllabus", "course", "courses", "subject", "subjects", "semester",
    "semesters", "exam", "exams", "examination", "gate", "attendance", "marks",
    "grade", "grades", "result", "results", "eligibility", "criteria",
    "process", "procedure", "department", "college", "university", "ddu",
    "btech", "mtech", "bachelor", "master", "engineering", "faculty", "campus",
    "library", "canteen", "scholarship", "scholarships", "timing", "timings",
    "contact", "number", "email", "address", "location", "student", "students",
    "name", "id", "hello", "hi", "hey", "good", "morning", "afternoon",
    "evening", "thank", "thanks", "thankyou", "bye", "goodbye", "yes", "okay",
    "ok", "fine", "sure", "sorry", "excuse", "sir", "madam", "mam", "maam",
    "call", "agent", "assistant", "speak", "talk", "understand", "english",
    "rule", "rules", "structure", "annual", "year", "first", "second", "third",
    "fourth", "last", "direct", "credit", "credits", "curriculum", "branch",
    "branches", "computer", "information", "technology", "mechanical",
    "chemical", "civil", "electrical", "ec", "ic", "ch", "cl", "mh", "ce", "it",
    "top", "recruiter", "recruiters", "company", "companies", "job", "jobs",
    "offer", "offers", "salary", "salaries", "lpa", "ctc", "internship",
    "internships", "project", "projects", "study", "learn", "teach", "teaching",
    "question", "questions", "ask", "asking", "answer", "answers", "query",
    "queries", "inform", "more", "much", "many", "good", "better", "best",
    "time", "times", "summarize", "summary", "subjective", "objective", "grading",
    "policy", "policies"
})

# Words that overlap between English and Romanized Indic spellings (e.g. 'the', 'me', 'to', 'do', 'no', 'so', 'he', 'or')
# Must not trigger false Indic classification when used in English sentences
_DUAL_INDIC_ENG_WORDS: frozenset[str] = frozenset({"the", "me", "to", "do", "no", "so", "he", "or"})
_UNAMBIGUOUS_GUJ_INDICATORS: frozenset[str] = _GUJ_INDICATORS - _DUAL_INDIC_ENG_WORDS
_UNAMBIGUOUS_HIN_INDICATORS: frozenset[str] = _HIN_INDICATORS - _DUAL_INDIC_ENG_WORDS
_UNAMBIGUOUS_INDIC_WORDS: frozenset[str] = _UNAMBIGUOUS_GUJ_INDICATORS | _UNAMBIGUOUS_HIN_INDICATORS

# Words that indicate Hindi even when phonetically written in Gujarati script
_HINDI_IN_GUJ_WORDS: frozenset[str] = frozenset({
    "બતાઓ", "બતાઈએ", "હૈ", "મેરા", "મેરી", "મેરે", "મુઝે", "ઉસકા", "ઉસકી", "ઉસકે",
    "કિતની", "કિતના", "ચાહિયે", "ક્યા", "આપકા", "આપકી", "કૈસે", "નહીં", "હોગા", "હોગી",
    "અભી", "કરકે", "બતાતી", "બતાતા", "કી", "કા", "કે"
})

_GUJ_SPECIFIC_WORDS: frozenset[str] = frozenset({
    "છે", "નથી", "માટે", "તમાર", "કેમ", "શું", "આપો", "વિશે", "કહો", "જણાવો", "આવડે", "જોઈએ"
})


class LLM:
    def __init__(self, db=None, **kwargs):
        self.db = None
        self.current_lang = "gu"  # Gujarati is the primary language
        self.active_student_id: str | None = None
        self.active_student_name: str | None = None
        self._active_provider = config.LLM_PROVIDER
        self._http_client: httpx.AsyncClient | None = None
        # Initialize conversation history with the assistant's opening welcome message
        self.history: list[dict[str, str]] = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]

    async def get_http_client(self) -> httpx.AsyncClient:
        """Returns a reused persistent httpx client with connection pooling."""
        if self._http_client is None or self._http_client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=60.0)
            self._http_client = httpx.AsyncClient(timeout=35.0, limits=limits)
        return self._http_client

    @contextlib.asynccontextmanager
    async def _borrow_client(self):
        """Borrows persistent HTTP client without closing it on block exit."""
        client = await self.get_http_client()
        yield client

    def detect_turn_language(self, text: str, fallback_lang: str = "gu", stt_lang: str | None = None) -> tuple[str, str, str]:
        """
        Determines caller language for the active turn with zero-latency dynamic switching.
        Hierarchy:
          - Gujarati (Primary)
          - Hindi & English (Secondary)
        Rule: If EVEN ONE word in caller's speech is non-English (Hindi, Hinglish, Gujarati, Gujlish),
        the agent MUST HOLD the previous language without changing to English.
        Only switch to English if 100% of the speech consists of English words.
        Returns: (lang_code, bcp47, directive_prefix)
        """
        clean = (text or "").strip()
        lower = clean.lower()
        words = set(re.findall(r'\b[a-zA-Z]+\b', lower))

        # 0. Standalone phone greetings (e.g. 'hello', 'hi', 'hey', 'halo', 'namaste'):
        # Callers in Gujarat universally answer the phone with 'Hello' as a telephone habit.
        # This MUST NOT switch the agent to English. Retain active conversation language (default Gujarati).
        if is_simple_greeting(clean):
            fb = (fallback_lang or "gu").lower()
            if "hi" in fb:
                return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी लिपि).]:\n"
            elif "en" in fb:
                return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in ENGLISH. You MUST respond 100% in ENGLISH.]:\n"
            else:
                return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Respond in primary language: GUJARATI (ગુજરાતી લિપિ).]:\n"

        # 0b. Abusive language fast-path language detection
        if is_abusive_intent(clean):
            if re.search(r"[\u0900-\u097f]", clean):
                return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller used Hindi. Respond in HINDI.]:\n"
            if re.search(r"[\u0a80-\u0aff]", clean):
                return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller used Gujarati. Respond in GUJARATI.]:\n"
            if any(re.search(p, lower) for p in [r"\b(bhosdina|bhosadina|chodi\s*na|lodha|loda|gadheda|kutro|bakwas\s+bandh|chup\s+tha|chup\s+bes)\b"]):
                return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller used Gujarati. Respond in GUJARATI.]:\n"
            if any(re.search(p, lower) for p in [r"\b(madarchod|bhenchod|behenchod|bhosdike|bhosadike|chutiya|chutiye|harami|kamina|kamine|kameene|gandu|randi|kutti|kutiya|saale|saala|suar|kutte|kutta|bakwas\s+band\s+kar|chup\s+kar|chup\s+baith|aukat)\b"]):
                return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller used Hindi. Respond in HINDI.]:\n"
            if any(re.search(p, lower) for p in [r"\b(fuck|fucking|fucker|fck|motherfucker|bitch|bastard|asshole|dick|pussy|cunt|shit|bullshit|bloody|idiot|stupid|shut\s*up|get\s+lost|go\s+to\s+hell)\b"]):
                return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller used English. Respond in ENGLISH.]:\n"

        # 0c. Identity inquiry fast-path language detection ("who are you")
        if is_who_are_you_intent(clean):
            if re.search(r"[\u0900-\u097f]", clean) or any(re.search(p, lower) for p in [r"\b(aap|tum|kaun|hain|naam|kya)\b"]):
                return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. Respond in HINDI.]:\n"
            if re.search(r"[\u0a80-\u0aff]", clean) or any(re.search(p, lower) for p in [r"\b(tame|tamaru|kon|chho|chhu|shu|che|chhe)\b"]):
                return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in GUJARATI. Respond in GUJARATI.]:\n"
            if any(re.search(p, lower) for p in [r"\b(who|are|you|is|this|name)\b"]):
                return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in ENGLISH. Respond in ENGLISH.]:\n"

        # 1. Native script detection (100% definitive)
        if re.search(r"[\u0900-\u097f]", clean):
            return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी लिपि). Every single word must be in Hindi. Do NOT use any Gujarati words or Gujarati characters under any circumstances.]:\n"

        if re.search(r"[\u0a80-\u0aff]", clean):
            # Check if text actually consists of Hindi words phonetically transcribed in Gujarati script
            has_hin_words = any(hw in clean for hw in _HINDI_IN_GUJ_WORDS)
            has_guj_words = any(gw in clean for gw in _GUJ_SPECIFIC_WORDS)
            if has_hin_words and not has_guj_words:
                return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी लिपि). Every single word must be in Hindi. Do NOT use any Gujarati words or Gujarati characters under any circumstances.]:\n"
            return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in GUJARATI (Primary). You MUST respond 100% in GUJARATI using native Gujarati script (ગુજરાતી). Do NOT use Hindi or English.]:\n"

        # Explicit language request triggers
        if "gujarati" in words:
            return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller requested GUJARATI (Primary). You MUST respond 100% in GUJARATI using native Gujarati script (ગુજરાતી).]:\n"
        if "hindi" in words:
            return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller requested HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी).]:\n"

        guj_overlap = len(words & _UNAMBIGUOUS_GUJ_INDICATORS)
        hin_overlap = len(words & _UNAMBIGUOUS_HIN_INDICATORS)

        # Check if caller used ANY Hindi, Hinglish, Gujarati, or Gujlish words (excluding dual English words like 'the', 'me')
        has_any_indic_word = bool(words & _UNAMBIGUOUS_INDIC_WORDS)

        # STRICT LANGUAGE HOLDING & INDIC RESOLUTION RULE:
        # If EVEN ONE word in user's speech is other than English (Hindi, Hinglish, Gujarati, Gujlish),
        # the agent MUST HOLD that previous language without changing to English!
        fb = (fallback_lang or "gu").lower()
        if has_any_indic_word:
            if guj_overlap > hin_overlap:
                return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in GUJARATI (Primary). You MUST respond 100% in GUJARATI using native Gujarati script (ગુજરાતી). Do NOT use Hindi or English.]:\n"
            elif hin_overlap > guj_overlap:
                return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी लिपि). Every single word must be in Hindi. Do NOT use any Gujarati words or Gujarati characters under any circumstances.]:\n"
            else:
                # Tie or shared Indic tokens (e.g. 'karo', 'to', 'par', etc.) -> HOLD ACTIVE CONVERSATION LANGUAGE
                if "hi" in fb:
                    return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller used Hindi/Hinglish token. Hold active language: HINDI (हिंदी लिपि). Every single word must be in Hindi.]:\n"
                else:
                    return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller used non-English token. Hold active language: GUJARATI (ગુજરાતી લિપિ).]:\n"

        # ONLY switch to English if the user speaks a genuine English sentence:
        # 1. Zero Indic words (has_any_indic_word == False).
        # 2. Caller explicitly asked for English ("english" in words)
        #    OR the sentence consists predominantly of recognized English words (>= 70% English ratio),
        #       contains at least one English grammatical/question marker, and has >= 2 words.
        eng_count = len(words & _COMMON_ENGLISH_WORDS)
        eng_ratio = (eng_count / len(words)) if words else 0.0
        has_english_grammar = bool(words & _ENGLISH_GRAMMAR_WORDS)
        is_full_english_line = (
            not has_any_indic_word
            and (
                "english" in words
                or (len(words) >= 2 and eng_ratio >= 0.70 and has_english_grammar and eng_count >= 2)
            )
        )
        if is_full_english_line:
            return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in ENGLISH. You MUST respond 100% in ENGLISH.]:\n"

        # Check acoustic STT language hint for neutral queries (e.g. 'Aarav Patel', 'STU1')
        if stt_lang and not is_simple_greeting(clean) and not has_any_indic_word:
            stt_l = stt_lang.lower()
            if "hi" in stt_l:
                return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी लिपि). Every single word must be in Hindi.]:\n"
            elif "en" in stt_l and "en" in fb:
                return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in ENGLISH. You MUST respond 100% in ENGLISH.]:\n"
            elif "gu" in stt_l:
                return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in GUJARATI (Primary). You MUST respond 100% in GUJARATI using native Gujarati script (ગુજરાતી).]:\n"

        # Neutral query (e.g. 'STU101' or numbers) -> retain session active language, fallback to Gujarati (Primary)
        if "hi" in fb:
            return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Continue in active conversation language: HINDI (हिंदी लिपि). Every single word must be in Hindi.]:\n"
        elif "en" in fb and not has_any_indic_word:
            return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Continue in active conversation language: ENGLISH.]:\n"
        else:
            return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Respond in primary language: GUJARATI (ગુજરાતી લિપિ).]:\n"

    def _prepare_messages(self, user_text: str, stt_lang: str | None = None) -> list[dict[str, str]]:
        """
        Prepares LLM messages with conversation history and standalone in-prompt knowledge.
        No external database queries or RAG vector retrieval needed.
        """
        clean_user_text = transform_query(user_text)

        active_prompt = get_system_prompt()
        messages = [{"role": "system", "content": active_prompt}]

        # Add previous conversation history turns cleanly
        for turn in self.history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        # Dynamically detect caller's active turn language (Gujarati primary, Hindi/English secondary)
        lang_code, bcp47, lang_cue = self.detect_turn_language(clean_user_text, self.current_lang, stt_lang=stt_lang)
        self.current_lang = lang_code

        is_hangup = is_hangup_intent(clean_user_text)
        directive = ""
        if is_hangup:
            farewell_map = {
                "gu": "તમારો દિવસ શુભ રહે. આભાર!",
                "hi": "आपका दिन शुभ हो। धन्यवाद!",
                "en": "Have a great day. Thank you!",
            }
            exact_farewell = farewell_map.get(lang_code, farewell_map["gu"])
            directive = (
                f"[MANDATORY CLOSING DIRECTIVE: The caller is concluding or declining further help. "
                f'You MUST reply ONLY with this exact farewell sentence in {lang_code}: "{exact_farewell}". '
                f"Do not ask any follow-up questions and do not say any other words.]" + chr(10)
            )
        elif is_agreement_intent(clean_user_text):
            help_phrase = FOLLOW_UP_HELP_PROMPTS.get(lang_code, FOLLOW_UP_HELP_PROMPTS["gu"])
            directive = (
                f'[DIRECTIVE: Caller agreed / acknowledged. Ask the follow-up question in {lang_code}: "{help_phrase}".]' + chr(10)
            )

        current_turn_content = f"{directive}{lang_cue}{clean_user_text}"
        messages.append({"role": "user", "content": current_turn_content})

        MAX_TURN_CHARS = 600
        self.history.append({"role": "user", "content": clean_user_text[:MAX_TURN_CHARS]})
        self._trim_history()

        return messages

    def _trim_history(self):
        max_msgs = config.MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            self.history[:] = self.history[-max_msgs:]

    def _execute_tool(self, tool_name: str, kwargs: dict) -> str:
        """Fallback tool stub. System operates with direct in-prompt knowledge without external DB or RAG."""
        return json.dumps({
            "status": "direct_prompt",
            "message": "All admission, curriculum, GATE subjects, and placement records are hardcoded in the prompt. Answer directly without tool calls."
        })

    def _parse_tool_call(self, text: str) -> tuple[str, dict] | None:
        """Parses TOOL_CALL: tool_name(key="val") pattern."""
        match = re.search(r"TOOL_CALL:\s*(\w+)\((.*)\)", text)
        if not match:
            return None
        tool_name = match.group(1)
        params_str = match.group(2).strip()

        kwargs = {}
        # 1. Try quoted matches first (e.g. key="val with spaces" or key='val with spaces')
        quoted_pairs = re.findall(r'(\w+)\s*=\s*["\']([^"\']*)["\']', params_str)
        if quoted_pairs:
            for k, v in quoted_pairs:
                kwargs[k] = v.strip()
        else:
            # 2. Fallback for unquoted pairs (e.g. identifier=Raj Mehta or department=CSE)
            unquoted_pairs = re.findall(r'(\w+)\s*=\s*([^,)]+)', params_str)
            for k, v in unquoted_pairs:
                kwargs[k] = v.strip().strip("'\"")

        return tool_name, kwargs

    def _format_conversation_history(self) -> str:
        """Formats preceding conversation history into an explicit readable transcript."""
        if not self.history:
            return ""
        lines = ["\n[PREVIOUS CALL CHAT HISTORY]"]
        for turn in self.history:
            role_label = "Caller" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {turn['content']}")
        lines.append("[END OF CHAT HISTORY]\n")
        return "\n".join(lines)

    def check_call_hangup_supervisor(self, user_text: str, agent_text: str = "") -> dict[str, Any]:
        """
        Supervisor evaluation matching the call_hangup_prompt schema:
        {"call_hangup": true|false, "reason": "..."}
        """
        if is_agent_farewell(agent_text):
            return {"call_hangup": True, "reason": "agent closing farewell delivered"}
        if is_hangup_intent(user_text):
            return {"call_hangup": True, "reason": "caller requested hangup or declined further help"}
        return {"call_hangup": False, "reason": "conversation continuing"}

    def check_call_hangup(self, user_text: str, agent_text: str = "") -> bool:
        """
        Fast zero-latency evaluation of whether call should be disconnected.
        """
        res = self.check_call_hangup_supervisor(user_text, agent_text)
        return bool(res.get("call_hangup", False))

    def record_silence_event(self, check_number: int, prompt_spoken: str):
        """
        Records a telephony system silence event into conversation history as a system event,
        never as caller speech.
        """
        event_tag = f"[SYSTEM_EVENT: SILENCE_CHECK_{check_number}]"
        self.history.append({"role": "system", "content": event_tag})
        self.history.append({"role": "assistant", "content": prompt_spoken})
        self._trim_history()


    def _get_provider_request(self, messages: list[dict], provider_override: str | None = None):
        """Builds URL, headers, and payload for streaming LLM calls."""
        provider = provider_override or getattr(self, "_active_provider", config.LLM_PROVIDER)
        if provider == "sarvam":
            headers = {
                "api-subscription-key": config.SARVAM_API_KEY.strip(),
                "Content-Type": "application/json",
            }
            sarvam_model = "sarvam-105b-conversations"
            payload = {
                "model": sarvam_model,
                "messages": messages,
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
                "stream": True,
            }
            return config.SARVAM_CHAT_URL, headers, payload
        elif provider == "groq":
            headers = {
                "Authorization": f"Bearer {config.GROQ_API_KEY.strip()}",
                "Content-Type": "application/json",
            }
            groq_model = config.LLM_MODEL if config.LLM_MODEL and ("llama" in config.LLM_MODEL.lower() or "mixtral" in config.LLM_MODEL.lower() or "gemma" in config.LLM_MODEL.lower()) else "llama-3.3-70b-versatile"
            payload = {
                "model": groq_model,
                "messages": messages,
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
                "stream": True,
            }
            return config.GROQ_URL, headers, payload
        elif provider == "openrouter":
            headers = {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY.strip()}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Shiv-61/Voice-Agent",
                "X-Title": "College Voice Agent",
            }
            payload = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
                "stream": True,
                "include_reasoning": False,
                "reasoning": {"effort": "none"},
            }
            return config.OPENROUTER_URL, headers, payload
        else:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_TOKENS,
                },
            }
            return config.OLLAMA_URL, headers, payload

    def _parse_stream_line(self, line: str, provider_override: str | None = None) -> str:
        """Extracts delta token text from SSE or JSON line."""
        line = line.strip()
        if not line:
            return ""
        provider = provider_override or getattr(self, "_active_provider", config.LLM_PROVIDER)
        if provider in ("sarvam", "openrouter", "groq"):
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    return ""
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        # Strictly extract content, ignore reasoning_content
                        return delta.get("content", "") or ""
                except Exception:
                    return ""
        else:
            try:
                data = json.loads(line)
                return data.get("message", {}).get("content", "") or ""
            except Exception:
                return ""
        return ""

    def reply_stream(self, user_text: str, stt_lang: str | None = None) -> Generator[str, None, None]:
        """
        Synchronously yields response tokens incrementally in real time.
        Used by CLI and synchronous callers.
        """
        messages = self._prepare_messages(user_text, stt_lang=stt_lang)
        turn_lang = self.current_lang

        # Direct zero-latency response for abusive language ("Kripya shanti se baat kare")
        if is_abusive_intent(user_text):
            abusive_map = ABUSIVE_CALM_PROMPTS
            exact_calm = abusive_map.get(turn_lang, abusive_map["gu"])
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": exact_calm})
            yield exact_calm
            return

        # Direct zero-latency response when caller explicitly asks "who are you"
        if is_who_are_you_intent(user_text):
            identity_map = WHO_ARE_YOU_PROMPTS
            exact_identity = identity_map.get(turn_lang, identity_map["gu"])
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": exact_identity})
            yield exact_identity
            return

        # Direct response for phone greetings during ongoing call (e.g. 'hello', 'hi', 'kem cho')
        # Since welcome message already introduced Priya at call start, do NOT repeat the self-intro!
        if getattr(self, "_last_turn_was_greeting", False):
            greeting_map = {
                "gu": "નમસ્તે! કહો, હું તમારી શું મદદ કરી શકું?",
                "hi": "नमस्ते! बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
                "en": "Hello! How can I assist you today?",
            }
            exact_greeting = greeting_map.get(turn_lang, greeting_map["gu"])
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": exact_greeting})
            yield exact_greeting
            return

        # Direct zero-latency response for call hangup intent (e.g. "phone cur kardo", "phone cut kardo", "bye")
        if is_hangup_intent(user_text):
            farewell_map = {
                "gu": "તમારો દિવસ શુભ રહે. આભાર!",
                "hi": "आपका दिन शुभ हो। धन्यवाद!",
                "en": "Have a great day. Thank you!",
            }
            exact_farewell = farewell_map.get(turn_lang, farewell_map["gu"])
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": exact_farewell})
            yield exact_farewell
            return

        # Direct response when caller agrees / acknowledges ("Okay", "Fine", "Understood", "ઠીક છે", "બરાબર", "ठीक है")
        if is_agreement_intent(user_text):
            help_map = FOLLOW_UP_HELP_PROMPTS
            exact_help = help_map.get(turn_lang, help_map["gu"])
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": exact_help})
            yield exact_help
            return

        hold_phrase = HOLD_PHRASES.get(turn_lang, HOLD_PHRASES["gu"])
        hold_phrase_yielded = False

        max_tool_iterations = 3
        current_iteration = 0

        try:
            with httpx.Client(timeout=35.0) as client:
                while current_iteration < max_tool_iterations:
                    current_iteration += 1
                    url, headers, payload = self._get_provider_request(messages)

                    stream_buffer = ""
                    recent_window = ""
                    is_tool_call = False
                    is_streaming_speech = False
                    full_reply = ""
                    in_think = False
                    checked_hold_dedup = not hold_phrase_yielded

                    with client.stream("POST", url, headers=headers, json=payload) as resp:
                        if resp.status_code == 429 and getattr(self, "_active_provider", config.LLM_PROVIDER) == "openrouter" and config.SARVAM_API_KEY:
                            print("⚠️ [llm] OpenRouter quota/rate limit (429) hit. Gracefully switching to Sarvam LLM (sarvam-105b-conversations)...")
                            self._active_provider = "sarvam"
                            current_iteration -= 1
                            continue
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            token = self._parse_stream_line(line)
                            if not token:
                                continue

                            # Filter thinking reasoning tokens (<think>...</think> or <thought>...</thought>)
                            if in_think:
                                if "</think>" in token:
                                    in_think = False
                                    token = token.split("</think>", 1)[1]
                                    if not token:
                                        continue
                                elif "</thought>" in token:
                                    in_think = False
                                    token = token.split("</thought>", 1)[1]
                                    if not token:
                                        continue
                                else:
                                    continue

                            if "<think>" in token or "<thought>" in token:
                                in_think = True
                                tag = "<think>" if "<think>" in token else "<thought>"
                                token = token.split(tag, 1)[0]
                                if not token:
                                    continue

                            # Detect whether model is issuing a tool call with sliding lookback window
                            recent_window = (recent_window + token)[-30:]

                            if not is_streaming_speech and not is_tool_call:
                                stream_buffer += token
                                stripped = stream_buffer.lstrip()
                                if "TOOL_CALL:" in stream_buffer or "TOOL_CALL:" in recent_window:
                                    is_tool_call = True
                                elif "TOOL_CALL:".startswith(stripped):
                                    continue
                                else:
                                    if hold_phrase_yielded and not checked_hold_dedup:
                                        if not stripped:
                                            continue
                                        if HOLD_PHRASE_REGEX.match(stripped):
                                            stream_buffer = HOLD_PHRASE_REGEX.sub("", stripped, count=1)
                                            checked_hold_dedup = True
                                            stripped = stream_buffer.lstrip()
                                            if not stripped:
                                                stream_buffer = ""
                                                continue
                                        elif len(stripped) >= 35 or any(p in stripped for p in [".", "!", "?", "।"]):
                                            checked_hold_dedup = True
                                        else:
                                            continue

                                    has_punct = any(p in stream_buffer for p in [".", "!", "?", "।", "\n"])
                                    if (has_punct or len(stripped) >= 12) and "TOOL_CALL:" not in stream_buffer:
                                        if is_prompt_leak(stream_buffer):
                                            print(f"🛑 [llm] Suppressed prompt leak buffer in stream: {stream_buffer}")
                                            stream_buffer = ""
                                            continue
                                        if stream_buffer:
                                            is_streaming_speech = True
                                            yield stream_buffer
                                            full_reply += stream_buffer
                                            stream_buffer = ""
                                continue

                            if is_tool_call:
                                stream_buffer += token
                            else:
                                if "TOOL_CALL:" in recent_window or "TOOL_CALL:" in token:
                                    is_tool_call = True
                                    is_streaming_speech = False
                                    if "TOOL_CALL:" in full_reply:
                                        idx = full_reply.index("TOOL_CALL:")
                                        stream_buffer = full_reply[idx:]
                                        full_reply = full_reply[:idx]
                                    elif "TOOL_CALL:" in token:
                                        idx = token.index("TOOL_CALL:")
                                        stream_buffer = token[idx:]
                                    else:
                                        stream_buffer = "TOOL_CALL:"
                                    continue
                                full_reply += token
                                if not is_prompt_leak(full_reply):
                                    yield token

                    # If remaining buffer wasn't flushed for short responses
                    if stream_buffer and not is_tool_call:
                        if hold_phrase_yielded and not checked_hold_dedup:
                            checked_hold_dedup = True
                            stripped = stream_buffer.lstrip()
                            if HOLD_PHRASE_REGEX.match(stripped):
                                stream_buffer = HOLD_PHRASE_REGEX.sub("", stripped, count=1)
                        if stream_buffer and not is_prompt_leak(stream_buffer):
                            full_reply += stream_buffer
                            yield stream_buffer
                        stream_buffer = ""

                    if is_tool_call:
                        if not hold_phrase_yielded:
                            yield f"{hold_phrase} "
                            hold_phrase_yielded = True
                        tool_info = self._parse_tool_call(stream_buffer)
                        if tool_info:
                            tool_name, kwargs = tool_info
                            tool_result = self._execute_tool(tool_name, kwargs)
                            tool_synth = f"Please synthesize a short, polite spoken answer for the caller in 1-2 sentences in {turn_lang} language."
                            messages.append({"role": "assistant", "content": stream_buffer})
                            messages.append({
                                "role": "user",
                                "content": f"TOOL_RESULT ({tool_name}): {tool_result}\n{tool_synth}",
                            })
                            continue
                        else:
                            # Tool parse failed, yield buffer as text
                            clean_text = clean_speech_text(stream_buffer)
                            if is_prompt_leak(clean_text):
                                clean_text = "I am here to help you with DDU IT queries. How may I assist you?"
                            clean_history_text = clean_text
                            if hold_phrase_yielded and not clean_history_text.startswith(hold_phrase):
                                clean_history_text = f"{hold_phrase} {clean_history_text}"
                            self.history.append({"role": "assistant", "content": clean_history_text})
                            yield clean_text
                            return
                    else:
                        clean_text = clean_speech_text(full_reply)
                        if is_prompt_leak(clean_text):
                            print(f"🛑 [llm] Suppressed prompt leak from final text: {clean_text}")
                            clean_text = ""

                        if not clean_text.strip():
                            turn_lang, _, _ = self.detect_turn_language(user_text, fallback_lang=self.current_lang)
                            lower_u = user_text.lower()
                            sem_detected = extract_semester(user_text)
                            if any(w in lower_u for w in ["admission", "admissions", "eligible", "eligibility", "criteria", "એડમિશન", "પ્રવેશ", "પાત્રતા", "લાયકાત", "प्रवेश"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ બીટેક આઈટીમાં પ્રવેશ માટે ૧૨ સાયન્સ પીસીએમમાં ઓછામાં ઓછા ૬૦% ગુણ, ગુજકેટ પરીક્ષા અને એસીપીસી મેરિટ રેન્ક જરૂરી છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू बी.टेक आईटी में प्रवेश के लिए 12वीं साइंस पीसीएम में न्यूनतम 60% अंक, वैध गुजकेट स्कोर और एसीपीसी मेरिट रैंक आवश्यक है।"
                                else:
                                    clean_text = "DDU B.Tech IT admission requires 12th Science PCM with minimum 60% aggregate, a valid GUJCET score, and ACPC merit counseling rank."
                            elif any(w in lower_u for w in ["fee", "fees", "cost", "ફી", "ખર્ચ", "फी", "फीस"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ બીટેક આઈટીમાં પ્રથમ વર્ષની ફી ૧,૬૬,૯૫૦ રૂપિયા અને બીજાથી ચોથા વર્ષ માટે વાર્ષિક ૧,૫૨,૦૦૦ રૂપિયા છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू बी.टेक आईटी में प्रथम वर्ष की फीस 1,66,950 रुपये और दूसरे से चौथे वर्ष के लिए 1,52,000 रुपये प्रति वर्ष है।"
                                else:
                                    clean_text = "DDU B.Tech IT first year fee is 1,66,950 rupees and second to fourth year fee is 1,52,000 rupees per year."
                            elif any(w in lower_u for w in ["gate", "ગેટ", "गेट"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ આઈટી અભ્યાસક્રમમાં ગેટના તમામ વિષયો જેમ કે મેથેમેટિક્સ, ડિજિટલ લોજિક, સીઓએ, ડેટા સ્ટ્રક્ચર્સ, એલ્ગોરિધમ્સ, ઓએસ, ડીબીએમએસ અને કમ્પ્યુટર નેટવર્ક્સ સંપૂર્ણપણે આવરી લેવામાં આવ્યા છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू आईटी पाठ्यक्रम में गेट के सभी 12 विषय जैसे इंजीनियरिंग मैथमेटिक्स, डिजिटल लॉजिक, COA, डेटा स्ट्रक्चर्स, एल्गोरिदम, OS, DBMS और कंप्यूटर नेटवर्क शामिल हैं।"
                                else:
                                    clean_text = "DDU IT curriculum covers all 12 GATE subjects including Engineering Mathematics, Digital Logic, COA, Programming & Data Structures, Algorithms, OS, DBMS, and Networks."
                            elif any(w in lower_u for w in ["સેમેસ્ટર", "સેમ", "સિલેબસ", "અભ્યાસક્રમ", "વિષય", "semester", "syllabus", "subject"]) or sem_detected:
                                sem_target = sem_detected if sem_detected else 1
                                if turn_lang == "gu":
                                    clean_text = f"ડીડીયુ આઈટી સેમેસ્ટર {sem_target} નો સંપૂર્ણ અભ્યાસક્રમ ઉપલબ્ધ છે. શું તમારે ચોક્કસ વિષય વિશે જાણવું છે?"
                                elif turn_lang == "hi":
                                    clean_text = f"डीडीयू आईटी सेमेस्टर {sem_target} का संपूर्ण पाठ्यक्रम उपलब्ध है। क्या आप किसी विशेष विषय के बारे में जानना चाहते हैं?"
                                else:
                                    clean_text = f"DDU IT Semester {sem_target} full curriculum is available. Would you like details on specific subjects?"
                            elif any(w in lower_u for w in ["placement", "પ્લેસમેન્ટ", "प्लेसमेंट", "salary", "package"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ આઈટીમાં સરેરાશ પેકેજ ૬ લાખ (6 LPA), સૌથી વધુ પેકેજ ૨૪ લાખ ઓફ-કેમ્પસ (24 LPA) છે અને ૨૦૦ વિદ્યાર્થીઓ પ્લેસ થયા છે. મુખ્ય રિક્રૂટર્સ TCS, Infosys, Crest Data Systems અને TatvaSoft છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू आईटी में औसत पैकेज 6 लाख (6 LPA), उच्चतम पैकेज 24 लाख ऑफ-कैंपस (24 LPA) है और 200 छात्र प्लेस हुए हैं। प्रमुख रिक्रूटर्स TCS, Infosys, Crest Data Systems और TatvaSoft हैं।"
                                else:
                                    clean_text = "DDU IT placement details: average package is 6 LPA, highest package is 24 LPA off-campus, with 200 placed students. Top recruiters include TCS, Infosys, Crest Data Systems, and TatvaSoft."
                            else:
                                if turn_lang == "gu":
                                    clean_text = "હું તમને એડમિશન પાત્રતા, ફી માળખું, સેમેસ્ટર વિષયો, ગેટ વિષયો અથવા પ્લેસમેન્ટ વિશે શું માહિતી આપું?"
                                elif turn_lang == "hi":
                                    clean_text = "मैं आपको प्रवेश पात्रता, फीस संरचना, सेमेस्टर विषय, गेट विषय या प्लेसमेंट के बारे में क्या जानकारी दूँ?"
                                else:
                                    clean_text = "How may I assist you with admission eligibility, fees structure, semester subjects, GATE subjects, or placements?"
                            yield clean_text

                        clean_history_text = clean_text
                        if hold_phrase_yielded and not clean_history_text.startswith(hold_phrase):
                            clean_history_text = f"{hold_phrase} {clean_history_text}"
                        self.history.append({"role": "assistant", "content": clean_history_text})
                        return

            fallback = "I have fetched the information. How else may I assist you with university admissions?"
            clean_fallback = fallback
            if hold_phrase_yielded and not clean_fallback.startswith(hold_phrase):
                clean_fallback = f"{hold_phrase} {clean_fallback}"
            self.history.append({"role": "assistant", "content": clean_fallback})
            yield fallback

        except Exception as err:
            print(f"[llm] Error communicating with {config.LLM_PROVIDER}: {err}")
            fallback_msg = get_error_message("system_error", turn_lang) or "I am sorry, I am having trouble accessing the university system at this moment. Please try again shortly."
            clean_fallback = fallback_msg
            if hold_phrase_yielded and not clean_fallback.startswith(hold_phrase):
                clean_fallback = f"{hold_phrase} {clean_fallback}"
            self.history.append({"role": "assistant", "content": clean_fallback})
            yield fallback_msg

    async def areply_stream(self, user_text: str, stt_lang: str | None = None) -> AsyncGenerator[str, None]:
        """
        Asynchronously yields response tokens incrementally in real time.
        Zero event-loop blocking for FastAPI and WebSocket servers.
        """
        messages = self._prepare_messages(user_text, stt_lang=stt_lang)
        turn_lang = self.current_lang

        # Direct zero-latency response for abusive language ("Kripya shanti se baat kare")
        if is_abusive_intent(user_text):
            abusive_map = ABUSIVE_CALM_PROMPTS
            exact_calm = abusive_map.get(turn_lang, abusive_map["gu"])
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": exact_calm})
            yield exact_calm
            return

        # Direct zero-latency response when caller explicitly asks "who are you"
        if is_who_are_you_intent(user_text):
            identity_map = WHO_ARE_YOU_PROMPTS
            exact_identity = identity_map.get(turn_lang, identity_map["gu"])
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": exact_identity})
            yield exact_identity
            return

        # Direct response for phone greetings during ongoing call (e.g. 'hello', 'hi', 'kem cho')
        # Since welcome message already introduced Priya at call start, do NOT repeat the self-intro!
        if getattr(self, "_last_turn_was_greeting", False):
            greeting_map = {
                "gu": "નમસ્તે! કહો, હું તમારી શું મદદ કરી શકું?",
                "hi": "नमस्ते! बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
                "en": "Hello! How can I assist you today?",
            }
            exact_greeting = greeting_map.get(turn_lang, greeting_map["gu"])
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": exact_greeting})
            yield exact_greeting
            return

        # Direct zero-latency response for call hangup intent (e.g. "phone cur kardo", "phone cut kardo", "bye")
        if is_hangup_intent(user_text):
            farewell_map = {
                "gu": "તમારો દિવસ શુભ રહે. આભાર!",
                "hi": "आपका दिन शुभ हो। धन्यवाद!",
                "en": "Have a great day. Thank you!",
            }
            exact_farewell = farewell_map.get(turn_lang, farewell_map["gu"])
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": exact_farewell})
            yield exact_farewell
            return

        # Direct response when caller agrees / acknowledges ("Okay", "Fine", "Understood", "ઠીક છે", "બરાબર", "ठीक है")
        if is_agreement_intent(user_text):
            help_map = FOLLOW_UP_HELP_PROMPTS
            exact_help = help_map.get(turn_lang, help_map["gu"])
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": exact_help})
            yield exact_help
            return

        hold_phrase = HOLD_PHRASES.get(turn_lang, HOLD_PHRASES["gu"])
        hold_phrase_yielded = False

        max_tool_iterations = 3
        current_iteration = 0

        try:
            async with self._borrow_client() as client:
                while current_iteration < max_tool_iterations:
                    current_iteration += 1
                    url, headers, payload = self._get_provider_request(messages)

                    stream_buffer = ""
                    recent_window = ""
                    is_tool_call = False
                    is_streaming_speech = False
                    full_reply = ""
                    in_think = False
                    checked_hold_dedup = not hold_phrase_yielded

                    async with client.stream("POST", url, headers=headers, json=payload) as resp:
                        if resp.status_code == 429 and getattr(self, "_active_provider", config.LLM_PROVIDER) == "openrouter" and config.SARVAM_API_KEY:
                            print("⚠️ [llm] OpenRouter quota/rate limit (429) hit. Gracefully switching to Sarvam LLM (sarvam-105b-conversations)...")
                            self._active_provider = "sarvam"
                            current_iteration -= 1
                            continue
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            token = self._parse_stream_line(line)
                            if not token:
                                continue

                            # Filter thinking reasoning tokens (<think>...</think> or <thought>...</thought>)
                            if in_think:
                                if "</think>" in token:
                                    in_think = False
                                    token = token.split("</think>", 1)[1]
                                    if not token:
                                        continue
                                elif "</thought>" in token:
                                    in_think = False
                                    token = token.split("</thought>", 1)[1]
                                    if not token:
                                        continue
                                else:
                                    continue

                            if "<think>" in token or "<thought>" in token:
                                in_think = True
                                tag = "<think>" if "<think>" in token else "<thought>"
                                token = token.split(tag, 1)[0]
                                if not token:
                                    continue

                            # Detect whether model is issuing a tool call with sliding lookback window
                            recent_window = (recent_window + token)[-30:]

                            if not is_streaming_speech and not is_tool_call:
                                stream_buffer += token
                                stripped = stream_buffer.lstrip()
                                if "TOOL_CALL:" in stream_buffer or "TOOL_CALL:" in recent_window:
                                    is_tool_call = True
                                elif "TOOL_CALL:".startswith(stripped):
                                    continue
                                else:
                                    if hold_phrase_yielded and not checked_hold_dedup:
                                        if not stripped:
                                            continue
                                        if HOLD_PHRASE_REGEX.match(stripped):
                                            stream_buffer = HOLD_PHRASE_REGEX.sub("", stripped, count=1)
                                            checked_hold_dedup = True
                                            stripped = stream_buffer.lstrip()
                                            if not stripped:
                                                stream_buffer = ""
                                                continue
                                        elif len(stripped) >= 35 or any(p in stripped for p in [".", "!", "?", "।"]):
                                            checked_hold_dedup = True
                                        else:
                                            continue

                                    has_punct = any(p in stream_buffer for p in [".", "!", "?", "।", "\n"])
                                    if (has_punct or len(stripped) >= 12) and "TOOL_CALL:" not in stream_buffer:
                                        if is_prompt_leak(stream_buffer):
                                            print(f"🛑 [llm] Suppressed prompt leak buffer in async stream: {stream_buffer}")
                                            stream_buffer = ""
                                            continue
                                        if stream_buffer:
                                            is_streaming_speech = True
                                            yield stream_buffer
                                            full_reply += stream_buffer
                                            stream_buffer = ""
                                continue

                            if is_tool_call:
                                stream_buffer += token
                            else:
                                if "TOOL_CALL:" in recent_window or "TOOL_CALL:" in token:
                                    is_tool_call = True
                                    is_streaming_speech = False
                                    if "TOOL_CALL:" in full_reply:
                                        idx = full_reply.index("TOOL_CALL:")
                                        stream_buffer = full_reply[idx:]
                                        full_reply = full_reply[:idx]
                                    elif "TOOL_CALL:" in token:
                                        idx = token.index("TOOL_CALL:")
                                        stream_buffer = token[idx:]
                                    else:
                                        stream_buffer = "TOOL_CALL:"
                                    continue
                                full_reply += token
                                if not is_prompt_leak(full_reply):
                                    yield token

                    # If remaining buffer wasn't flushed for short responses
                    if stream_buffer and not is_tool_call:
                        if hold_phrase_yielded and not checked_hold_dedup:
                            checked_hold_dedup = True
                            stripped = stream_buffer.lstrip()
                            if HOLD_PHRASE_REGEX.match(stripped):
                                stream_buffer = HOLD_PHRASE_REGEX.sub("", stripped, count=1)
                        if stream_buffer and not is_prompt_leak(stream_buffer):
                            full_reply += stream_buffer
                            yield stream_buffer
                        stream_buffer = ""

                    if is_tool_call:
                        if not hold_phrase_yielded:
                            yield f"{hold_phrase} "
                            hold_phrase_yielded = True
                        tool_info = self._parse_tool_call(stream_buffer)
                        if tool_info:
                            tool_name, kwargs = tool_info
                            # Run tool in worker thread if blocking
                            tool_result = await asyncio.to_thread(self._execute_tool, tool_name, kwargs)
                            tool_synth = f"Please synthesize a short, polite spoken answer for the caller in 1-2 sentences in {turn_lang} language."
                            messages.append({"role": "assistant", "content": stream_buffer})
                            messages.append({
                                "role": "user",
                                "content": f"TOOL_RESULT ({tool_name}): {tool_result}\n{tool_synth}",
                            })
                            continue
                        else:
                            clean_text = clean_speech_text(stream_buffer)
                            if is_prompt_leak(clean_text):
                                clean_text = "I am here to help you with DDU IT queries. How may I assist you?"
                            clean_history_text = clean_text
                            if hold_phrase_yielded and not clean_history_text.startswith(hold_phrase):
                                clean_history_text = f"{hold_phrase} {clean_history_text}"
                            self.history.append({"role": "assistant", "content": clean_history_text})
                            yield clean_text
                            return
                    else:
                        clean_text = clean_speech_text(full_reply)
                        if is_prompt_leak(clean_text):
                            print(f"🛑 [llm] Suppressed prompt leak from async final text: {clean_text}")
                            clean_text = ""

                        if not clean_text.strip():
                            turn_lang, _, _ = self.detect_turn_language(user_text, fallback_lang=self.current_lang)
                            lower_u = user_text.lower()
                            sem_detected = extract_semester(user_text)
                            if any(w in lower_u for w in ["admission", "admissions", "eligible", "eligibility", "criteria", "એડમિશન", "પ્રવેશ", "પાત્રતા", "લાયકાત", "प्रवेश"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ બીટેક આઈટીમાં પ્રવેશ માટે ૧૨ સાયન્સ પીસીએમમાં ઓછામાં ઓછા ૬૦% ગુણ, ગુજકેટ પરીક્ષા અને એસીપીસી મેરિટ રેન્ક જરૂરી છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू बी.टेक आईटी में प्रवेश के लिए 12वीं साइंस पीसीएम में न्यूनतम 60% अंक, वैध गुजकेट स्कोर और एसीपीसी मेरिट रैंक आवश्यक है।"
                                else:
                                    clean_text = "DDU B.Tech IT admission requires 12th Science PCM with minimum 60% aggregate, a valid GUJCET score, and ACPC merit counseling rank."
                            elif any(w in lower_u for w in ["fee", "fees", "cost", "ફી", "ખર્ચ", "फी", "फीस"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ બીટેક આઈટીમાં પ્રથમ વર્ષની ફી ૧,૬૬,૯૫૦ રૂપિયા અને બીજાથી ચોથા વર્ષ માટે વાર્ષિક ૧,૫૨,૦૦૦ રૂપિયા છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू बी.टेक आईटी में प्रथम वर्ष की फीस 1,66,950 रुपये और दूसरे से चौथे वर्ष के लिए 1,52,000 रुपये प्रति वर्ष है।"
                                else:
                                    clean_text = "DDU B.Tech IT first year fee is 1,66,950 rupees and second to fourth year fee is 1,52,000 rupees per year."
                            elif any(w in lower_u for w in ["gate", "ગેટ", "गेट"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ આઈટી અભ્યાસક્રમમાં ગેટના તમામ વિષયો જેમ કે મેથેમેટિક્સ, ડિજિટલ લોજિક, સીઓએ, ડેટા સ્ટ્રક્ચર્સ, એલ્ગોરિધમ્સ, ઓએસ, ડીબીએમએસ અને કમ્પ્યુટર નેટવર્ક્સ સંપૂર્ણપણે આવરી લેવામાં આવ્યા છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू आईटी पाठ्यक्रम में गेट के सभी 12 विषय जैसे इंजीनियरिंग मैथमेटिक्स, डिजिटल लॉजिक, COA, डेटा स्ट्रक्चर्स, एल्गोरिदम, OS, DBMS और कंप्यूटर नेटवर्क शामिल हैं।"
                                else:
                                    clean_text = "DDU IT curriculum covers all 12 GATE subjects including Engineering Mathematics, Digital Logic, COA, Programming & Data Structures, Algorithms, OS, DBMS, and Networks."
                            elif any(w in lower_u for w in ["સેમેસ્ટર", "સેમ", "સિલેબસ", "અભ્યાસક્રમ", "વિષય", "semester", "syllabus", "subject"]) or sem_detected:
                                sem_target = sem_detected if sem_detected else 1
                                if turn_lang == "gu":
                                    clean_text = f"ડીડીયુ આઈટી સેમેસ્ટર {sem_target} નો સંપૂર્ણ અભ્યાસક્રમ ઉપલબ્ધ છે. શું તમારે ચોક્કસ વિષય વિશે જાણવું છે?"
                                elif turn_lang == "hi":
                                    clean_text = f"डीडीयू आईटी सेमेस्टर {sem_target} का संपूर्ण पाठ्यक्रम उपलब्ध है। क्या आप किसी विशेष विषय के बारे में जानना चाहते हैं?"
                                else:
                                    clean_text = f"DDU IT Semester {sem_target} full curriculum is available. Would you like details on specific subjects?"
                            elif any(w in lower_u for w in ["placement", "પ્લેસમેન્ટ", "प्लेसमेंट", "salary", "package"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ આઈટીમાં સરેરાશ પેકેજ ૬ લાખ (6 LPA), સૌથી વધુ પેકેજ ૨૪ લાખ ઓફ-કેમ્પસ (24 LPA) છે અને ૨૦૦ વિદ્યાર્થીઓ પ્લેસ થયા છે. મુખ્ય રિક્રૂટર્સ TCS, Infosys, Crest Data Systems અને TatvaSoft છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू आईटी में औसत पैकेज 6 लाख (6 LPA), उच्चतम पैकेज 24 लाख ऑफ-कैंपस (24 LPA) है और 200 छात्र प्लेस हुए हैं। प्रमुख रिक्रूटर्स TCS, Infosys, Crest Data Systems और TatvaSoft हैं।"
                                else:
                                    clean_text = "DDU IT placement details: average package is 6 LPA, highest package is 24 LPA off-campus, with 200 placed students. Top recruiters include TCS, Infosys, Crest Data Systems, and TatvaSoft."
                            else:
                                if turn_lang == "gu":
                                    clean_text = "હું તમને એડમિશન પાત્રતા, ફી માળખું, સેમેસ્ટર વિષયો, ગેટ વિષયો અથવા પ્લેસમેન્ટ વિશે શું માહિતી આપું?"
                                elif turn_lang == "hi":
                                    clean_text = "मैं आपको प्रवेश पात्रता, फीस संरचना, सेमेस्टर विषय, गेट विषय या प्लेसमेंट के बारे में क्या जानकारी दूँ?"
                                else:
                                    clean_text = "How may I assist you with admission eligibility, fees structure, semester subjects, GATE subjects, or placements?"
                            yield clean_text

                        clean_history_text = clean_text
                        if hold_phrase_yielded and not clean_history_text.startswith(hold_phrase):
                            clean_history_text = f"{hold_phrase} {clean_history_text}"
                        self.history.append({"role": "assistant", "content": clean_history_text})
                        return

            fallback = "I have fetched the information. How else may I assist you with university admissions?"
            clean_fallback = fallback
            if hold_phrase_yielded and not clean_fallback.startswith(hold_phrase):
                clean_fallback = f"{hold_phrase} {clean_fallback}"
            self.history.append({"role": "assistant", "content": clean_fallback})
            yield fallback

        except (asyncio.CancelledError, GeneratorExit):
            # Rollback dangling user turn on cancellation
            if self.history and self.history[-1].get("role") == "user":
                self.history.pop()
            raise
        except Exception as err:
            print(f"[llm] Async communication error with {config.LLM_PROVIDER}: {err}")
            fallback_msg = get_error_message("system_error", turn_lang) or "I am sorry, I am having trouble accessing the university system at this moment. Please try again shortly."
            clean_fallback = fallback_msg
            if hold_phrase_yielded and not clean_fallback.startswith(hold_phrase):
                clean_fallback = f"{hold_phrase} {clean_fallback}"
            self.history.append({"role": "assistant", "content": clean_fallback})
            yield fallback_msg
