import os
import time
import json
import logging
import datetime
import re
import requests
from bs4 import BeautifulSoup
from smolagents import Tool, CodeAgent, OpenAIServerModel

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# =========================================================
# 1. Kredensial dari GitHub Secrets — JANGAN hardcode
# =========================================================
TELEGRAM_TOKEN     = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
GOOGLE_API_KEY     = os.environ["GOOGLE_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
NVIDIA_API_KEY     = os.environ.get("NVIDIA_API_KEY", "")
ZAI_API_KEY        = os.environ.get("ZAI_API_KEY", "")
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")

TELEGRAM_MAX_CHARS = 3800

REPORT_MARKERS = [
    "## 📊 LAPORAN MENDALAM",
    "Geopolitik",
    "Olahraga",
    "Teknologi",
    "Indonesia",
    "Trending Indonesia",
]
MIN_REPORT_LENGTH = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("market-bot")

HISTORY_FILE = "history.json"
MAX_HISTORY = 5

DOMAIN_INTERNASIONAL = [
    "reuters.com", "bloomberg.com", "cnbc.com", "bbc.com", "bbc.co.uk",
    "aljazeera.com", "ft.com", "espn.com", "skysports.com", "uefa.com",
    "fifa.com", "techcrunch.com", "theverge.com", "wired.com",
    "technologyreview.com", "arstechnica.com", "theguardian.com",
]
DOMAIN_INDONESIA = [
    "cnnindonesia.com", "cnbcindonesia.com", "bisnis.com", "kompas.com",
    "detik.com", "kontan.co.id", "antara.com", "tempo.co", "liputan6.com",
    "bola.com", "tribunnews.com", "jawapos.com", "suara.com", "okezone.com",
]

# Global untuk melacak URL yang berhasil di-fetch
FETCHED_URLS = set()

# Opsional: menyimpan teks hasil fetch untuk keperluan log/debug
FETCHED_CONTENT = {}

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning(f"Gagal baca history: {e}")
        return []
    if not isinstance(data, list):
        return []
    valid = []
    for entry in data:
        if isinstance(entry, dict) and "tanggal" in entry and "ringkasan" in entry:
            valid.append(entry)
    return valid

def simpan_history(laporan):
    history = load_history()
    history.append({
        "tanggal": datetime.date.today().isoformat(),
        "ringkasan": laporan[:1500],
    })
    history = history[-MAX_HISTORY:]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        log.info(f"History disimpan ({len(history)} entri).")
    except Exception as e:
        log.warning(f"Gagal simpan history: {e}")

def ringkasan_history():
    history = load_history()
    if not history:
        return "(belum ada histori laporan sebelumnya)"
    return "\n".join(
        f"- [{h['tanggal']}] {h['ringkasan'][:300]}..." for h in history
    )

def laporan_valid(teks, fetched_urls):
    """
    Validasi ketat:
    - minimal panjang
    - wajib ada judul laporan
    - minimal 4 topik
    - minimal 4 URL dari domain whitelist DAN terdaftar di fetched_urls
    - tidak boleh ada pola output rusak
    - minimal 2 tanggal publikasi (case-insensitive)
    """
    if not teks or len(teks.strip()) < MIN_REPORT_LENGTH:
        return False, f"Terlalu pendek ({len(teks.strip()) if teks else 0} karakter, minimal {MIN_REPORT_LENGTH})"

    if "## 📊 LAPORAN MENDALAM" not in teks:
        return False, "Tidak ada judul '## 📊 LAPORAN MENDALAM'"

    topik = ["Geopolitik", "Olahraga", "Teknologi", "Indonesia", "Trending Indonesia"]
    ditemukan = sum(1 for t in topik if t.lower() in teks.lower())
    if ditemukan < 4:
        return False, f"Hanya {ditemukan} topik ditemukan, minimal 4"

    # Ekstrak URL dari laporan
    urls = re.findall(r'https?://[^\s\)]+', teks)
    domain_whitelist = DOMAIN_INTERNASIONAL + DOMAIN_INDONESIA
    valid_urls = [u for u in urls if any(domain in u for domain in domain_whitelist)]
    if len(valid_urls) < 4:
        return False, f"Hanya {len(valid_urls)} URL whitelist ditemukan, minimal 4"

    # Pastikan minimal 4 URL yang valid juga ada di fetched_urls
    matching_fetched = [u for u in valid_urls if any(u.startswith(fu) or fu in u for fu in fetched_urls)]
    if len(matching_fetched) < 4:
        return False, f"Hanya {len(matching_fetched)} URL yang benar-benar di-fetch, minimal 4"

    bad_patterns = [
        "</code", "User Safety", "Response Safety",
        '"tool": "web_search"', '"request_id"', '```',
        "Error fetching the webpage", "Code parsing failed",
    ]
    for pat in bad_patterns:
        if pat in teks:
            return False, f"Terdeteksi output rusak: {pat}"

    tanggal_patterns = [
        r"\b\d{1,2}\s+(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]
    jumlah_tanggal = 0
    for pat in tanggal_patterns:
        jumlah_tanggal += len(re.findall(pat, teks, re.IGNORECASE))
    if jumlah_tanggal < 2:
        return False, f"Hanya {jumlah_tanggal} tanggal ditemukan, minimal 2"

    return True, "OK"

def _kirim_satu(pesan, parse_mode=None):
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": pesan}
    if parse_mode:
        data["parse_mode"] = parse_mode
    try:
        r = requests.post(url, data=data, timeout=30)
        return r.status_code == 200
    except Exception as e:
        log.error(f"Koneksi Telegram error: {e}")
        return False

def _pecah(pesan):
    chunks = []
    while len(pesan) > TELEGRAM_MAX_CHARS:
        potong = pesan.rfind("\n", 0, TELEGRAM_MAX_CHARS)
        if potong < TELEGRAM_MAX_CHARS * 0.5:
            potong = TELEGRAM_MAX_CHARS
        chunks.append(pesan[:potong])
        pesan = pesan[potong:].lstrip("\n ")
    if pesan:
        chunks.append(pesan)
    return chunks

def kirim_ke_telegram(pesan):
    log.info("Mengirim laporan ke Telegram...")
    chunks = _pecah(pesan)
    if len(chunks) > 1:
        log.info(f"Pesan dipecah jadi {len(chunks)} bagian.")
    for i, chunk in enumerate(chunks, 1):
        prefix = f"📄 Bagian {i}/{len(chunks)}\n\n" if len(chunks) > 1 else ""
        teks   = prefix + chunk
        if _kirim_satu(teks, parse_mode="Markdown"):
            log.info(f"✅ Bagian {i}/{len(chunks)} terkirim (Markdown)!")
        else:
            if _kirim_satu(teks):
                log.info(f"✅ Bagian {i}/{len(chunks)} terkirim (plain text)!")
            else:
                log.error(f"❌ Bagian {i}/{len(chunks)} GAGAL terkirim.")
        time.sleep(1)

class RecentNewsSearchTool(Tool):
    name        = "web_search"
    description = (
        "Cari berita/informasi TERBARU dari 24-48 jam terakhir. "
        "Untuk topik global gunakan query Bahasa Inggris. "
        "Untuk topik Indonesia gunakan Bahasa Indonesia. "
        "Kembalikan STRING berisi judul, ringkasan, URL, dan tanggal publikasi jika tersedia."
    )
    inputs      = {"query": {"type": "string", "description": "Kata kunci pencarian"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        results = None
        for attempt in range(3):
            try:
                if attempt == 0:
                    results = DDGS(backend="html").text(query, timelimit="d", max_results=4)
                elif attempt == 1:
                    results = DDGS().text(query, timelimit="d", max_results=4)
                else:
                    results = DDGS(backend="html").text(query, timelimit="w", max_results=4)
                if results:
                    break
            except Exception as e:
                log.warning(f"Percobaan {attempt+1} gagal: {e}")
                continue

        if not results:
            return "Tidak ada hasil ditemukan, coba kata kunci lain."

        out = ""
        for r in results:
            title = r.get('title', '')
            body  = r.get('body', '')
            url   = r.get('href', '')
            date_str = r.get('date') or r.get('published') or r.get('timestamp') or ''
            if not date_str:
                m = re.search(r"\b(\d{1,2}\s+\w+\s+\d{4})\b", body)
                if m:
                    date_str = m.group(1)
            if not date_str:
                date_str = "tanggal tidak tersedia"
            out += f"- {title}\n  Tanggal: {date_str}\n  {body}\n  URL: {url}\n\n"
        return out

class FetchWebpageTool(Tool):
    name        = "fetch_webpage"
    description = (
        "Ambil isi halaman web dari URL yang diberikan dan kembalikan teks artikel yang sudah dibersihkan. "
        "Gunakan setelah mendapatkan URL dari web_search."
    )
    inputs      = {"url": {"type": "string", "description": "URL halaman web yang akan diambil"}}
    output_type = "string"

    def forward(self, url: str) -> str:
        global FETCHED_URLS, FETCHED_CONTENT
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # Buang elemen yang tidak relevan
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button"]):
                tag.decompose()

            # Coba ambil konten utama
            main = soup.find("article") or soup.find("main") or soup.body
            paragraphs = main.find_all("p") if main else []
            text = "\n".join(p.get_text(strip=True) for p in paragraphs)

            # Jika tidak ada paragraf, ambil semua teks
            if not text:
                text = soup.get_text(separator="\n", strip=True)

            # Potong hingga 2500 karakter
            text = text[:2500]

            # Tandai URL berhasil di-fetch dan simpan konten
            FETCHED_URLS.add(url)
            FETCHED_CONTENT[url] = text
            return text
        except Exception as e:
            return f"Error fetching the webpage: {e}"

class FormatError(Exception):
    """Error yang menandakan model menghasilkan format yang salah."""
    pass

class FallbackModel:
    """
    Multi‑provider dengan beberapa model per provider.
    Model diurutkan dari yang paling stabil ke yang lebih murah/cadangan.
    Jika model mati (404/410/retired) atau sering gagal format, otomatis ditandai.
    """
    def __init__(self, providers):
        self.providers = []
        for p in providers:
            entry = {
                "name":       p["name"],
                "api_base":   p["api_base"],
                "api_key":    p["api_key"],
                "models":     p["models"],
                "model_objs": [],
                "dead":       [False] * len(p["models"]),
                "current_idx": 0,
            }
            for model_id in p["models"]:
                try:
                    m = OpenAIServerModel(
                        model_id=model_id,
                        api_base=p["api_base"],
                        api_key=p["api_key"],
                        client_kwargs={"max_retries": 0, "timeout": 45.0},
                        retry=False,
                    )
                    entry["model_objs"].append(m)
                except Exception as e:
                    log.warning(f"Gagal siapkan model {model_id} di {p['name']}: {e}")
                    entry["model_objs"].append(None)
                    entry["dead"][len(entry["model_objs"])-1] = True
            self.providers.append(entry)
            log.info(f"Provider siap: {p['name']} dengan {len(p['models'])} model")

        if not self.providers:
            raise RuntimeError("Tidak ada provider AI yang bisa disiapkan!")

    def _is_model_dead_error(self, e):
        msg = str(e).lower()
        dead_markers = [
            "not found", "404", "410", "retired", "deprecated",
            "no longer available", "model_not_found", "unknown model",
            "does not exist", "not available"
        ]
        return any(marker in msg for marker in dead_markers)

    def _is_rate_limit_error(self, e):
        msg = str(e).lower()
        return "rate limit" in msg or "429" in msg or "too many requests" in msg

    def _is_format_error(self, output):
        if not isinstance(output, str):
            return False
        bad_patterns = ["</code", "```", "Code parsing failed"]
        return any(pat in output for pat in bad_patterns)

    def _try_all(self, method_name, *args, **kwargs):
        last_err = None
        for entry in self.providers:
            if all(entry["dead"]):
                log.info(f"Provider {entry['name']} semua model mati, skip.")
                continue

            while entry["current_idx"] < len(entry["models"]):
                if entry["dead"][entry["current_idx"]]:
                    entry["current_idx"] += 1
                    continue

                model_obj = entry["model_objs"][entry["current_idx"]]
                if model_obj is None:
                    entry["dead"][entry["current_idx"]] = True
                    entry["current_idx"] += 1
                    continue

                try:
                    log.info(f"Mencoba provider {entry['name']} dengan model {entry['models'][entry['current_idx']]}...")
                    result = getattr(model_obj, method_name)(*args, **kwargs)
                    # Periksa format output
                    if self._is_format_error(result):
                        log.warning(f"Model {entry['models'][entry['current_idx']]} menghasilkan format salah. Tandai mati.")
                        entry["dead"][entry["current_idx"]] = True
                        entry["current_idx"] += 1
                        last_err = FormatError("Format output tidak valid")
                        continue
                    log.info(f"✅ Berhasil pakai: {entry['name']} / {entry['models'][entry['current_idx']]}")
                    return result
                except Exception as e:
                    log.warning(f"⚠️ {entry['name']} / {entry['models'][entry['current_idx']]} gagal: {e}")
                    if self._is_model_dead_error(e):
                        log.warning(f"Model {entry['models'][entry['current_idx']]} ditandai mati (retired/404/410).")
                        entry["dead"][entry["current_idx"]] = True
                        entry["current_idx"] += 1
                        continue
                    elif self._is_rate_limit_error(e):
                        log.warning("Rate limit tercapai, pindah ke provider berikutnya.")
                        last_err = e
                        break
                    else:
                        last_err = e
                        entry["current_idx"] += 1
                        continue

            if entry["current_idx"] >= len(entry["models"]):
                log.warning(f"Semua model di {entry['name']} sudah dicoba dan gagal.")

        raise Exception(f"Semua provider gagal! Error terakhir: {last_err}")

    def generate(self, messages, stop_sequences=None, **kwargs):
        return self._try_all("generate", messages, stop_sequences=stop_sequences, **kwargs)

    def __call__(self, messages, stop_sequences=None, **kwargs):
        return self._try_all("__call__", messages, stop_sequences=stop_sequences, **kwargs)

    def __getattr__(self, attr):
        for entry in self.providers:
            if entry["model_objs"]:
                return getattr(entry["model_objs"][0], attr)
        raise AttributeError(attr)


def buat_agent():
    log.info("Menyiapkan AI dengan fallback chain multi‑model (stabil → murah, free tier saja)...")

    daftar_provider = [
        {
            "name": "Gemini",
            "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": GOOGLE_API_KEY,
            # Urutan dari paling stabil (free tier) ke paling murah
            "models": [
                "gemini-3.6-flash",          # paling stabil di free tier
                "gemini-3.5-flash",          # cadangan
                "gemini-3.1-flash-lite",     # lebih hemat
                "gemini-3.5-flash-lite",     # paling hemat
            ],
        },
    ]

    if NVIDIA_API_KEY:
        daftar_provider.append({
            "name": "NVIDIA NIM",
            "api_base": "https://integrate.api.nvidia.com/v1",
            "api_key": NVIDIA_API_KEY,
            "models": [
                "nvidia/nemotron-3-super-120b-a12b",
                "meta/llama-4-maverick-17b-128e-instruct",
            ],
        })
    else:
        log.info("NVIDIA_API_KEY belum diset — provider NVIDIA di-skip.")

    if ZAI_API_KEY:
        daftar_provider.append({
            "name": "Z.AI",
            "api_base": "https://api.z.ai/api/paas/v4/",
            "api_key": ZAI_API_KEY,
            "models": ["glm-4.5-flash"],
        })
    else:
        log.info("ZAI_API_KEY belum diset — provider Z.AI di-skip.")

    daftar_provider.append({
        "name": "OpenRouter (auto-router gratis)",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
        "models": ["openrouter/free"],
    })

    if GROQ_API_KEY:
        daftar_provider.append({
            "name": "Groq (opsional)",
            "api_base": "https://api.groq.com/openai/v1",
            "api_key": GROQ_API_KEY,
            "models": [
                "llama-3.3-70b-versatile",
            ],
        })

    model = FallbackModel(daftar_provider)

    return CodeAgent(
        tools=[
            RecentNewsSearchTool(),
            FetchWebpageTool(),
        ],
        model=model,
        additional_authorized_imports=["datetime", "os", "re", "requests", "bs4"],
        max_steps=10,   # Naikkan dari 7 menjadi 10 agar cukup untuk riset
    )


def hitung_visit_sukses(agent):
    """Hitung berapa kali tool fetch_webpage dipanggil dan berhasil."""
    jumlah = 0
    for step in agent.memory.steps:
        code = getattr(step, "code_action", None)
        obs  = getattr(step, "observations", None) or ""
        if code and "fetch_webpage(" in code and "Error fetching the webpage:" not in obs:
            jumlah += 1
    return jumlah


def jalankan_analisa_harian():
    global FETCHED_URLS, FETCHED_CONTENT
    FETCHED_URLS = set()
    FETCHED_CONTENT = {}

    log.info("=" * 55)
    log.info("MEMULAI ANALISA PASAR & BERITA GLOBAL OTOMATIS...")
    log.info("=" * 55)

    tanggal      = datetime.date.today().strftime("%d %B %Y")
    histori      = ringkasan_history()

    tugas = f"""
Hari ini tanggal {tanggal}. Gunakan HANYA berita dari 1-2 hari terakhir (maksimal 48 jam ke belakang).

Laporan sebelumnya — JANGAN ulang topik/angka yang persis sama, cari yang baru:
{histori}

Kamu adalah analis intelijen senior, jurnalis ekonomi, pengamat olahraga, dan pakar teknologi.
Buat laporan mendalam untuk 5 topik berikut:

1. **Geopolitik & Ekonomi Global**
   Cari: berita geopolitik internasional terkini dan dampaknya ke pasar kripto/saham.
   Sumber target: Reuters, Bloomberg, CNBC, BBC, Al Jazeera, Financial Times.
   Query pencarian: gunakan Bahasa Inggris.

2. **Olahraga Global**
   Cari: hasil pertandingan atau berita transfer pemain dari 24-48 jam terakhir.
   Sumber target: ESPN, BBC Sport, Sky Sports, UEFA.com, FIFA.com.
   Query pencarian: gunakan Bahasa Inggris.

3. **Teknologi & AI Terbaru**
   Cari: satu berita teknologi, AI, atau sains yang konkret dari 1-2 hari terakhir.
   Sumber target: TechCrunch, The Verge, Wired, MIT Technology Review, Ars Technica.
   Query pencarian: gunakan Bahasa Inggris.

4. **Indonesia Update**
   Cari DUA hal terpisah:
   a) Ekonomi: kondisi IHSG hari ini (level dan persentase perubahan), kurs Rupiah terhadap USD,
      dan satu berita ekonomi domestik terbaru yang signifikan.
   b) Olahraga: satu update Timnas Indonesia, liga lokal, atau atlet Indonesia di ajang internasional.
   Sumber WAJIB dari media besar Indonesia: CNN Indonesia, CNBC Indonesia, Bisnis.com, Kompas.com, Detik.com, Kontan.co.id, Antara, Tempo.co, Liputan6, Bola.com.
   Query pencarian: gunakan Bahasa Indonesia.

5. **Trending Indonesia**
   Cari: satu berita viral/ trending yang sedang ramai dibicarakan di Indonesia dalam 1-2 hari terakhir.
   Bisa tentang bencana alam, sosial, politik ringan, fenomena unik, atau apa pun yang banyak diberitakan.
   Sumber: media besar Indonesia (sama seperti di atas).
   Contoh (hanya ilustrasi): "Erupsi Anak Krakatau hari ini", "Viral video ...", "Fenomena ...".
   Query pencarian: gunakan Bahasa Indonesia dengan kata kunci "viral", "trending", "ramai", atau kejadian aktual.

CARA KERJA YANG BENAR (sistem akan VERIFIKASI secara teknis):
- LANGKAH 1: Untuk SETIAP topik, lakukan SATU pencarian (web_search) dengan query spesifik.
- LANGKAH 2: Pilih SATU URL terbaik dari hasil pencarian, lalu kunjungi dengan fetch_webpage(url).
- LANGKAH 3: Ekstrak data konkret: angka, nama, tanggal, kutipan langsung dari artikel yang dibaca.
- LANGKAH 4: JANGAN melakukan pencarian berulang untuk topik yang sama. Jika halaman error, coba URL lain dari hasil pencarian yang sama, tetapi jangan lebih dari 2 kali percobaan.
- LANGKAH 5: Setelah semua topik selesai, langsung tulis laporan akhir dalam format naratif.

PENTING TENTANG FORMAT OUTPUT KODE:
- Gunakan SELALU tag <code> ... </code> untuk blok kode Python.
- JANGAN gunakan triple backtick (```) atau ```python.
- JANGAN mencampur <code> dengan tag lain.
- Setiap langkah harus dimulai dengan pemikiran singkat, lalu blok kode, contoh:
  <code>
  hasil = web_search(query="...")
  print(hasil)
  </code>
  Kemudian langkah berikutnya:
  <code>
  artikel = fetch_webpage(url="...")
  print(artikel)
  </code>

PENTING TENTANG ANTI-HALUSINASI:
- Setiap angka penting (harga, level, skor, persentase) yang kamu tulis harus benar-benar ada di teks hasil fetch_webpage.
- Jangan menambahkan angka dari ingatan atau perkiraan.
- Untuk setiap topik, tulis minimal SATU kalimat kutipan langsung dari artikel yang kamu baca (gunakan tanda kutip).
- Jika kamu tidak menemukan data spesifik di halaman, tulis "Data spesifik tidak ditemukan di sumber" daripada mengarang.

PENTING TENTANG KEDALAMAN LAPORAN:
- Setiap topik minimal 2-3 paragraf naratif, bukan satu paragraf singkat.
- Untuk setiap berita, WAJIB sertakan:
  * Tanggal publikasi atau tanggal kejadian (misal: "6 September 2026").
  * Nama media sumber dan URL.
  * Kutipan langsung singkat dari artikel (1-2 kalimat).
  * Analisis singkat: mengapa ini penting, dampaknya, atau konteksnya.
- Jangan hanya menulis kesimpulan kering, bangun cerita yang hidup dan mudah dipahami.

FORMAT LAPORAN YANG DIHARAPKAN — WAJIB diawali dengan judul persis:
"## 📊 LAPORAN MENDALAM — {tanggal.upper()}"
lalu tiap bagian pakai heading topiknya (Geopolitik, Olahraga, Teknologi, Indonesia, Trending Indonesia).

ATURAN KETAT:
- Setiap topik WAJIB punya minimal 1 URL sumber valid yang dicantumkan.
- Sumber harus berasal dari daftar domain yang disebutkan di atas (atau media besar lain yang relevan).
- DILARANG mengarang angka, skor, atau kutipan — hanya dari artikel yang beneran dibaca.
- Jangan menampilkan hasil mentah (JSON, request_id, dll) di laporan akhir.
- Panjang laporan TIDAK dibatasi — sistem Telegram otomatis pecah jadi beberapa pesan.
- Tulis bahasa Indonesia santai, boleh campur Inggris, seperti teman diskusi yang pintar.
- JANGAN keluarkan teks lain selain laporan itu sendiri.
"""

    MIN_VISIT = 4
    MAX_COBA  = 2

    agent = buat_agent()

    try:
        hasil       = None
        n_visit     = 0

        for percobaan in range(1, MAX_COBA + 1):
            log.info(f"Menjalankan agent (percobaan {percobaan}/{MAX_COBA})...")
            hasil   = agent.run(tugas)
            n_visit = hitung_visit_sukses(agent)
            log.info(f"Validasi: {n_visit}x fetch_webpage sukses (minimum {MIN_VISIT}).")
            log.info(f"URL yang berhasil di-fetch: {FETCHED_URLS}")

            ok, alasan = laporan_valid(hasil, FETCHED_URLS)
            if not ok:
                log.warning(f"Laporan tidak valid: {alasan}")
                if percobaan < MAX_COBA:
                    log.warning("Mengulang percobaan karena output tidak valid...")
                    continue

            if n_visit >= MIN_VISIT and ok:
                log.info("Validasi LULUS — laporan berbasis riset nyata dan format benar.")
                break
            elif percobaan < MAX_COBA:
                log.warning(f"Kurang riset atau format salah. Coba ulang...")

        # Validasi final
        ok, alasan = laporan_valid(hasil, FETCHED_URLS)
        if not ok:
            pesan_error = f"❌ Laporan hari ini gagal divalidasi ({alasan}). Tidak dikirim/disimpan untuk menjaga kualitas history."
            log.error(pesan_error)
            kirim_ke_telegram(pesan_error)
            return

        # Jika n_visit kurang dari minimum, batalkan pengiriman
        if n_visit < MIN_VISIT:
            pesan_error = f"❌ Bot hanya berhasil mengunjungi {n_visit} sumber (minimal {MIN_VISIT}). Laporan tidak dikirim karena berpotensi tidak akurat."
            log.error(pesan_error)
            kirim_ke_telegram(pesan_error)
            return

        log.info("LAPORAN FINAL:")
        log.info(hasil)

        kirim_ke_telegram(hasil)
        simpan_history(hasil)
        log.info("Selesai — laporan terkirim dan history disimpan.")

    except Exception as e:
        pesan_error = f"❌ Bot gagal generate laporan: {e}"
        log.error(pesan_error, exc_info=True)
        try:
            kirim_ke_telegram(pesan_error)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        jalankan_analisa_harian()
    except Exception as e:
        log.error(f"Error fatal: {e}", exc_info=True)
        raise
