import os
import time
import json
import logging
import datetime
import requests
from smolagents import Tool, CodeAgent, VisitWebpageTool, OpenAIServerModel

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

# Marker laporan yang WAJIB ada
REPORT_MARKERS = ["## 📊 LAPORAN MENDALAM", "Geopolitik", "Olahraga", "Teknologi", "Indonesia"]
MIN_REPORT_LENGTH = 400

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

def laporan_valid(teks):
    """
    Validasi ketat:
    - minimal panjang
    - wajib ada judul laporan
    - minimal 3 bagian topik (Geopolitik, Olahraga, Teknologi, Indonesia)
    - tidak boleh ada pola output rusak (JSON, tag </code, dll)
    """
    if not teks or len(teks.strip()) < MIN_REPORT_LENGTH:
        return False, f"Terlalu pendek ({len(teks.strip()) if teks else 0} karakter, minimal {MIN_REPORT_LENGTH})"

    if "## 📊 LAPORAN MENDALAM" not in teks:
        return False, "Tidak ada judul '## 📊 LAPORAN MENDALAM'"

    # Cek minimal 3 topik muncul sebagai heading atau kata
    topik = ["Geopolitik", "Olahraga", "Teknologi", "Indonesia"]
    ditemukan = sum(1 for t in topik if t.lower() in teks.lower())
    if ditemukan < 3:
        return False, f"Hanya {ditemukan} topik ditemukan, minimal 3 (Geopolitik, Olahraga, Teknologi, Indonesia)"

    # Pola output rusak yang sering muncul
    bad_patterns = [
        "</code", "User Safety", "Response Safety",
        '"tool": "web_search"', '"request_id"', '```',
    ]
    for pat in bad_patterns:
        if pat in teks:
            return False, f"Terdeteksi output rusak: {pat}"

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
        elif _kirim_satu(teks):
            log.info(f"✅ Bagian {i}/{len(chunks)} terkirim (plain text)!")
        else:
            log.error(f"❌ Bagian {i}/{len(chunks)} GAGAL terkirim.")
        time.sleep(1)


class RecentNewsSearchTool(Tool):
    name        = "web_search"
    description = (
        "Cari berita/informasi TERBARU dari 24 jam terakhir. "
        "Untuk topik global gunakan query Bahasa Inggris. "
        "Untuk topik Indonesia gunakan Bahasa Indonesia. "
        "Kembalikan STRING berisi judul, ringkasan, dan URL."
    )
    inputs      = {"query": {"type": "string", "description": "Kata kunci pencarian"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        # Hanya gunakan backend html duckduckgo untuk mengurangi request ke banyak engine
        try:
            # Parameter backend="html" memaksa hanya satu engine
            results = DDGS(backend="html").text(query, timelimit="d", max_results=5)
        except Exception:
            # Fallback ke default
            try:
                results = DDGS().text(query, timelimit="d", max_results=5)
            except Exception as e:
                return f"Pencarian gagal: {e}"

        if not results:
            # Coba 7 hari jika 24 jam kosong
            try:
                results = DDGS(backend="html").text(query, timelimit="w", max_results=5)
            except Exception:
                return "Tidak ada hasil ditemukan."

        if not results:
            return "Tidak ada hasil ditemukan, coba kata kunci lain."

        out = ""
        for r in results:
            out += f"- {r.get('title','')}\n  {r.get('body','')}\n  URL: {r.get('href','')}\n\n"
        return out


class FallbackModel:
    """
    Multi‑provider dengan beberapa model per provider.
    Model diurutkan dari yang paling ringan/murah ke yang lebih mahal.
    Jika model mati (404/410/retired), otomatis ditandai dan tidak dicoba lagi.
    Tidak ada rate limit proaktif, karena error 429 akan ditangani langsung.
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

    def _try_all(self, method_name, *args, **kwargs):
        last_err = None
        for entry in self.providers:
            # Lewati provider jika semua model sudah dead
            if all(entry["dead"]):
                log.info(f"Provider {entry['name']} semua model mati, skip.")
                continue

            # Coba model mulai dari current_idx
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
                        break  # keluar while, lanjut provider berikutnya
                    else:
                        # Error lain (timeout, 5xx, dsb) → coba model berikutnya di provider yang sama
                        last_err = e
                        entry["current_idx"] += 1
                        continue

            # Jika keluar dari while karena current_idx >= len models, tandai semua sudah dicoba
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
    log.info("Menyiapkan AI dengan fallback chain multi‑model (otomatis pilih dari murah ke mahal)...")

    daftar_provider = [
        {
            "name": "Gemini",
            "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": GOOGLE_API_KEY,
            # Urutan dari yang PALING MURAH ke lebih mahal
            "models": [
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
                "gemini-2.5-flash-lite",
                "gemini-3.5-flash",
                "gemini-3.6-flash",
                "gemini-3.7-flash",
                "gemini-3.8-flash",
                "gemini-3.1-pro-preview",
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
                "deepseek-ai/deepseek-v3.1",
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
                "openai/gpt-oss-120b",
            ],
        })

    model = FallbackModel(daftar_provider)

    return CodeAgent(
        tools=[
            RecentNewsSearchTool(),
            VisitWebpageTool(max_output_length=2500),  # Batasi output agar hemat token
        ],
        model=model,
        additional_authorized_imports=["datetime", "os", "re"],
        max_steps=8,  # Dikurangi dari 10 ke 8
    )


def hitung_visit_sukses(agent):
    jumlah = 0
    for step in agent.memory.steps:
        code = getattr(step, "code_action", None)
        obs  = getattr(step, "observations", None) or ""
        if code and "visit_webpage(" in code and "Error fetching" not in obs:
            jumlah += 1
    return jumlah


def jalankan_analisa_harian():
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
Buat laporan mendalam untuk 4 topik ini:

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
   Sumber WAJIB dari media besar Indonesia: CNN Indonesia (cnnindonesia.com), CNBC Indonesia (cnbcindonesia.com),
   Bisnis.com, Kompas.com, Detik.com, Kontan.co.id, Antara, atau Tempo.co.
   Query pencarian: gunakan Bahasa Indonesia.

CARA KERJA YANG BENAR (sistem akan VERIFIKASI secara teknis):
- LANGKAH 1: Untuk SETIAP topik, lakukan SATU pencarian (web_search) dengan query spesifik.
- LANGKAH 2: Pilih SATU URL terbaik dari hasil pencarian itu, lalu kunjungi dengan visit_webpage(url).
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

FORMAT LAPORAN YANG DIHARAPKAN — WAJIB diawali dengan judul persis:
"## 📊 LAPORAN MENDALAM — {tanggal.upper()}"
lalu tiap bagian pakai heading topiknya (Geopolitik, Olahraga, Teknologi, Indonesia).

ATURAN KETAT:
- Setiap topik WAJIB punya minimal 1 URL sumber valid yang dicantumkan.
- DILARANG mengarang angka, skor, atau kutipan — hanya dari artikel yang beneran dibaca.
- Jangan menampilkan hasil mentah (JSON, request_id, dll) di laporan akhir.
- Panjang laporan TIDAK dibatasi — sistem Telegram otomatis pecah jadi beberapa pesan.
- Tulis bahasa Indonesia santai, boleh campur Inggris, seperti teman diskusi yang pintar.
- JANGAN keluarkan teks lain selain laporan itu sendiri.
"""

    MIN_VISIT = 3
    MAX_COBA  = 2

    agent = buat_agent()

    try:
        hasil       = None
        n_visit     = 0

        for percobaan in range(1, MAX_COBA + 1):
            log.info(f"Menjalankan agent (percobaan {percobaan}/{MAX_COBA})...")
            hasil   = agent.run(tugas)
            n_visit = hitung_visit_sukses(agent)
            log.info(f"Validasi: {n_visit}x visit_webpage sukses (minimum {MIN_VISIT}).")

            ok, alasan = laporan_valid(hasil)
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

        ok, alasan = laporan_valid(hasil)
        if not ok:
            pesan_error = f"❌ Laporan hari ini gagal divalidasi ({alasan}). Tidak dikirim/disimpan untuk menjaga kualitas history."
            log.error(pesan_error)
            kirim_ke_telegram(pesan_error)
            return

        if n_visit < MIN_VISIT:
            peringatan = (
                f"⚠️ *Catatan sistem:* AI hanya mengunjungi {n_visit} sumber "
                f"(kurang dari {MIN_VISIT} yang diharapkan). Verifikasi mandiri disarankan.\n\n"
            )
            hasil = peringatan + hasil

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
