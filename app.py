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
GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
GOOGLE_API_KEY     = os.environ["GOOGLE_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

TELEGRAM_MAX_CHARS = 3800

# =========================================================
# 2. Logging ke file + console
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("market-bot")

# =========================================================
# 3. History (biar AI nggak ngulang topik yang sama)
# =========================================================
HISTORY_FILE       = "history.json"
MAX_HISTORY        = 5


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


# =========================================================
# 4. Kirim Telegram (otomatis pecah kalau panjang)
# =========================================================
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


# =========================================================
# 5. Tool pencarian 24 jam terakhir
# =========================================================
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
        try:
            results = DDGS().text(query, timelimit="d", max_results=8)
        except Exception as e:
            return f"Pencarian gagal: {e}"

        if not results:
            try:
                results = DDGS().text(query, timelimit="w", max_results=8)
            except Exception:
                return "Tidak ada hasil ditemukan."

        if not results:
            return "Tidak ada hasil ditemukan, coba kata kunci lain."

        out = ""
        for r in results:
            out += f"- {r.get('title','')}\n  {r.get('body','')}\n  URL: {r.get('href','')}\n\n"
        return out


# =========================================================
# 6. Multi-provider AI dengan fallback cepat
#    Urutan: Gemini Flash-Lite (1500 req/day) → Groq → OpenRouter
#
#    Kenapa Flash-Lite PERTAMA:
#    - Gemini 2.5 Flash (lama) cuma 20 req/hari → habis cepat
#    - Gemini 2.5 Flash-Lite = 1000 req/hari, 15 RPM → jauh lebih lega
#    - Kualitas Flash-Lite cukup untuk task riset+laporan ini
#    - Lebih patuh instruksi dibanding model kecil OpenRouter random
# =========================================================
class FallbackModel:
    def __init__(self, providers):
        self.providers = []
        for p in providers:
            try:
                m = OpenAIServerModel(
                    model_id=p["model_id"],
                    api_base=p["api_base"],
                    api_key=p["api_key"],
                    # Matikan retry internal openai SDK — biar langsung pindah provider
                    # kalau kena error, nggak nunggu ratusan detik.
                    client_kwargs={"max_retries": 0, "timeout": 60.0},
                    # Matikan retry internal smolagents — lapisan berbeda dari atas.
                    retry=False,
                )
                self.providers.append({
                    "name":       p["name"],
                    "model":      m,
                    "rpm_limit":  p.get("rpm_limit"),   # proaktif skip kalau hampir limit
                    "timestamps": [],
                })
                log.info(f"Provider siap: {p['name']}")
            except Exception as e:
                log.warning(f"Gagal siapkan provider {p['name']}: {e}")

        if not self.providers:
            raise RuntimeError("Tidak ada provider AI yang bisa disiapkan!")

    def __getattr__(self, attr):
        return getattr(self.providers[0]["model"], attr)

    def _boleh_pakai(self, entry):
        """Cek proaktif RPM — skip kalau udah mepet limit, biar nggak kena 429."""
        limit = entry.get("rpm_limit")
        if not limit:
            return True
        now = time.time()
        entry["timestamps"] = [t for t in entry["timestamps"] if now - t < 60]
        return len(entry["timestamps"]) < limit

    def _try_all(self, method_name, *args, **kwargs):
        last_err = None
        for entry in self.providers:
            if not self._boleh_pakai(entry):
                log.info(f"Skip {entry['name']} (proaktif, deket RPM limit)...")
                continue
            try:
                log.info(f"Mencoba provider: {entry['name']}...")
                result = getattr(entry["model"], method_name)(*args, **kwargs)
                entry["timestamps"].append(time.time())
                log.info(f"✅ Berhasil pakai: {entry['name']}")
                return result
            except Exception as e:
                log.warning(f"⚠️ {entry['name']} gagal: {e}")
                last_err = e

        # Kalau semua skip proaktif, paksa coba provider pertama
        entry = self.providers[0]
        log.warning("Semua provider di-skip proaktif, paksa coba provider pertama...")
        try:
            result = getattr(entry["model"], method_name)(*args, **kwargs)
            entry["timestamps"].append(time.time())
            log.info(f"✅ Berhasil pakai (paksa): {entry['name']}")
            return result
        except Exception as e:
            last_err = e

        raise Exception(f"Semua provider gagal! Error terakhir: {last_err}")

    def generate(self, messages, stop_sequences=None, **kwargs):
        return self._try_all("generate", messages, stop_sequences=stop_sequences, **kwargs)

    def __call__(self, messages, stop_sequences=None, **kwargs):
        return self._try_all("__call__", messages, stop_sequences=stop_sequences, **kwargs)


def buat_agent():
    log.info("Menyiapkan AI dengan fallback chain (Flash-Lite → Groq → OpenRouter)...")
    model = FallbackModel([
        {
            # PROVIDER UTAMA: Gemini 2.5 Flash-Lite
            # Free tier: 1000 req/hari, 15 RPM — jauh lebih lega dari Flash biasa (20/hari).
            # Kualitas cukup bagus, patuh instruksi, format output konsisten.
            "name":      "Gemini 2.5 Flash-Lite",
            "model_id":  "gemini-2.5-flash-lite",
            "api_base":  "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key":   GOOGLE_API_KEY,
            "rpm_limit": 13,  # buffer dari limit asli 15 RPM
        },
        {
            # CADANGAN 1: Groq — cepat tapi ada limit token per menit (12.000 TPM)
            "name":      "Groq (Llama 3.3 70B)",
            "model_id":  "llama-3.3-70b-versatile",
            "api_base":  "https://api.groq.com/openai/v1",
            "api_key":   GROQ_API_KEY,
            "rpm_limit": None,
        },
        {
            # CADANGAN 2: OpenRouter — pakai openrouter/free (router resmi, nggak bisa ditarik)
            "name":      "OpenRouter (auto-router gratis)",
            "model_id":  "openrouter/free",
            "api_base":  "https://openrouter.ai/api/v1",
            "api_key":   OPENROUTER_API_KEY,
            "rpm_limit": None,
        },
    ])

    return CodeAgent(
        tools=[
            RecentNewsSearchTool(),
            # max_output_length=5000: batasi output per halaman biar konteks nggak membengkak
            # ke puluhan ribu token (default aslinya 40.000 karakter — terlalu besar).
            VisitWebpageTool(max_output_length=5000),
        ],
        model=model,
        additional_authorized_imports=["datetime", "os", "re"],
        max_steps=18,
    )


# =========================================================
# 7. Validasi teknis: cek beneran ada visit_webpage sukses
# =========================================================
def hitung_visit_sukses(agent):
    """
    Introspeksi riwayat langkah agent — hitung berapa kali visit_webpage()
    beneran dipanggil DAN berhasil (bukan error). Ini penegakan teknis,
    bukan cuma percaya klaim si AI di laporannya.
    """
    jumlah = 0
    for step in agent.memory.steps:
        code = getattr(step, "code_action", None)
        obs  = getattr(step, "observations", None) or ""
        if code and "visit_webpage(" in code and "Error fetching" not in obs:
            jumlah += 1
    return jumlah


# =========================================================
# 8. Fungsi utama analisa harian
# =========================================================
def jalankan_analisa_harian():
    log.info("=" * 55)
    log.info("MEMULAI ANALISA PASAR & BERITA GLOBAL OTOMATIS...")
    log.info("=" * 55)

    tanggal      = datetime.date.today().strftime("%d %B %Y")
    histori      = ringkasan_history()

    tugas = f"""
Hari ini tanggal {tanggal}. Fokus HANYA pada berita dari 1-2 hari terakhir (maksimal 48 jam ke belakang).

Laporan sebelumnya (JANGAN ulangi topik/angka yang persis sama, cari yang baru):
{histori}

Kamu adalah analis intelijen, pengamat olahraga, dan jurnalis teknologi senior.
Buat laporan 4 topik berikut:

1. **Geopolitik & Ekonomi Global** — berita internasional terbaru dan dampaknya ke kripto/saham.
2. **Olahraga Global** — hasil/jadwal pertandingan terkini (World Cup, Liga Champions, transfer pemain, dll).
3. **Teknologi & AI Terbaru** — satu fakta/terobosan teknologi atau AI dari 1-2 hari terakhir.
4. **Indonesia Update** — kondisi IHSG dan Rupiah hari ini, berita ekonomi domestik terbaru, plus satu update olahraga Indonesia. Sumber WAJIB dari media besar: CNN Indonesia, CNBC Indonesia, Bisnis.com, Kompas, Detik, Kontan, atau sejenisnya.

ATURAN WAJIB:
- Gunakan `web_search` dulu untuk cari berita, lalu `visit_webpage(url)` untuk baca isi artikel lengkapnya.
- WAJIB kunjungi minimal 1 artikel per topik via `visit_webpage` sebelum menulis laporan topik itu.
- Sistem akan MENDETEKSI secara teknis apakah kamu beneran baca artikel atau cuma mengarang.
- Cantumkan ANGKA SPESIFIK, nama konkret, tanggal/waktu di setiap poin — bukan kalimat generik.
- Khusus olahraga: sebutkan nama tim, skor, atau jadwal (jam+tanggal) yang konkret.
- Cantumkan URL sumber di setiap poin/topik.
- Untuk topik global, query pencarian dalam Bahasa Inggris. Untuk topik Indonesia, gunakan Bahasa Indonesia.
- Tulis dalam bahasa Indonesia santai, boleh campur sedikit Inggris.
- Panjang laporan tidak dibatasi — laporan panjang akan otomatis dipecah ke beberapa pesan Telegram.
"""

    MIN_VISIT = 3  # minimal 3 topik terbukti di-visit (sedikit lebih longgar dari 4)
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

            if n_visit >= MIN_VISIT:
                log.info("Validasi LULUS — laporan berbasis riset nyata.")
                break
            elif percobaan < MAX_COBA:
                log.warning(f"Kurang riset ({n_visit}x visit). Coba ulang...")

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


# =========================================================
# 9. Entry point — jalan SEKALI, cron GitHub Actions yang atur jadwal
# =========================================================
if __name__ == "__main__":
    try:
        jalankan_analisa_harian()
    except Exception as e:
        log.error(f"Error fatal: {e}", exc_info=True)
        raise
