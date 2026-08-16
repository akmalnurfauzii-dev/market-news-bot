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
# Opsional — kalau belum ada secret CEREBRAS_API_KEY, fallback Cerebras di-skip otomatis.
CEREBRAS_API_KEY   = os.environ.get("CEREBRAS_API_KEY", "")

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
#    Urutan: Gemini Flash → Groq (gpt-oss-120b) → OpenRouter → Cerebras
#
#    ⚠️ CATATAN PENTING (update 17 Agustus 2026):
#    - Groq resmi men-deprecate llama-3.3-70b-versatile & llama-3.1-8b-instant
#      per 17 Juni 2026. Model diganti ke openai/gpt-oss-120b.
#    - Cerebras katalog free tier-nya SANGAT volatile — pernah collapse dari
#      belasan model jadi cuma 2 model dalam semalam. Jangan taruh sebagai
#      provider utama, hanya cadangan terakhir kalau 3 provider lain gagal.
#    - Selalu cek dashboard/docs resmi tiap provider tiap beberapa bulan,
#      karena nama model gratis bisa berubah tanpa notifikasi ke user.
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
    log.info("Menyiapkan AI dengan fallback chain (Gemini → Groq → OpenRouter → Cerebras)...")

    daftar_provider = [
        {
            # PROVIDER UTAMA: Gemini 2.5 Flash
            # Terbukti paling patuh instruksi dan hasilnya paling detail dari semua provider.
            # Masih free tier per Agustus 2026 (1.500 req/hari, 15 RPM).
            "name":      "Gemini 2.5 Flash",
            "model_id":  "gemini-2.5-flash",
            "api_base":  "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key":   GOOGLE_API_KEY,
            "rpm_limit": 4,   # buffer dari limit asli 5 RPM (kalau masih di quota lama)
        },
        {
            # CADANGAN 1: Groq — model DIPERBARUI 17 Agustus 2026.
            # llama-3.3-70b-versatile SUDAH DEPRECATED sejak 17 Juni 2026.
            # openai/gpt-oss-120b adalah pengganti resmi yang direkomendasikan Groq.
            "name":      "Groq (GPT-OSS 120B)",
            "model_id":  "openai/gpt-oss-120b",
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
    ]

    # CADANGAN 3 (opsional): Cerebras — hanya diaktifkan kalau CEREBRAS_API_KEY di-set.
    # ⚠️ Katalog model gratis Cerebras SANGAT sering berubah tanpa peringatan.
    # gpt-oss-120b dipilih karena per 17 Agustus 2026 ini salah satu model paling stabil
    # di free tier mereka (model lain seperti zai-glm-4.7 sedang dalam proses deprecation
    # di tanggal yang sama). Context window free tier Cerebras juga dibatasi ~8K token,
    # jadi taruh ini SELALU sebagai fallback TERAKHIR, bukan primary.
    if CEREBRAS_API_KEY:
        daftar_provider.append({
            "name":      "Cerebras (GPT-OSS 120B)",
            "model_id":  "gpt-oss-120b",
            "api_base":  "https://api.cerebras.ai/v1",
            "api_key":   CEREBRAS_API_KEY,
            "rpm_limit": None,
        })
    else:
        log.info("CEREBRAS_API_KEY belum di-set — fallback Cerebras di-skip (opsional).")

    model = FallbackModel(daftar_provider)

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
LANGKAH 1 — Untuk tiap topik: web_search() dulu cari artikel relevan dari sumber terpercaya di atas.
LANGKAH 2 — Kunjungi artikel via visit_webpage(url) untuk baca isi lengkapnya.
LANGKAH 3 — Ekstrak data konkret: angka, nama, tanggal, kutipan langsung dari artikel yang dibaca.
LANGKAH 4 — Tulis laporan SETELAH membaca, bukan mengarang dari ingatan.

FORMAT LAPORAN YANG DIHARAPKAN (seperti ini, bukan kesimpulan singkat):
❌ SALAH: "Pasar kripto mengalami volatilitas akibat sentimen global."
✅ BENAR: "Bitcoin turun 2,3% ke $63.400 pada Senin pagi setelah data inflasi AS bulan Juni
  menunjukkan CPI naik 3,1% YoY, lebih tinggi dari ekspektasi 2,9%. Analis dari JPMorgan
  menyebut ini bisa menunda pemangkasan suku bunga Fed ke kuartal 4. (Sumber: Reuters)"

❌ SALAH: "IHSG bergerak mixed hari ini."
✅ BENAR: "IHSG ditutup melemah 47,3 poin (-0,72%) ke level 6.534,21 pada Senin 28 Juli 2026,
  tertekan oleh aksi jual asing senilai Rp892 miliar. Sektor perbankan turun paling dalam -1,4%.
  (Sumber: CNBC Indonesia)"

ATURAN KETAT:
- Setiap topik WAJIB punya minimal 1 URL sumber valid yang dicantumkan.
- DILARANG mengarang angka, skor, atau kutipan — hanya dari artikel yang beneran dibaca.
- Kalau halaman web error/403, coba URL lain dari hasil pencarian yang sama.
- Panjang laporan TIDAK dibatasi — sistem Telegram otomatis pecah jadi beberapa pesan.
- Tulis bahasa Indonesia santai, boleh campur Inggris, seperti teman diskusi yang pintar.
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
