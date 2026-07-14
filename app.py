import os
import time
import json
import logging
import datetime
import requests
from smolagents import Tool, CodeAgent, VisitWebpageTool, OpenAIServerModel

try:
    from ddgs import DDGS  # package baru, pakai ini kalau sudah di-pip install
except ImportError:
    from duckduckgo_search import DDGS  # fallback ke package lama

# =========================================================
# 1. Kredensial diambil dari Environment Variable / GitHub Secrets
#    JANGAN hardcode key di sini lagi!
# =========================================================
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# Batas aman Telegram sendMessage adalah 4096 karakter.
# Dikasih buffer di bawahnya biar aman dari karakter escape dsb.
TELEGRAM_MAX_CHARS = 3800

# =========================================================
# Logging ke file (run.log) + tetep tampil di console
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
# History biar AI gak ngulang topik yang sama
# =========================================================
HISTORY_FILE = "history.json"
MAX_HISTORY_ENTRIES = 5


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning(f"Gagal baca {HISTORY_FILE}: {e}")
        return []

    if not isinstance(data, list):
        log.warning(
            f"{HISTORY_FILE} isinya bukan list (tipe: {type(data).__name__}). "
            f"Mengabaikan history lama dan mulai dari kosong."
        )
        return []

    # Saring entri yang formatnya nggak sesuai skema {"tanggal": ..., "ringkasan": ...}
    # supaya history.json lama/rusak/beda-format nggak bikin crash, cukup di-skip.
    valid_entries = []
    for i, entry in enumerate(data):
        if (
            isinstance(entry, dict)
            and "tanggal" in entry
            and "ringkasan" in entry
            and isinstance(entry["ringkasan"], str)
        ):
            valid_entries.append(entry)
        else:
            log.warning(
                f"Entri history #{i} formatnya tidak sesuai (dapat: {type(entry).__name__} "
                f"= {str(entry)[:100]!r}), entri ini di-skip."
            )

    return valid_entries


def simpan_history(laporan_baru):
    history = load_history()
    history.append({
        "tanggal": datetime.date.today().isoformat(),
        "ringkasan": laporan_baru[:1500],  # potong biar file gak membengkak
    })
    history = history[-MAX_HISTORY_ENTRIES:]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        log.info(f"History disimpan ({len(history)} entri).")
    except Exception as e:
        log.warning(f"Gagal simpan {HISTORY_FILE}: {e}")


def ringkasan_history_untuk_prompt():
    history = load_history()  # sudah dijamin cuma berisi entri dict yang valid
    if not history:
        return "(belum ada histori laporan sebelumnya)"
    bagian = []
    for h in history:
        bagian.append(f"- [{h['tanggal']}] {h['ringkasan'][:300]}...")
    return "\n".join(bagian)


# =========================================================
# 2. Fungsi Kirim Telegram Pintar dengan Chunking + Fallback
# =========================================================
def _kirim_single_message(pesan, parse_mode=None):
    """Kirim 1 pesan ke Telegram. Return True kalau sukses."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": pesan}
    if parse_mode:
        data["parse_mode"] = parse_mode

    try:
        response = requests.post(url, data=data, timeout=30)
        if response.status_code == 200:
            return True
        log.warning(f"Gagal kirim (parse_mode={parse_mode}, status={response.status_code}): {response.text}")
        return False
    except Exception as e:
        log.error(f"Kesalahan koneksi Telegram: {e}")
        return False


def _split_pesan(pesan, max_len=TELEGRAM_MAX_CHARS):
    """
    Pecah pesan panjang jadi beberapa bagian tanpa motong di tengah kata,
    kalau bisa potong di baris baru dulu biar rapi.
    """
    chunks = []
    while len(pesan) > max_len:
        potong_di = pesan.rfind("\n", 0, max_len)
        if potong_di == -1 or potong_di < max_len * 0.5:
            potong_di = pesan.rfind(" ", 0, max_len)
        if potong_di == -1:
            potong_di = max_len

        chunks.append(pesan[:potong_di])
        pesan = pesan[potong_di:].lstrip("\n ")

    if pesan:
        chunks.append(pesan)
    return chunks


def _kirim_sebagai_dokumen(pesan, filename="laporan.txt"):
    """Fallback terakhir: kirim sebagai file .txt kalau semua cara lain gagal."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(pesan)
        with open(filename, "rb") as f:
            response = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": "Laporan (dikirim sebagai file karena kepanjangan)"},
                files={"document": f},
                timeout=60,
            )
        if response.status_code == 200:
            log.info("Berhasil dikirim sebagai file dokumen!")
            return True
        log.error(f"Gagal kirim dokumen juga. Status: {response.status_code}, Detail: {response.text}")
        return False
    except Exception as e:
        log.error(f"Kesalahan saat kirim dokumen: {e}")
        return False
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def kirim_ke_telegram(pesan):
    log.info("Mengirim laporan ke Telegram...")

    if len(pesan) <= TELEGRAM_MAX_CHARS:
        if _kirim_single_message(pesan, parse_mode="Markdown"):
            log.info("Berhasil dikirim (Format Markdown)!")
            return
        log.warning("Markdown gagal, coba teks biasa...")
        if _kirim_single_message(pesan):
            log.info("Berhasil dikirim (Format Teks Biasa)!")
            return
        log.warning("Teks biasa juga gagal, coba kirim sebagai dokumen...")
        _kirim_sebagai_dokumen(pesan)
        return

    log.info(f"Pesan sepanjang {len(pesan)} karakter, akan dipecah jadi beberapa bagian...")
    chunks = _split_pesan(pesan)
    semua_sukses = True

    for i, chunk in enumerate(chunks, start=1):
        prefix = f"📄 Bagian {i}/{len(chunks)}\n\n"
        chunk_dengan_prefix = prefix + chunk

        sukses = _kirim_single_message(chunk_dengan_prefix, parse_mode="Markdown")
        if not sukses:
            sukses = _kirim_single_message(chunk_dengan_prefix)

        if not sukses:
            semua_sukses = False
            log.error(f"Bagian {i}/{len(chunks)} gagal terkirim.")

        time.sleep(1)

    if semua_sukses:
        log.info(f"Semua {len(chunks)} bagian berhasil dikirim!")
    else:
        log.warning("Sebagian pesan gagal terkirim, coba kirim ulang full laporan sebagai dokumen...")
        _kirim_sebagai_dokumen(pesan)


# =========================================================
# 3. Tool Pencarian Berita KHUSUS 24 Jam Terakhir
# =========================================================
class RecentNewsSearchTool(Tool):
    name = "web_search"
    description = (
        "Cari berita/informasi TERBARU di internet (dibatasi hanya 24 jam terakhir). "
        "Mengembalikan STRING berisi daftar judul, ringkasan singkat, dan URL asli. "
        "Gunakan query dalam Bahasa Inggris untuk topik global/makro agar hasil lebih relevan."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Kata kunci pencarian, sebaiknya spesifik dan dalam Bahasa Inggris untuk topik global (misal: 'gold price today', 'crypto market news')"
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        try:
            results = DDGS().text(query, timelimit="d", max_results=8)
        except Exception as e:
            return f"Pencarian gagal: {e}"

        if not results:
            try:
                results = DDGS().text(query, timelimit="w", max_results=8)
            except Exception as e:
                return f"Pencarian gagal: {e}"

        if not results:
            return "Tidak ada hasil ditemukan. Coba kata kunci lain yang lebih spesifik atau dalam Bahasa Inggris."

        formatted = ""
        for r in results:
            formatted += f"- {r.get('title', '')}\n  {r.get('body', '')}\n  URL: {r.get('href', '')}\n\n"
        return formatted


# =========================================================
# 4. Model AI dengan Sistem Cadangan Berlapis (Fallback)
#    Urutan coba: Groq -> Google Gemini -> OpenRouter
# =========================================================
class FallbackModel:
    def __init__(self, providers):
        self.providers = []
        for p in providers:
            try:
                model_instance = OpenAIServerModel(
                    model_id=p["model_id"],
                    api_base=p["api_base"],
                    api_key=p["api_key"],
                )
                self.providers.append({"name": p["name"], "model": model_instance})
            except Exception as e:
                log.warning(f"Gagal menyiapkan provider {p['name']}: {e}")

        if not self.providers:
            raise RuntimeError("Tidak ada provider AI yang berhasil disiapkan!")

    def __getattr__(self, attr):
        return getattr(self.providers[0]["model"], attr)

    def _try_all(self, method_name, *args, **kwargs):
        last_error = None
        for entry in self.providers:
            try:
                log.info(f"Mencoba provider AI: {entry['name']}...")
                method = getattr(entry["model"], method_name)
                result = method(*args, **kwargs)
                log.info(f"Berhasil pakai provider: {entry['name']}")
                return result
            except Exception as e:
                log.warning(f"Provider {entry['name']} gagal: {e}")
                last_error = e
                continue
        raise Exception(f"Semua provider AI gagal dicoba! Error terakhir: {last_error}")

    def generate(self, messages, stop_sequences=None, **kwargs):
        return self._try_all("generate", messages, stop_sequences=stop_sequences, **kwargs)

    def __call__(self, messages, stop_sequences=None, **kwargs):
        return self._try_all("__call__", messages, stop_sequences=stop_sequences, **kwargs)


def buat_agent():
    log.info("Menyiapkan sistem AI dengan 3 lapis cadangan (Groq -> Google -> OpenRouter)...")
    model = FallbackModel([
        {
            "name": "Groq (Llama 3.3 70B)",
            "model_id": "llama-3.3-70b-versatile",
            "api_base": "https://api.groq.com/openai/v1",
            "api_key": GROQ_API_KEY,
        },
        {
            "name": "Google Gemini 2.5 Flash",
            "model_id": "gemini-2.5-flash",
            "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": GOOGLE_API_KEY,
        },
        {
            "name": "OpenRouter (Llama 3.3 70B Free)",
            "model_id": "meta-llama/llama-3.3-70b-instruct:free",
            "api_base": "https://openrouter.ai/api/v1",
            "api_key": OPENROUTER_API_KEY,
        },
    ])

    search_tool = RecentNewsSearchTool()
    visit_tool = VisitWebpageTool()

    log.info("Merakit agen super...")
    return CodeAgent(
        tools=[search_tool, visit_tool],
        model=model,
        additional_authorized_imports=["datetime", "os", "re"],
        max_steps=10
    )


# =========================================================
# 5. Fungsi Utama Analisa Harian
# =========================================================
def jalankan_analisa_harian():
    log.info("=" * 50)
    log.info("MEMULAI ANALISA PASAR & BERITA GLOBAL OTOMATIS...")
    log.info("=" * 50)

    tanggal_hari_ini = datetime.date.today().strftime("%d %B %Y")
    histori_sebelumnya = ringkasan_history_untuk_prompt()

    tugas = f"""
    Hari ini tanggal {tanggal_hari_ini}. HANYA gunakan berita dan data dari 24 jam terakhir.
    Jika hasil pencarian ternyata berita lama (lebih dari 2 hari), abaikan dan cari ulang dengan kata kunci lain.

    Berikut ringkasan laporan-laporan SEBELUMNYA yang sudah dikirim (JANGAN ulangi topik/angka yang sama persis,
    cari perkembangan terbaru atau sudut pandang/berita lain yang belum pernah dibahas):
    {histori_sebelumnya}

    Kamu adalah seorang analis intelijen, pengamat olahraga, dan jurnalis teknologi senior.
    Tugasmu hari ini adalah mencari dan menganalisa 3 topik utama:
    1. Geopolitik & Ekonomi Global (fokus pada berita luar negeri internasional dan dampaknya ke Kripto/Saham).
    2. Update Olahraga Global yang sedang tren hari ini (seperti update World Cup, Liga Champions, transfer pemain bintang, dll).
    3. Satu fakta teknologi, sains, atau AI terbaru hari ini.

    ATURAN AGENTIK SANGAT KETAT:
    - Alat `web_search` mengembalikan STRING panjang (bukan list), berisi daftar judul, ringkasan, dan URL yang sudah rapi per baris.
    - Untuk topik global/makro, gunakan query Bahasa Inggris agar hasil pencarian lebih relevan (mesin pencari lebih kaya hasil untuk Bahasa Inggris).
    - Gunakan `visit_webpage(url_tersebut)` untuk masuk dan membaca isi beritanya secara penuh.
    - Pastikan URL yang kamu kunjungi TIDAK memiliki spasi (contoh salah: 'bbc. com/sport', contoh benar: 'bbc.com/sport').
    - Ekstrak DATA VALID, ANGKA SPESIFIK, dan FAKTA NYATA dari dalam artikel. Jangan berikan kesimpulan kosong tanpa data penjelas!
    - Jika 2 pencarian berturut-turut tidak menemukan hasil relevan, JANGAN cari terus, langsung lanjut menulis laporan dengan data yang sudah ada.
    - Tulis laporan akhir secara MENDALAM dan RINCI per topik — sertakan ANGKA SPESIFIK, PERSENTASE, NILAI NOMINAL,
      dan konteks/latar belakang yang jelas untuk tiap poin (bukan cuma kesimpulan umum tanpa data pendukung).
      Panjang laporan TIDAK dibatasi; laporan panjang akan otomatis dipecah jadi beberapa pesan Telegram, jadi jangan
      memotong analisis demi keringkasan.
    - Wajib sertakan URL sumber referensi asli yang valid di SETIAP poin/topik (bukan cuma sekali di akhir), agar user
      bisa memverifikasi tiap klaim ke sumber aslinya masing-masing.
    - Gunakan bahasa Indonesia santai (campur sedikit bahasa Inggris gaul layaknya teman diskusi yang sangat pintar).
    """

    agent = buat_agent()

    try:
        log.info("Menjalankan agent...")
        hasil = agent.run(tugas)

        # Catat isi laporan lengkap ke run.log (bukan cuma status kirim),
        # biar log-nya selengkap output yang kamu lihat di PowerShell.
        log.info("=" * 50)
        log.info("HASIL LAPORAN LENGKAP:")
        log.info("=" * 50)
        log.info(hasil)
        log.info("=" * 50)

        kirim_ke_telegram(hasil)
        simpan_history(hasil)
        log.info("Selesai satu siklus analisa dengan sukses.")
    except Exception as e:
        error_msg = f"Waduh bro, semua provider AI-nya gagal saat mikir nih: {e}"
        log.error(error_msg, exc_info=True)
        try:
            kirim_ke_telegram(error_msg)
        except Exception as e2:
            log.error(f"Kirim pesan error ke Telegram juga gagal: {e2}")


# =========================================================
# 6. Entry point — jalan SEKALI per eksekusi.
#    Penjadwalan tiap 12 jam sekarang dihandle oleh GitHub Actions cron,
#    bukan while True + time.sleep(43200) lagi.
# =========================================================
if __name__ == "__main__":
    try:
        jalankan_analisa_harian()
    except Exception as e:
        # Pengaman terakhir: kalau ada error di luar dugaan (misal semua provider
        # gagal SAAT buat_agent(), sebelum masuk try/except di dalam fungsi),
        # tetap kecatat jelas di run.log alih-alih bikin job merah tanpa penjelasan.
        log.error(f"Error fatal yang tidak tertangani: {e}", exc_info=True)
        raise
