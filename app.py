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
#    Urutan coba: Google Gemini -> Groq -> OpenRouter
# =========================================================
def _estimasi_token(messages):
    """
    Estimasi KASAR jumlah token dari daftar messages, dipakai buat pre-check proaktif
    SEBELUM manggil API (kita belum tau token asli sebelum dapat respons). Heuristik:
    ~1 token per 3 karakter (sedikit overestimate, sengaja dibuat konservatif/aman).
    """
    total_karakter = 0
    for m in messages:
        konten = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(konten, list):
            for bagian in konten:
                if isinstance(bagian, dict):
                    total_karakter += len(str(bagian.get("text", bagian)))
                else:
                    total_karakter += len(str(bagian))
        else:
            total_karakter += len(str(konten))
    return total_karakter // 3


class FallbackModel:
    def __init__(self, providers):
        self.providers = []
        for p in providers:
            try:
                extra_kwargs = {}
                if "extra_body" in p:
                    # extra_body diteruskan ke openai client's create() call apa adanya.
                    # Dipakai OpenRouter buat fitur 'models' array (multi-model auto-fallback
                    # dalam satu request) -- lihat konfigurasi provider OpenRouter di atas.
                    extra_kwargs["extra_body"] = p["extra_body"]

                model_instance = OpenAIServerModel(
                    model_id=p["model_id"],
                    api_base=p["api_base"],
                    api_key=p["api_key"],
                    # client_kwargs max_retries=0: matikan retry level HTTP client (openai SDK).
                    client_kwargs={"max_retries": 0, "timeout": 60.0},
                    # retry=False: MATIKAN JUGA retry internal smolagents sendiri (ApiModel.retryer).
                    # Ini lapisan TERPISAH dari client_kwargs di atas! Defaultnya smolagents retry
                    # 3x dengan wait 60s x 2^percobaan (60s, 120s, 240s + jitter) khusus buat error
                    # rate limit (429) -- ini yang beneran nyebabin stall 2-10 menit per provider,
                    # bukan client_kwargs. Dengan retry=False, begitu kena 429, LANGSUNG dilempar
                    # ke _try_all() di bawah buat pindah provider dalam hitungan detik.
                    retry=False,
                    **extra_kwargs,
                )
                self.providers.append({
                    "name": p["name"],
                    "model": model_instance,
                    # rpm_limit: batas REQUEST/menit yang kita tetapkan sendiri (dipakai Gemini).
                    "rpm_limit": p.get("rpm_limit"),
                    "call_timestamps": [],
                    # tpm_limit: batas TOKEN/menit yang kita tetapkan sendiri (dipakai Groq).
                    # Beda dari rpm_limit: ini ngitung TOKEN kumulatif, bukan jumlah request,
                    # soalnya Groq nolak berdasarkan total token terpakai dalam 60 detik terakhir
                    # (dikonfirmasi dari pesan error API: "Used 8452, Requested 9925").
                    "tpm_limit": p.get("tpm_limit"),
                    "token_usage_log": [],  # list (timestamp, jumlah_token) dari call SUKSES
                })
            except Exception as e:
                log.warning(f"Gagal menyiapkan provider {p['name']}: {e}")

        if not self.providers:
            raise RuntimeError("Tidak ada provider AI yang berhasil disiapkan!")

    def __getattr__(self, attr):
        return getattr(self.providers[0]["model"], attr)

    def _dalam_batas_rpm(self, entry):
        """Proaktif: masih dalam jatah REQUEST/menit sendiri? (dipakai buat Gemini)"""
        limit = entry.get("rpm_limit")
        if not limit:
            return True
        now = time.time()
        entry["call_timestamps"] = [t for t in entry["call_timestamps"] if now - t < 60]
        return len(entry["call_timestamps"]) < limit

    def _dalam_batas_tpm(self, entry, messages):
        """Proaktif: masih dalam jatah TOKEN/menit sendiri? (dipakai buat Groq).
        Pakai kombinasi token ASLI (dari call sukses sebelumnya) + estimasi kasar
        buat request yang mau dikirim sekarang, biar nggak nabrak limit sebelum kejadian."""
        limit = entry.get("tpm_limit")
        if not limit:
            return True
        now = time.time()
        entry["token_usage_log"] = [(t, tok) for t, tok in entry["token_usage_log"] if now - t < 60]
        token_terpakai = sum(tok for _, tok in entry["token_usage_log"])
        estimasi_request_ini = _estimasi_token(messages) + 1500  # +buffer buat token completion
        return (token_terpakai + estimasi_request_ini) <= limit

    def _try_all(self, method_name, *args, **kwargs):
        messages = args[0] if args else []
        last_error = None
        ada_yang_dicoba = False

        for entry in self.providers:
            if not self._dalam_batas_rpm(entry):
                log.info(
                    f"Skip {entry['name']} sementara (udah pakai {entry['rpm_limit']}x "
                    f"dalam 1 menit terakhir, proaktif hindari kena limit)..."
                )
                continue
            if not self._dalam_batas_tpm(entry, messages):
                log.info(
                    f"Skip {entry['name']} sementara (proyeksi token bakal ngelewatin "
                    f"{entry['tpm_limit']} token/menit, proaktif hindari kena limit)..."
                )
                continue

            ada_yang_dicoba = True
            try:
                log.info(f"Mencoba provider AI: {entry['name']}...")
                method = getattr(entry["model"], method_name)
                result = method(*args, **kwargs)

                # Catat pemakaian: request-count buat RPM, token ASLI (kalau ada) buat TPM.
                entry["call_timestamps"].append(time.time())
                token_asli = None
                if getattr(result, "token_usage", None):
                    token_asli = result.token_usage.input_tokens + result.token_usage.output_tokens
                if token_asli is None:
                    token_asli = _estimasi_token(messages)  # fallback kalau info asli nggak ada
                entry["token_usage_log"].append((time.time(), token_asli))

                log.info(f"Berhasil pakai provider: {entry['name']}")
                return result
            except Exception as e:
                log.warning(f"Provider {entry['name']} gagal: {e}")
                last_error = e
                continue

        if not ada_yang_dicoba:
            # Semua provider lagi 'jeda proaktif' -- daripada nyerah tanpa nyoba apapun,
            # lebih baik paksa coba provider pertama sebagai upaya terakhir.
            log.warning("Semua provider lagi dalam jeda proaktif, coba paksa provider pertama...")
            entry = self.providers[0]
            try:
                method = getattr(entry["model"], method_name)
                result = method(*args, **kwargs)
                entry["call_timestamps"].append(time.time())
                log.info(f"Berhasil pakai provider: {entry['name']} (paksa)")
                return result
            except Exception as e:
                last_error = e

        raise Exception(f"Semua provider AI gagal dicoba! Error terakhir: {last_error}")

    def generate(self, messages, stop_sequences=None, **kwargs):
        return self._try_all("generate", messages, stop_sequences=stop_sequences, **kwargs)

    def __call__(self, messages, stop_sequences=None, **kwargs):
        return self._try_all("__call__", messages, stop_sequences=stop_sequences, **kwargs)


def buat_agent():
    log.info("Menyiapkan sistem AI dengan 3 lapis cadangan (Google -> Groq -> OpenRouter)...")

    model = FallbackModel([
        {
            # Diletakkan PERTAMA: Gemini terbukti patuh instruksi 'wajib visit_webpage per topik'
            # dan hasilnya jauh lebih detail (ada angka spesifik). Groq (Llama 3.3 70B) terbukti
            # SELALU skip visit_webpage & sering kena rate limit 429.
            "name": "Google Gemini 2.5 Flash",
            "model_id": "gemini-2.5-flash",
            "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": GOOGLE_API_KEY,
            # rpm_limit=4: limit ASLI dari Google cuma 5 request/menit (ditemukan dari error API
            # 22 Juli 2026: quotaValue '5', quotaId GenerateRequestsPerMinutePerProjectPerModel).
            # Dikasih buffer jadi 4 (bukan 5 persis) buat jaga-jaga dari selisih waktu/jitter.
            # Kalau limit ini kesundul, sistem otomatis SKIP ke Groq/OpenRouter buat step itu,
            # tanpa nunggu 429 dulu -- baru balik pakai Gemini lagi begitu jendela 1 menitnya lewat.
            "rpm_limit": 4,
        },
        {
            "name": "Groq (Llama 3.3 70B)",
            "model_id": "llama-3.3-70b-versatile",
            "api_base": "https://api.groq.com/openai/v1",
            "api_key": GROQ_API_KEY,
            # tpm_limit=11000: limit ASLI Groq itu 12.000 token/menit (dikonfirmasi dari error API
            # 24 Juli 2026: "Limit 12000, Used 8452, Requested 9925"). Ini KUMULATIF dalam window
            # 60 detik, BUKAN per-request -- jadi meski 1 request kecil, kalau total pemakaian
            # 1 menit terakhir udah deket limit, tetep ditolak. Dikasih buffer jadi 11000 (bukan
            # 12000 persis) buat jaga-jaga dari estimasi token yang nggak 100% presisi.
            "tpm_limit": 11000,
        },
        {
            "name": "OpenRouter (auto-router gratis)",
            # PAKAI 'openrouter/free', BUKAN nama model provider tertentu (misal 'meta-llama/...').
            # Ini router RESMI dari OpenRouter sendiri (diluncurkan Feb 2026) yang otomatis milih
            # model gratis yang lagi hidup & mendukung fitur yang dibutuhkan (dikonfirmasi dari
            # openrouter.ai/openrouter/free). Karena ini ENTITAS ROUTER OpenRouter sendiri -- bukan
            # model dari provider pihak ketiga -- dia nggak akan pernah "ditarik dari tier gratis"
            # kayak yang udah 3x kejadian sama kita pas hardcode nama model spesifik
            # (meta-llama/llama-3.3-70b-instruct:free lalu openai/gpt-oss-120b:free, dua-duanya
            # akhirnya 404 dalam hitungan hari). Context window 200.000 token, $0/$0.
            "model_id": "openrouter/free",
            "api_base": "https://openrouter.ai/api/v1",
            "api_key": OPENROUTER_API_KEY,
        },
    ])

    search_tool = RecentNewsSearchTool()
    # max_output_length=5000 (default aslinya 40.000 karakter!). Halaman web penuh menu
    # navigasi, footer, iklan yang ikut ke-scrape dan numpuk ke memori percakapan tanpa guna.
    # Ini akar masalah sebenarnya kenapa konteks bisa meledak sampai 30.000-60.000+ token
    # dalam 4-6 kali visit halaman, jauh sebelum masalah rate limit provider jadi relevan.
    visit_tool = VisitWebpageTool(max_output_length=5000)

    log.info("Merakit agen super...")
    return CodeAgent(
        tools=[search_tool, visit_tool],
        model=model,
        additional_authorized_imports=["datetime", "os", "re"],
        max_steps=18
    )


# =========================================================
# 5. Fungsi Utama Analisa Harian
# =========================================================
def hitung_visit_webpage_sukses(agent):
    """
    Introspeksi LANGSUNG ke riwayat step agent (agent.memory.steps), BUKAN percaya
    klaim laporan dari si AI. Hitung berapa kali visit_webpage() BENERAN dipanggil
    di kode yang dieksekusi (code_action) DAN beneran berhasil (observations-nya
    nggak mengandung pesan error semacam 'Error fetching the webpage').

    Ini penegakan TEKNIS, bukan cuma instruksi di prompt -- soalnya kita udah
    kebukti prompt doang ('WAJIB visit_webpage') bisa diabaikan model kalau dia
    'yakin' udah tau jawabannya (kejadian 25 Juli 2026: laporan Copa América
    yang di-final_answer() di Step 1 TANPA riset apapun).
    """
    jumlah = 0
    for step in agent.memory.steps:
        code = getattr(step, "code_action", None)
        obs = getattr(step, "observations", None) or ""
        if code and "visit_webpage(" in code and "Error fetching the webpage" not in obs:
            jumlah += 1
    return jumlah


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
    Tugasmu hari ini adalah mencari dan menganalisa 4 topik utama:
    1. Geopolitik & Ekonomi Global (fokus pada berita luar negeri internasional dan dampaknya ke Kripto/Saham).
    2. Update Olahraga Global yang sedang tren hari ini (seperti update World Cup, Liga Champions, transfer pemain bintang, dll).
    3. Satu fakta teknologi, sains, atau AI terbaru hari ini.
    4. Indonesia Update — kondisi IHSG hari ini, nilai tukar Rupiah, berita ekonomi domestik terbaru, DAN
       satu update olahraga Indonesia (misal Timnas, liga lokal, atlet Indonesia di ajang internasional).
       Sumber WAJIB dari media besar Indonesia: CNN Indonesia, CNBC Indonesia, Bisnis.com, Kompas.com,
       Detik.com, Kontan, atau media besar sejenis -- JANGAN pakai blog/situs kecil yang nggak jelas kredibilitasnya.

    ATURAN AGENTIK SANGAT KETAT (INI PENEGAKAN TEKNIS, BUKAN SEKADAR SARAN -- laporan yang melanggar ini
    akan DITOLAK OTOMATIS oleh sistem dan kamu akan diminta ulang):
    - Alat `web_search` mengembalikan STRING panjang (bukan list), berisi daftar judul, ringkasan, dan URL yang sudah rapi per baris.
    - Untuk topik global/makro, gunakan query Bahasa Inggris agar hasil pencarian lebih relevan (mesin pencari lebih kaya hasil untuk Bahasa Inggris).
      Untuk topik Indonesia (poin 4), gunakan query Bahasa Indonesia.
    - `web_search` HANYA memberi judul & cuplikan singkat — itu TIDAK CUKUP buat jadi bahan laporan. Untuk SETIAP dari
      4 topik, kamu WAJIB minimal 1x `visit_webpage(url)` ke artikel yang relevan untuk membaca isi lengkapnya SEBELUM
      menulis bagian itu di laporan. JANGAN PERNAH memanggil `final_answer` di step pertama atau tanpa visit_webpage
      sama sekali -- sistem TAHU dan akan mendeteksi ini secara teknis dari riwayat eksekusi kodemu, bukan cuma
      percaya kalimat di laporanmu.
    - Pastikan URL yang kamu kunjungi TIDAK memiliki spasi (contoh salah: 'bbc. com/sport', contoh benar: 'bbc.com/sport').
    - Ekstrak DATA VALID, ANGKA SPESIFIK, dan FAKTA NYATA dari dalam artikel. Jangan berikan kesimpulan kosong tanpa data penjelas!
      JANGAN PERNAH mengarang angka/skor/kutipan yang kedengaran masuk akal tapi nggak beneran ada di artikel yang kamu baca.
    - DILARANG KERAS menulis kalimat generik/klise yang bisa ditulis tanpa baca berita sama sekali, contoh kalimat
      TERLARANG: "banyak laga-laga terpopuler dunia yang menayangkan tim terkenal", "pasar bergerak dinamis",
      "teknologi terus berkembang pesat". Setiap kalimat WAJIB mengandung fakta konkret: nama tim/orang/perusahaan
      spesifik, tanggal/jam, skor/hasil, angka/persentase/nominal, atau kutipan fakta langsung dari artikel.
    - Khusus topik olahraga (global maupun Indonesia): WAJIB sebutkan pertandingan/hasil KONKRET — nama kedua tim,
      skor atau jadwal (tanggal+jam) pertandingannya, bukan cuma "banyak pertandingan seru hari ini".
    - Jika 2 pencarian berturut-turut tidak menemukan hasil relevan, JANGAN cari terus, langsung lanjut menulis laporan dengan data yang sudah ada.
    - Tulis laporan akhir secara MENDALAM dan RINCI per topik — sertakan ANGKA SPESIFIK, PERSENTASE, NILAI NOMINAL,
      dan konteks/latar belakang yang jelas untuk tiap poin (bukan cuma kesimpulan umum tanpa data pendukung).
      Panjang laporan TIDAK dibatasi; laporan panjang akan otomatis dipecah jadi beberapa pesan Telegram, jadi jangan
      memotong analisis demi keringkasan.
    - Wajib sertakan URL sumber referensi asli yang valid di SETIAP poin/topik (bukan cuma sekali di akhir), agar user
      bisa memverifikasi tiap klaim ke sumber aslinya masing-masing.
    - SEBELUM memanggil `final_answer`, cek ulang draftmu sendiri: apakah SETIAP dari 4 topik (1) sudah dikunjungi
      minimal 1 URL via visit_webpage, (2) punya minimal 1 angka/tanggal/nama spesifik, (3) punya minimal 1 URL sumber
      tercantum? Kalau ada topik yang belum memenuhi 3 syarat itu, cari & baca lagi sebelum menulis final_answer.
    - Gunakan bahasa Indonesia santai (campur sedikit bahasa Inggris gaul layaknya teman diskusi yang sangat pintar).
    """

    MIN_VISIT_WEBPAGE_SUKSES = 4  # minimal 1x per topik (sekarang ada 4 topik)
    MAX_PERCOBAAN = 2  # dibatasi biar nggak muter-muter terus kalau providernya kesulitan riset

    agent = buat_agent()

    try:
        hasil = None
        jumlah_visit = 0

        for percobaan in range(1, MAX_PERCOBAAN + 1):
            log.info(f"Menjalankan agent (percobaan {percobaan}/{MAX_PERCOBAAN})...")
            hasil = agent.run(tugas)
            jumlah_visit = hitung_visit_webpage_sukses(agent)
            log.info(f"Validasi teknis: {jumlah_visit}x visit_webpage sukses terdeteksi (minimal {MIN_VISIT_WEBPAGE_SUKSES}).")

            if jumlah_visit >= MIN_VISIT_WEBPAGE_SUKSES:
                log.info("Validasi LULUS -- laporan terbukti berbasis riset asli.")
                break
            else:
                log.warning(
                    f"Percobaan {percobaan}: laporan DICURIGAI tanpa riset yang cukup "
                    f"(cuma {jumlah_visit}x visit_webpage sukses). "
                    + ("Mencoba ulang dari awal..." if percobaan < MAX_PERCOBAAN else "Sudah percobaan terakhir.")
                )

        # Kalau setelah semua percobaan tetap nggak lolos validasi, JANGAN diam-diam
        # kirim seolah-olah valid -- kasih peringatan jelas di depan laporan biar user
        # tahu ini perlu diverifikasi ulang, bukan langsung dipercaya.
        if jumlah_visit < MIN_VISIT_WEBPAGE_SUKSES:
            # User secara eksplisit minta: JANGAN kirim info yang belum diverifikasi riset,
            # walau dikasih label peringatan sekalipun -- mending nggak kirim laporan substantif
            # sama sekali daripada berisiko ngirim yang separuh ngarang.
            log.error(
                f"Laporan GAGAL validasi riset setelah {MAX_PERCOBAAN}x percobaan "
                f"(cuma {jumlah_visit}x visit_webpage sukses, minimal {MIN_VISIT_WEBPAGE_SUKSES}). "
                f"Laporan TIDAK dikirim ke Telegram."
            )
            log.info("Isi laporan yang GAGAL validasi (buat debugging, TIDAK dikirim ke Telegram):")
            log.info(hasil)

            pesan_gagal = (
                f"⚠️ Analisa hari ini GAGAL memenuhi standar riset minimal "
                f"({jumlah_visit}x kunjungi halaman dari minimal {MIN_VISIT_WEBPAGE_SUKSES}x yang dibutuhkan "
                f"untuk 4 topik), setelah {MAX_PERCOBAAN}x percobaan.\n\n"
                f"Laporan TIDAK dikirim karena berisiko berisi informasi yang belum terverifikasi. "
                f"Sistem akan coba lagi di jadwal berikutnya."
            )
            kirim_ke_telegram(pesan_gagal)
            log.info("Selesai satu siklus analisa -- GAGAL validasi, laporan tidak dikirim.")
            return

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
#    Penjadwalan tiap 12 jam dihandle oleh GitHub Actions cron.
# =========================================================
if __name__ == "__main__":
    try:
        jalankan_analisa_harian()
    except Exception as e:
        log.error(f"Error fatal yang tidak tertangani: {e}", exc_info=True)
        raise
