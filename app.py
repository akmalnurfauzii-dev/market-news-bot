import os
import time
import datetime
import requests
import sys
import json
from smolagents import Tool, CodeAgent, VisitWebpageTool, OpenAIServerModel

# =========================================================
# 0. Sistem Logging Otomatis ke File (run.log)
# =========================================================
class Logger(object):
    def __init__(self, filename="run.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

# Semua print() sekarang akan muncul di console DAN tersimpan di run.log
sys.stdout = Logger("run.log")

try:
    from ddgs import DDGS  
except ImportError:
    from duckduckgo_search import DDGS  

# =========================================================
# 1. Kredensial via Environment (Konek ke GitHub Secrets)
# =========================================================
# JANGAN TULIS KEY DISINI! Ini akan otomatis ngambil dari GitHub Secrets
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

TELEGRAM_MAX_CHARS = 3800

# =========================================================
# 2. Fungsi Kirim Telegram
# =========================================================
def _kirim_single_message(pesan, parse_mode=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": pesan}
    if parse_mode:
        data["parse_mode"] = parse_mode
    try:
        response = requests.post(url, data=data, timeout=30)
        if response.status_code == 200:
            return True
        print(f"⚠️ Gagal kirim (parse_mode={parse_mode}, status={response.status_code}): {response.text}")
        return False
    except Exception as e:
        print(f"❌ Kesalahan koneksi Telegram: {e}")
        return False

def _split_pesan(pesan, max_len=TELEGRAM_MAX_CHARS):
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
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(pesan)
        with open(filename, "rb") as f:
            response = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": "Laporan kepanjangan"},
                files={"document": f},
                timeout=60,
            )
        if response.status_code == 200:
            print("✅ Berhasil dikirim sebagai file dokumen!")
            return True
        print(f"❌ Gagal kirim dokumen. Status: {response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Kesalahan saat kirim dokumen: {e}")
        return False
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def kirim_ke_telegram(pesan):
    print("Mengirim laporan ke Telegram...")
    if len(pesan) <= TELEGRAM_MAX_CHARS:
        if _kirim_single_message(pesan, parse_mode="Markdown"):
            print("✅ Berhasil dikirim (Format Markdown)!")
            return
        if _kirim_single_message(pesan):
            print("✅ Berhasil dikirim (Format Teks Biasa)!")
            return
        _kirim_sebagai_dokumen(pesan)
        return
    print(f"ℹ️ Pesan {len(pesan)} karakter, akan dipecah...")
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
            print(f"❌ Bagian {i}/{len(chunks)} gagal terkirim.")
        time.sleep(1) 
    if semua_sukses:
        print(f"✅ Semua {len(chunks)} bagian berhasil dikirim!")
    else:
        _kirim_sebagai_dokumen(pesan)

# =========================================================
# 3. Tool Pencarian Berita
# =========================================================
class RecentNewsSearchTool(Tool):
    name = "web_search"
    description = (
        "Cari berita TERBARU di internet (24 jam terakhir). "
        "Kembalikan STRING berisi judul, ringkasan, dan URL."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Kata kunci pencarian, gunakan Bahasa Inggris"
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
            return "Tidak ada hasil ditemukan."
        formatted = ""
        for r in results:
            formatted += f"- {r.get('title', '')}\n  {r.get('body', '')}\n  URL: {r.get('href', '')}\n\n"
        return formatted

# =========================================================
# 4. Model AI Fallback
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
                print(f"⚠️ Gagal menyiapkan provider {p['name']}: {e}")
        if not self.providers:
            raise RuntimeError("Tidak ada provider AI yang berhasil disiapkan!")
    def __getattr__(self, attr):
        return getattr(self.providers[0]["model"], attr)
    def _try_all(self, method_name, *args, **kwargs):
        last_error = None
        for entry in self.providers:
            try:
                print(f"🔄 Mencoba provider AI: {entry['name']}...")
                method = getattr(entry["model"], method_name)
                result = method(*args, **kwargs)
                print(f"✅ Berhasil pakai provider: {entry['name']}")
                return result
            except Exception as e:
                print(f"⚠️ Provider {entry['name']} gagal: {e}")
                last_error = e
                continue
        raise Exception(f"Semua provider gagal! Error: {last_error}")
    def generate(self, messages, stop_sequences=None, **kwargs):
        return self._try_all("generate", messages, stop_sequences=stop_sequences, **kwargs)
    def __call__(self, messages, stop_sequences=None, **kwargs):
        return self._try_all("__call__", messages, stop_sequences=stop_sequences, **kwargs)

print("Menyiapkan sistem AI...")
model = FallbackModel([
    {"name": "Groq", "model_id": "llama-3.3-70b-versatile", "api_base": "https://api.groq.com/openai/v1", "api_key": GROQ_API_KEY},
    {"name": "Gemini", "model_id": "gemini-2.5-flash", "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/", "api_key": GOOGLE_API_KEY},
    {"name": "OpenRouter", "model_id": "meta-llama/llama-3.3-70b-instruct:free", "api_base": "https://openrouter.ai/api/v1", "api_key": OPENROUTER_API_KEY}
])

search_tool = RecentNewsSearchTool()
visit_tool = VisitWebpageTool()
agent = CodeAgent(tools=[search_tool, visit_tool], model=model, additional_authorized_imports=["datetime", "os", "re"], max_steps=10)

# =========================================================
# 5. Fungsi Utama Analisa Harian & History Manager
# =========================================================
def jalankan_analisa_harian():
    print("\n" + "=" * 50)
    print("🕒 MEMULAI ANALISA PASAR & BERITA GLOBAL OTOMATIS...")
    print("=" * 50)
    
    tanggal_hari_ini = datetime.date.today().strftime("%d %B %Y")
    
    # Baca History
    history_file = "history.json"
    histori_topik = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                histori_topik = json.load(f)
        except:
            histori_topik = []
            
    teks_histori = "\n".join([f"- {h}" for h in histori_topik]) if histori_topik else "Belum ada histori."

    tugas = f"""
    Hari ini tanggal {tanggal_hari_ini}. HANYA gunakan berita dari 24 jam terakhir.
    
    ⚠️ DAFTAR TOPIK YANG SUDAH DIBAHAS SEBELUMNYA:
    {teks_histori}
    
    TUGASMU:
    Cari 3 topik utama: Geopolitik/Ekonomi Global, Update Olahraga Global, dan Fakta Teknologi/AI terbaru.
    JANGAN bahas topik atau berita yang persis sama dengan DAFTAR TOPIK YANG SUDAH DIBAHAS di atas. Cari berita atau sudut pandang yang benar-benar baru.
    
    ATURAN:
    - Tulis ringkas, maksimal 3000 karakter.
    - Sertakan URL asli valid.
    - Gunakan bahasa Indonesia santai.
    """

    try:
        hasil = agent.run(tugas)
        kirim_ke_telegram(hasil)
        
        # Simpan History baru (Simpan 5 run terakhir saja biar file gak kebesaran)
        ringkasan_singkat = f"Laporan {tanggal_hari_ini} telah sukses dikirim."
        histori_topik.append(ringkasan_singkat)
        if len(histori_topik) > 5:
            histori_topik.pop(0)
            
        with open(history_file, "w") as f:
            json.dump(histori_topik, f)
            
    except Exception as e:
        error_msg = f"Waduh bro, semua provider AI gagal mikir nih: {e}"
        print(error_msg)
        kirim_ke_telegram(error_msg)

# =========================================================
# 6. Eksekusi Cloud (Sekali Jalan)
# =========================================================
if __name__ == "__main__":
    jalankan_analisa_harian()
    print("\n✅ Agen selesai dieksekusi 1x. Menunggu trigger GitHub Actions berikutnya (cron/manual)...")
