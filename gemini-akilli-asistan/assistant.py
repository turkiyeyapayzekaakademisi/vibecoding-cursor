"""Ollama ile terminal tabanlı tarih ve kültür asistanı."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from ollama import Client, ResponseError
from pydantic import BaseModel, Field, ValidationError


DEFAULT_MODEL = "qwen3:4b"
DEFAULT_HOST = "http://localhost:11434"
ENV_PATH = Path(__file__).resolve().parent / ".env"
EXIT_COMMANDS = {"cikis", "çıkış", "exit", "quit", "q"}
CLEAR_COMMANDS = {"temizle", "clear"}
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_BLOCK = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class Oneri(BaseModel):
    baslik: str = Field(description="Önerilen eser, yer, konu veya kaynağın adı")
    tur: Literal["kitap", "muze", "belgesel", "birincil_kaynak", "konu", "yer"] = Field(
        description="Önerinin türü"
    )
    neden: str = Field(description="Bu önerinin soruyla neden ilgili olduğu")
    donem_veya_yer: str = Field(description="Önerinin ilgili dönemi veya coğrafyası")


class TarihKulturYaniti(BaseModel):
    ozet: str = Field(description="1-2 cümlelik net özet")
    donem: str = Field(description="Tarihsel dönem veya yaklaşık tarih aralığı")
    yer: str = Field(description="Coğrafi veya kültürel bağlam")
    kisiler: list[str] = Field(description="İlgili kişi, topluluk veya kurumlar")
    baglam: str = Field(description="Ne olduğu, kimlerin rol aldığı ve neden önemli olduğu")
    kulturel_anlam: str = Field(description="Günlük yaşam, sanat, inanç veya toplumsal etki")
    belirsizlikler: list[str] = Field(
        description="Tartışmalı, efsanevi veya kesin olmayan noktalar"
    )
    oneriler: list[Oneri] = Field(
        min_length=2,
        max_length=4,
        description="İleri okuma, ziyaret veya araştırma için 2-4 somut öneri",
    )


SYSTEM_INSTRUCTION = """
Sen Tarih ve Kültür Asistanısın. Amacın olayları, kişileri, dönemleri ve kültürel pratikleri
doğru, bağlama oturtulmuş ve anlaşılır şekilde anlatmak.

Her yanıtı verilen JSON şemasına birebir uyarak üret. Serbest metin, markdown veya
şema dışı alan kullanma. Düşünce etiketi veya açıklama yazma.

Kapsam:
- Dünya ve Türkiye tarihi: dönemler, devletler, savaşlar, reformlar, günlük yaşam
- Kültür: sanat, edebiyat, din, gelenek, yemek, mimari, müzik, dil, toplumsal yapı
- Karşılaştırma: farklı toplumlar, dönemler ve kültürler arasındaki benzerlik/fark
- Kavramlar: tarihsel terimler, ekoller, kültürel kurumlar

Alan kuralları:
- ozet: 1-2 cümlelik net giriş.
- donem ve yer: mümkün olduğunca somut yaz. Yaklaşık tarihleri "yaklaşık" diye işaretle.
- kisiler: boş bırakma; ilgili figür yoksa dönemin toplumsal aktörünü yaz.
- baglam: ne, kim, neden önemli.
- kulturel_anlam: günlük yaşam, sanat, inanç veya toplumsal etki.
- belirsizlikler: mit, efsane, tartışmalı yorum veya emin olunmayan noktayı ayır.
  Yoksa boş dizi döndür.
- oneriler: her zaman 2-4 somut öneri ver. Her öneride baslik, tur, neden,
  donem_veya_yer doldur. Uydurma kitap veya müze adı verme; emin değilsen
  türü "konu" yap ve araştırılacak başlığı yaz.

Doğruluk kuralları:
- Emin olmadığın bilgiyi uydurma.
- Tek bir anlatıyı mutlak doğru gibi sunma.
- Mit ile tarihsel kaydı ayır.
- Modern siyaset veya ideolojik taraf tutma.
- Kültürleri hiyerarşik yargılama.

Üslup:
- Meraklı, sakin ve öğretici ol.
- Gereksiz jargondan kaçın; terim kullanırsan hemen açıkla.
- Kullanıcı hangi dilde yazarsa o dilde yanıtla.
""".strip()


def load_settings() -> tuple[str, str]:
    load_dotenv(ENV_PATH)
    host = os.getenv("OLLAMA_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return host, model


def new_history() -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_INSTRUCTION}]


def response_schema() -> dict[str, Any]:
    return TarihKulturYaniti.model_json_schema()


def installed_model_names(client: Client) -> set[str]:
    listing = client.list()
    models = getattr(listing, "models", None) or listing.get("models", [])
    names: set[str] = set()
    for item in models:
        name = getattr(item, "model", None) or getattr(item, "name", None)
        if name is None and isinstance(item, dict):
            name = item.get("model") or item.get("name")
        if name:
            names.add(name)
            names.add(name.split(":")[0])
    return names


def ensure_ready(client: Client, model: str) -> bool:
    try:
        names = installed_model_names(client)
    except Exception:
        print("Ollama'ya bağlanılamadı.")
        print("1. Ollama'nın çalıştığından emin ol")
        print("2. Gerekirse `ollama serve` çalıştır")
        return False

    short_name = model.split(":")[0]
    if model not in names and short_name not in names:
        print(f"{model} yüklü değil.")
        print(f"Şunu çalıştır: ollama pull {model}")
        return False
    return True


def parse_response(text: str) -> TarihKulturYaniti:
    cleaned = THINK_BLOCK.sub("", text).strip()
    cleaned = FENCE_BLOCK.sub("", cleaned).strip()
    return TarihKulturYaniti.model_validate_json(cleaned)


def print_answer(answer: TarihKulturYaniti) -> None:
    print("Asistan:")
    print(answer.model_dump_json(indent=2, ensure_ascii=False))
    print()


def generate_answer(
    client: Client, model: str, history: list[dict[str, str]], prompt: str
) -> TarihKulturYaniti:
    history.append({"role": "user", "content": prompt})
    try:
        response = client.chat(
            model=model,
            messages=history,
            format=response_schema(),
            think=False,
            options={"temperature": 0.2},
        )
        raw = response.message.content or ""
        answer = parse_response(raw)
    except Exception:
        if history and history[-1]["role"] == "user":
            history.pop()
        raise

    history.append({"role": "assistant", "content": answer.model_dump_json(ensure_ascii=False)})
    return answer


def ask(client: Client, model: str, history: list[dict[str, str]], prompt: str) -> None:
    print_answer(generate_answer(client, model, history, prompt))


def print_help() -> None:
    print("Komutlar:")
    print("  çıkış / exit / quit  → programı kapat")
    print("  temizle / clear      → sohbet geçmişini sıfırla")
    print()


def main() -> int:
    host, model = load_settings()
    client = Client(host=host)
    if not ensure_ready(client, model):
        return 1

    history = new_history()

    print("Tarih ve Kültür Asistanı")
    print(f"Model: {model}")
    print(f"Ollama: {host}")
    print("Çıktı: structured JSON")
    print_help()

    while True:
        try:
            user_input = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşürüz.")
            return 0

        if not user_input:
            continue

        command = user_input.lower()
        if command in EXIT_COMMANDS:
            print("Görüşürüz.")
            return 0
        if command in CLEAR_COMMANDS:
            history = new_history()
            print("Sohbet geçmişi temizlendi.\n")
            continue

        try:
            ask(client, model, history, user_input)
        except ValidationError as exc:
            print(f"\nJSON şeması bozuldu: {exc}\n")
        except ResponseError as exc:
            print(f"\nOllama hatası: {exc}\n")
        except Exception as exc:
            print(f"\nHata: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
