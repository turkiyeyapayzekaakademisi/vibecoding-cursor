# Diabetes Tahmini

Sklearn `diabetes` veri seti ile linear regression modeli kurup tahminleri R² skoru ile değerlendirmek.

## Proje Planı

| # | Görev | Durum |
|---|---|---|
| 1 | `venv` kur ve aktif et | Tamamlandı |
| 2 | Gerekli kütüphaneleri `pip install` ile kur | Tamamlandı |
| 3 | Sklearn `diabetes` veri seti ile linear regression modeli geliştir | Tamamlandı |
| 4 | Tahminleri R² skoru ile değerlendir | Tamamlandı |

**Sonuç:** `r2_score = 0.452603`

## Ortam

- Sanal ortam: `venv/`
- Paketler: `scikit-learn`, `numpy`, `pytest` (`requirements.txt`)

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.diabetes_regression
```

## Kod Kuralları

`skill/` klasöründe iki skill var.

### Teknik skill

- Log altyapısı kullan (`src/logging_config.py`)
- Yorumları İngilizce yaz
- Dosya, fonksiyon ve değişken isimlerinde snake_case kullan

### Non-teknik skill

- Long term memory kullanma
- Bu dosyadaki proje planına sadık kal

## Test

| # | Görev | Durum |
|---|---|---|
| 1 | Unit test oluştur | Tamamlandı |
| 2 | Test sonuçlarını `.txt` olarak göster | Tamamlandı |

```powershell
python -m pytest tests/ -v
```

Sonuçlar `test_results.txt` dosyasına yazılır. 5 unit test geçti, durum: **PASSED**.
