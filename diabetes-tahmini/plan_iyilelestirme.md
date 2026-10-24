# R² İyileştirme Planı

Mevcut skor: **0.452603** (tek `train_test_split`, `LinearRegression`, `random_state=42`).

Bu değer hatalı bir eğitimden gelmiyor. Sklearn `diabetes` seti 442 satır ve 10 özellikten oluşur; hedef gürültülüdür. Düz linear regression burada genelde **0.40–0.50** bandında kalır. Hedef: **0.50+** güvenilir skor, mümkünse **0.52–0.55**. 0.80+ bu veri ile gerçekçi değildir.

## Neden düşük?

1. Model çok basit: sadece düz lineer ilişki varsayıyor.
2. `s1`–`s6` serum özellikleri birbiriyle ilişkili; katsayılar kararsızlaşıyor.
3. Tek split var. 89 test satırında R² kolay oynar.
4. Etkileşim yok. `bmi * bp` gibi kombinasyonlar hedefi tek başına `bmi` veya `bp`'den daha iyi açıklayabilir.
5. Ölçüm sadece R². RMSE / MAE bakılmadığı için hata büyüklüğü görünmüyor.

## Yapılacaklar

Sıra önemli: önce ölçümü sağlamlaştır, sonra modeli zorla.

### 1. Değerlendirmeyi düzelt

Tek split yerine **5-fold cross-validation** kullan. Aynı seed ile ortalama R² ve standart sapmayı yaz.

Ayrıca test setinde **RMSE** ve **MAE** raporla. R² tek başına yetmez.

Çıktı: `test_results.txt` içine baseline + yeni skorlar.

### 2. Regularization dene

Özellikler zaten ölçekli. `Ridge`, `Lasso` ve `ElasticNet` ile `RidgeCV` / `LassoCV` / `ElasticNetCV` çalıştır.

Multicollinearity varsa Ridge en temiz kazancı verir. Lasso zayıf özellikleri sıfırlar.

Karşılaştır: `LinearRegression` vs `Ridge` vs `Lasso` vs `ElasticNet`.

### 3. Özellik üret

`PolynomialFeatures(degree=2, include_bias=False)` ile kare ve çarpım terimleri ekle.

442 satır için degree 3 fazla; overfit eder. Degree 2 + Ridge/Lasso birlikte dene.

Elle bakılacak etkileşimler:

- `bmi * bp`
- `bmi * s5`
- `bp * s5`

### 4. Özellik seç

Tüm 10 özelliği körlemesine kullanma.

- Korelasyon / `f_regression` ile zayıf özellikleri ele
- Lasso'nun sıfırladığı özellikleri düş
- `RFECV` ile otomatik seçim yap

Az ve anlamlı özellik, bu sette çoğu zaman daha iyi geneller.

### 5. Daha güçlü ama hâlâ küçük modeller

Veri küçük. Ağır modeller gerekmez. Sırayla:

1. `Ridge` + polynomial features
2. `SVR(kernel="rbf")` (C ve gamma için küçük grid)
3. `HistGradientBoostingRegressor` veya `RandomForestRegressor` (max_depth düşük tut)

Hangi model CV R²'yi yükseltirse onu seç. Tek test skoru ile karar verme.

### 6. Aykırı değer ve hedef kontrolü

Hedef dağılımına bak. Aşırı uçları:

- IQR ile işaretle
- veya `HuberRegressor` dene

Birkaç uç nokta 89 satırlık testte R²'yi ciddi düşürebilir.

## Deney sonuçları

Karar kuralı: en iyi model **CV R²** ile seçildi, tek test skoru ile değil.

**Kazanan:** `manual_interactions_ridge` (`bmi*bp`, `bmi*s5`, `bp*s5` + RidgeCV)

| Metrik | Baseline | Kazanan |
|---|---|---|
| CV R² | 0.4804 ± 0.0409 | **0.4820 ± 0.0438** |
| Test R² | 0.452603 | **0.472049** |
| Test RMSE | 53.85 | **52.89** |
| Test MAE | 42.79 | **42.02** |
| Train–test gap | 0.0753 | **0.0624** |

En yüksek tek test R² `poly2_lasso` (0.5135) ama CV R²'si 0.4638; holdout şansı olabilir. SVR test R² 0.5080, fakat gap 0.115 (overfit).

Kabul: test R² ve gap geçti. CV R² 0.50 eşiğine kimse çıkmadı; bu setin tavanına yakın.

Tüm tablo `test_results.txt` içinde.

## Uygulama sırası

| Adım | İş | Gerçek etki | Durum |
|---|---|---|---|
| 1 | 5-fold CV + RMSE/MAE | Baseline CV R² 0.4804; test 0.4526 | Tamamlandı |
| 2 | Ridge / Lasso / ElasticNet CV | Lasso test R² 0.4612 | Tamamlandı |
| 3 | Degree 2 + Ridge/Lasso | poly2_lasso test 0.5135, CV 0.4638 | Tamamlandı |
| 4 | Feature selection | RFECV test 0.4714; k=6 zararlı | Tamamlandı |
| 5 | SVR / küçük tree | SVR test 0.5080 ama overfit | Tamamlandı |
| 6 | Outlier / Huber | IQR 0 satır sildi; Huber 0.4500 | Tamamlandı |

Her adımdan sonra aynı CV protokolü ile karşılaştır. Baseline **0.452603** altına düşen değişikliği alma.

## Kodda değişecek yerler

- `src/diabetes_regression.py`: model seçimi, CV, ek metrikler
- `tests/test_diabetes_regression.py`: yeni pipeline için unit test
- `test_results.txt`: model adı, CV R², test R², RMSE, MAE
- Loglar aynı altyapıda kalacak (`src/logging_config.py`)

Kurallar değişmez: İngilizce yorum, snake_case, print yok.

## Kabul kriteri

İyileştirme başarılı sayılır:

- CV ortalama R² **≥ 0.50**
- Test R², baseline **0.452603** üstünde
- Train R² ile test R² farkı **0.10** altında (overfit yok)
- Sonuçlar `test_results.txt` içinde karşılaştırılabilir

## Yapılmayacaklar

- Derin öğrenme (veri yetmez)
- Degree 3+ polynomial
- Yeni dış veri seti
- Rastgele hiperparametre denemesi; her deneme loglanacak
