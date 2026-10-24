## Genel
Plan doğru sırada: veri → EDA → temizlik → eğitim → test → hiperparametre araması.
Sklearn `load_breast_cancer` ikili sınıflandırma için uygun; karar ağacı yorumlanabilir.

## Veri seti
- 569 örnek, 30 sayısal özellik, 2 sınıf (malignant / benign)
- Kategorik feature yok
- Kayıp değer ve tekrarlayan satır yok
- Sınıf dağılımı dengesiz (~357 benign, ~212 malignant)

## EDA
- `describe()`, sınıf dağılımı, korelasyon ısı haritası
- Sınıfa göre boxplot (görsel için Min-Max)

## Temizlik
- Kayıp / tekrar kontrolü yapıldı; sonuç “yok”
- Aykırı değer IQR ile tespit edildi, silinmedi
- Karar ağacı ölçeklemeye ihtiyaç duymaz

## Eğitim / test
- `train_test_split(test_size=0.2, stratify=y, random_state=42)`
- İlk model: `DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)`
- Baseline: `DummyClassifier(most_frequent)`
- Metrikler: accuracy, confusion matrix, classification report, ROC-AUC
- Feature importance ve `plot_tree`

## Hiperparametre araması
- Arama yalnızca train üzerinde, `StratifiedKFold(5)`
- Test seti seçim için kullanılmaz
- `GridSearchCV`, scoring=`roc_auc`
- Izgara: `max_depth`, `min_samples_leaf`, `min_samples_split`, `criterion`
- En iyi seti ısı haritası ve derinlik eğrisi ile görselleştir
- En iyi ağacı holdout testte değerlendir

## Proje düzeni
- `requirements.txt`: scikit-learn, pandas, matplotlib, seaborn
- Scriptler: `eda.py`, `temizlik.py`, `split.py`, `model.py`, `hiperparametre.py`
- Sonuçları “tanı aracı” diye sunma; eğitim amaçlı

## Uygulama sırası
0. [x] requirements.txt ve kurulumları yap
1. [x] Veriyi yükle, EDA yazdır + görselleştir
2. [x] Temizlik kontrolleri
3. [x] Stratified train/test split
4. [x] Karar ağacı (sınırlı derinlik) + baseline
5. [x] Confusion matrix + classification report + ROC-AUC
6. [x] Feature importance ve ağaç görseli
7. [x] Karar ağacı hiperparametre araması (GridSearchCV) + görselleştir

## En iyi set (şu an)
- `criterion=gini`, `max_depth=4`, `min_samples_leaf=10`, `min_samples_split=2`
- CV ROC-AUC: 0.965 | Test ROC-AUC: 0.968 | Test accuracy: 0.95
- `min_samples_split` 2/5/10 aynı skor (leaf=10 iken etkisiz)
