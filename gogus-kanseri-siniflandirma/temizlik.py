"""Kayıp değer, tekrar ve aykırı değer kontrolleri. Satır silinmez."""

import pandas as pd

from eda import OUTPUT_DIR, load_data

IQR_K = 1.5


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"target", "target_name"}]


def check_missing(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    total = int(missing.sum())
    print("=== Kayıp değer ===")
    print(f"Toplam kayıp hücre: {total}")
    if total:
        print(missing[missing > 0].to_string())
    else:
        print("Kayıp değer yok.")


def check_duplicates(df: pd.DataFrame) -> None:
    cols = feature_columns(df)
    n_full = int(df.duplicated().sum())
    n_features = int(df.duplicated(subset=cols).sum())
    print("\n=== Tekrarlayan satırlar ===")
    print(f"Tam satır tekrarı: {n_full}")
    print(f"Özellik tekrarı (target hariç): {n_features}")
    if n_full == 0 and n_features == 0:
        print("Tekrarlayan satır yok.")


def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_K * iqr
    upper = q3 + IQR_K * iqr
    return (series < lower) | (series > upper)


def check_outliers(df: pd.DataFrame) -> pd.DataFrame:
    cols = feature_columns(df)
    rows = []
    any_outlier = pd.Series(False, index=df.index)

    for col in cols:
        mask = iqr_outlier_mask(df[col])
        any_outlier |= mask
        n = int(mask.sum())
        rows.append(
            {
                "feature": col,
                "n_outlier": n,
                "percent": round(100 * n / len(df), 1),
            }
        )

    report = pd.DataFrame(rows).sort_values("n_outlier", ascending=False)
    n_rows = int(any_outlier.sum())

    print("\n=== Aykırı değer (IQR, k=1.5) ===")
    print(f"En az bir aykırı değere sahip satır: {n_rows} / {len(df)}")
    print("Özellik bazında (silinmedi):")
    print(report.to_string(index=False))
    print(
        "\nNot: Tıbbi ölçümlerde uç değer gerçek olabilir. "
        "Karar ağacı aykırı değere dayanıklıdır; satır silinmedi."
    )

    report.to_csv(OUTPUT_DIR / "aykiri_deger_ozet.csv", index=False)
    print(f"\nÖzet kaydedildi: {OUTPUT_DIR / 'aykiri_deger_ozet.csv'}")
    return report


def main() -> None:
    df = load_data()
    check_missing(df)
    check_duplicates(df)
    check_outliers(df)


if __name__ == "__main__":
    main()
