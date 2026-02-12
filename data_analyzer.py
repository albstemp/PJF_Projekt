import pandas as pd
import os


class StatsAnalyzer:
    clean_csv: str
    noise_csv: str

    def __init__(self, clean_csv: str = "metadata_pro_train_clean.csv",
                 noise_csv: str = "metadata_pro_test_noise.csv") -> None:
        self.clean_csv = clean_csv
        self.noise_csv = noise_csv

    def analyze(self) -> None:
        clean_exists: bool = os.path.exists(self.clean_csv)
        noise_exists: bool = os.path.exists(self.noise_csv)

        if not clean_exists and not noise_exists:
            print("Błąd: Nie znaleziono żadnych plików metadanych do analizy.")
            return

        df_clean: pd.DataFrame = pd.read_csv(self.clean_csv) if clean_exists else pd.DataFrame(columns=['key_id'])
        df_noise: pd.DataFrame = pd.read_csv(self.noise_csv) if noise_exists else pd.DataFrame(columns=['key_id'])

        clean_counts: pd.Series = df_clean['key_id'].value_counts().rename('Czyste')
        noise_counts: pd.Series = df_noise['key_id'].value_counts().rename('Zanieczyszczone')

        final_stats: pd.DataFrame = pd.concat([clean_counts, noise_counts], axis=1).fillna(0).astype(int)
        final_stats = final_stats.sort_values(by='Czyste', ascending=False)
        final_stats = final_stats.reset_index()
        final_stats.columns = ['Znak', 'Czysty', 'Zanieczyszczone']

        print(final_stats.to_string(index=False, justify='center'))


if __name__ == "__main__":
    analyzer: StatsAnalyzer = StatsAnalyzer()
    analyzer.analyze()