# Predykcja Zapotrzebowania na Energię Elektryczną

Projekt ten ma na celu prognozowanie krótkoterminowego zapotrzebowania na energię elektryczną przy użyciu różnych, iteracyjnie ulepszanych modeli uczenia maszynowego.

## Struktura Projektu

```
projektAI/
│
├── data/
│   └── combined_data.csv
│
├── outputs/
│   └── ... (wykresy i inne pliki wyjściowe)
│
├── src/
│   ├── models/
│   │   ├── mlp_model.py
│   │   ├── committee_model.py
│   │   ├── modular_model.py
│   │   └── rule_aided_model.py
│   │
│   ├── preprocessing/
│   │   └── feature_engineering.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   └── visualization/
│       └── visualize.py
│
├── .gitignore
├── README.md
└── pyproject.toml
```

- **`data/`**: Zawiera dane wejściowe do modeli.
- **`notebooks/`**: Notatniki Jupyter do eksploracji danych.
- **`outputs/`**: Katalog na pliki wyjściowe (wykresy), z podkatalogami dla każdego modelu.
- **`src/`**: Główny katalog z kodem źródłowym.
  - **`models/`**: Implementacje różnych modeli predykcyjnych.
  - **`preprocessing/`**: Skrypty do przetwarzania wstępnego danych.
  - **`training/`**: Główny skrypt do uruchamiania treningu modeli.
  - **`visualization/`**: Funkcje do wizualizacji wyników.

## Ewolucja Modeli

Projekt przedstawia ewolucję podejścia do prognozowania, zaczynając od prostego modelu, a kończąc na bardziej złożonym i skutecznym rozwiązaniu.

1.  **MLP (Multi-layer Perceptron)**: Prosty model sieci neuronowej, który prognozuje od razu całe 24 godziny. Stanowi punkt wyjścia i bazę do porównań.
2.  **Committee Model**: Ulepszenie polegające na stworzeniu "komitetu" 24 wyspecjalizowanych modeli, gdzie każdy prognozuje obciążenie tylko dla jednej, konkretnej godziny.
3.  **Modular Model**: Wariant komitetu, w którym prognoza dla każdej godziny jest uśredniana z wyników K niezależnych modułów (sieci neuronowych), co stabilizuje predykcje.
4.  **Rule-Aided Model**: **Najbardziej zaawansowany i najskuteczniejszy model w projekcie.** Bazuje na modelu modularnym, ale dodatkowo wprowadza logikę regułową – usuwa z danych treningowych dni nietypowe (np. święta, długie weekendy), które mogłyby zaburzać proces uczenia. Dzięki temu osiąga najlepsze wyniki prognoz.

## Użycie

Głównym sposobem uruchamiania treningu jest użycie skryptu `src/training/train.py`. Należy podać jako argument nazwę modelu, który chcemy wytrenować.

```bash
# Przykład dla modelu Rule-Aided
python src/training/train.py rule_aided

# Przykład dla modelu MLP
python src/training/train.py mlp
```

Możliwe jest również uruchomienie każdego modelu indywidualnie:
```bash
python src/models/mlp_model.py
```

Wyniki działania modelu, w tym metryki MAPE i wykresy, zostaną zapisane w katalogu `outputs/` w dedykowanym podfolderze dla danego modelu.

## Wymagania

Do uruchomienia projektu potrzebne są następujące biblioteki:

- `pandas`
- `numpy`
- `tensorflow`
- `scikit-learn`
- `matplotlib`
- `holidays`

Można je zainstalować za pomocą `pip`:

```bash
pip install -r requirements.txt
```
(Zakładając istnienie pliku `requirements.txt` - można go stworzyć z `pyproject.toml`)
