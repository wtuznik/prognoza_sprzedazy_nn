# Prognozowanie sprzedaży z wykorzystaniem sieci neuronowej

Projekt stworzony w ramach studiów. Celem projektu jest prognozowanie sprzedaży oraz analiza stanów magazynowych na podstawie danych z hurtowni przy użyciu sieci neuronowej, z wizualizacją w Power BI.

## Opis projektu

Model wykorzystuje sieć neuronową do przewidywania rzeczywistej sprzedaży produktów na podstawie dostępnych cech wejściowych. Dane są wygenerowane na potrzeby wyszkolenia i przetestowania modelu.

Pipeline projektu obejmuje:
1. Wczytanie i przetworzenie danych wejściowych
2. Podział danych na zbiór treningowy i testowy
3. Skalowanie cech (MinMaxScaler)
4. Zbudowanie i wytrenowanie modelu sieci neuronowej
5. Ewaluację modelu na zbiorze testowym
6. Wygenerowanie prognoz dla pełnego zbioru danych
7. Eksport wyników do pliku CSV
8. Wizualizację wyników w Power BI

## Technologie

- **Python**
- **Pandas / NumPy** - przetwarzanie i manipulacja danymi
- **Scikit-learn** - podział danych oraz skalowanie cech
- **TensorFlow / Keras** - budowa i trenowanie sieci neuronowej
- **Power BI** - wizualizacja wyników prognozy

## Dashboard Power BI

Dashboard wizualizuje wyniki prognozy oraz wspiera zarządzanie zapasami:
- Porównanie rzeczywistej sprzedaży z prognozą AI w czasie oraz w podziale na kategorie produktów
- Rejestr ryzyka przeterminowania stanów magazynowych (produkty z krótkim terminem przydatności)
- Analiza braków i nadwyżek magazynowych na podstawie prognozy popytu

## Architektura modelu

Sieć neuronowa typu Sequential składająca się z:
- Warstwy wejściowej dopasowanej do liczby cech
- Dwóch warstw ukrytych (64 i 32 neurony, aktywacja ReLU)
- Warstwy wyjściowej (1 neuron, regresja)

Model trenowany z wykorzystaniem optymalizatora Adam, funkcji straty MSE oraz metryki MAE.

## Uruchomienie

```bash
pip install -r requirements.txt
python model.py
```

## Wynik

Model generuje prognozę sprzedaży dla każdego produktu, którą następnie eksportuje się do pliku CSV i wizualizuje w dashboardzie Power BI.
