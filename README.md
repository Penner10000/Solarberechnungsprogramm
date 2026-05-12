# PV Ertrags-Simulator

Simuliert und vergleicht den wochentlichen PV-Energieertrag verschiedener Anlagenkonfigurationen basierend auf Clear-Sky-Einstrahlung (pvlib).

## Installation

### Voraussetzungen

- Python 3.10 oder hoher

### Einrichtung

```bash
# Repository klonen oder entpacken
cd solar

# Virtuelle Umgebung erstellen (empfohlen)
python -m venv .venv

# Aktivieren:
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Abhangigkeiten installieren
pip install -r requirements.txt
```

### Starten

Mit aktivierter venv:
```bash
streamlit run app.py
```

Ohne venv (nutzt systemweites Python):
```bash
python -m streamlit run app.py
```
```

Die App offnet sich im Browser unter `http://localhost:8501`.

---

## Benutzung

### 1. Anlage hinzufugen (Sidebar links)

Im Formular "Neue Anlage" die Parameter eintragen:

| Feld | Beschreibung |
|---|---|
| Name | Freier Bezeichner |
| Nennleistung (Wp) | Modulleistung in Watt-Peak |
| Azimut | 0= Nord, 90= Ost, 180= Sud, 270= West |
| Neigung | Dachwinkel 0 (flach) bis 90 (senkrecht) |
| Breiten- / Langengrad | Koordinaten des Standorts |
| Wechselrichter-Limit | 0 = kein Limit, z.B. 800 fur Balkonkraftwerk |
| Farbe | Diagramm-Farbe der Anlage |

Gespeicherte Anlagen konnen per Klick auf \(\u270f\) bearbeitet, \(\uD83D\uDCCB\) dupliziert oder \(\uD83D\uDDD1\) geloscht werden. Die Daten bleiben in `pv_plants.json` erhalten.

### 2. Simulationsparameter (Sidebar)

**Umgebung & Modell:**
- Transpositionsmodell: Perez (empfohlen, genaueste Diffusstrahlberechnung), Haydavies, Isotrop
- Albedo: Bodenreflexion (0,20 = Gras, 0,70 = Schnee)

**Elektrische Verluste:**
- System-Verlustfaktor: Pauschalabzug fur Kabel, Verschmutzung (0 % = ideal)
- Wechselrichter-Wirkungsgrad: 96 % = realistisch, 100 % = ideal
- Temperaturkoeffizient: 0 = ideal, -0,004/K = kristalline Module

> **Hinweis:** Alle Verlustwerte sind standardmaig auf 0 % (Idealfall) gesetzt. Fur realistische Werte die dokumentierten Empfehlungen eintragen.

### 3. Vergleich (Hauptbereich)

- Im Dropdown-Menu die zu vergleichenden Anlagen auswahlen
- Diagramm-Einstellungen: Absolut (kWh) / Spezifisch (kWh/kWp), Linie / Balken gruppiert / Balken gestapelt
- Die Vergleichstabelle zeigt Jahresertrag, Maximalwoche, Clipping-Verluste
- Wirtschaftlichkeit: Strompreis und Betrachtungszeitraum einstellbar, inkl. Degradation

---

## Physikalisches Modell

1. **Clear-Sky-Einstrahlung** via `pvlib.location.Location.get_clearsky()`
2. **POA-Transposition** (Perez / Haydavies / Isotrop) mit Air-Mass und extraterrestrischer Strahlung
3. **Zelltemperatur** via SAPM-Modell mit saisonalem Lufttemperatur-Profil
4. **DC-Leistung** via PVWatts-Modell inkl. Temperaturkoeffizient
5. **AC-Wandlung** via PVWatts-Wechselrichter-Modell
6. **Systemverluste** (konfigurierbar, nach AC-Wandlung)
7. **Clipping** auf Wechselrichter-Limit
8. **Wochen-Normalisierung**: Durchschnittlicher Tagesertrag \(\times\) 7 (52 standardisierte Wochen)

---

## Projektstruktur

```
solar/
  app.py             # Hauptanwendung
  requirements.txt   # Python-Abhangigkeiten
  README.md          # Diese Datei
  pv_plants.json     # Gespeicherte Anlagen (wird automatisch erstellt)
```

---

## Lizenz

MIT
