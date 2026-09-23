# Dynasearch – eine Suche, die mehrere unabhängige Züge auf einmal kombiniert – Streamlit-Demo

Achtes Stück der **Trajektorien-Metaheuristiken-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", zweites Stück des **Nachbarschafts-Zweigs**:
dieselbe Rundtour wie in der [hill-climbing-demo](../hill-climbing-demo), der [simulated-annealing-demo](../simulated-annealing-demo), der [iterated-local-search-demo](../iterated-local-search-demo), der [variable-neighborhood-search-demo](../variable-neighborhood-search-demo), der [tabu-search-demo](../tabu-search-demo), der [grasp-demo](../grasp-demo) und der [lin-kernighan-demo](../lin-kernighan-demo) (ein Depot, n Kundenstopps in einem 100 × 100-km-Gebiet), dieselbe untere Schranke.

**Einordnung in die Reihe:** **Dynasearch** (Potts & van de Velde 1995) ist wie Lin-Kernighan ein Kind der Wurzel Hill Climbing im **Nachbarschafts-Zweig** - kein Kind der ILS/VNS/Tabu/GRASP-Kette. Ein normaler 2-opt-Abstieg (beste Verbesserung, voller Rescan) wendet pro Iteration GENAU EINEN verbessernden Zug an, dann wird die komplette Nachbarschaft neu bewertet. Dynasearch sucht stattdessen die gewinn-maximale MENGE paarweise UNABHÄNGIGER (nicht überlappender) verbessernder 2-opt-Züge und wendet sie ALLE GLEICHZEITIG an - eine Auswahl aus exponentiell vielen möglichen Kombinationen, aber exakt gelöst durch dynamische Programmierung in O(n²) (klassisches gewichtetes Intervall-Scheduling), nicht durch Ausprobieren.
```
hill-climbing-demo (Wurzel: nur bergab, bleibt im ersten Optimum stecken)        [gebaut]
  ├─ simulated-annealing-demo (nimmt Verschlechterungen an, Abkühlplan)          [gebaut]
  ├─ iterated-local-search-demo (stört ein gutes Optimum mit fester Störstärke)  [gebaut]
  │     └─ variable-neighborhood-search-demo (Störstärke eskaliert + Reset)     [gebaut]
  ├─ tabu-search-demo (immer der beste Zug, Gedächtnis gegen Rückwege)          [gebaut]
  ├─ grasp-demo (randomisierte Konstruktion, viele Neuanfänge)                  [gebaut]
  └─ Nachbarschafts-Zweig
        ├─ lin-kernighan-demo (variable Tiefe statt fixer 2-opt-Nachbarschaft)  [gebaut]
        └─ dynasearch-demo (viele unabhängige Züge auf einmal statt einer)      [dieses Stück]
              └─ VRP-Nachbarschaften (inter-route-Züge)                        [nicht gebaut]
```

Ergebnis in Kürze: **Dynasearch braucht bei EINEM Abstieg immer deutlich weniger Bewertungen bis zur Konvergenz als sequentielles bestes-Verbesserung-2-opt (2.0x bei 20 Stopps, wachsend auf 3.9x bei 200 Stopps) und gewinnt den restart-basierten Budget-Vergleich bei JEDEM gemessenen Budget (10 Tausend bis 2 Millionen) klar** - die durchgehendste, am wenigsten von Umschlagpunkten unterbrochene Budget-Geschichte der ganzen Linie. Aber ein ehrlicher, nicht-monotoner Zweitbefund: die GÜTE des von einem einzelnen Abstieg erreichten Lokaloptimums dreht sich mit der Instanzgröße um - bei kleinen/mittleren Instanzen (bis 60 Stopps) findet Dynasearch das bessere Optimum, ab etwa 80 Stopps ist sequentiell leicht genauer. Bei FESTEM Budget (200 Tausend) schlägt das durch: Dynasearch gewinnt klar bis 60 Stopps, verliert aber ab 100 Stopps knapp - bei so wenig Budget pro Stopp passt für beide kaum ein Neustart, und dann zählt nur noch die (bei großen Instanzen leicht schlechtere) Qualität eines einzelnen Dynasearch-Abstiegs.

| Frage | Ergebnis (60 gleichverteilte Stopps, 200 Tausend Vorschläge, sofern nicht anders angegeben; Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds; Abstand = Prozent über der 1-Baum-Schranke) |
|---|---|
| Standardfall | ✅ Dynasearch **3.75 %** über der Schranke gegen **6.33 %** für sequentiell (gleiches Budget) - klarer Sieg |
| **Budget-Sweep (Dynasearch gegen sequentiell)** | ✅ 10T-50T: **5.81/7.85 %** (Dynasearch konvergiert, sequentiell noch nicht). 100T: **4.67/7.85 %**. 500T: **2.69/3.98 %**. 1M: **1.93/2.58 %**. 2M: **1.25/1.91 %** - Dynasearch gewinnt bei JEDEM gemessenen Budget klar |
| **Ein einzelner Abstieg: Bewertungen bis Konvergenz** | ✅ n=20: **2961/3301** (1.1x). n=60: **50122/101776** (2.0x). n=100: **182123/517665** (2.8x). n=200: **1126283/4438804** (3.9x) - Dynasearch IMMER weniger, der Faktor wächst mit n |
| **Ein einzelner Abstieg: erreichte Güte** | ⚠️ n=20: **0.60/1.85 %** (Dynasearch besser). n=60: **5.81/7.85 %** (Dynasearch besser). n=100: **9.74/8.78 %** (sequentiell besser). n=200: **10.03/9.56 %** (sequentiell besser) - dreht sich ab ~80 Stopps |
| **Skalierung bei festem Budget (200 Tausend)** | ⚠️ n=20/40/60: Dynasearch gewinnt klar (0.05/0.05, 1.02/1.04, 3.75/6.33 %). AB n=100: sequentiell gewinnt (9.74/8.78, 10.10/9.49, **10.03/9.56 %** bei n=200) |

## Was die Demo zeigt

1. **Dynasearch in Aktion** (Schritt-Slider + Abspielen): **Instanz** → **Abstieg** (Iterations-Regler + ▶️ Abstieg abspielen: Länge der Tour über die Rescans, die Tour nach der gewählten Iteration mit hervorgehobenen geänderten Knoten - zeigt bei Dynasearch mehrere gleichzeitig geänderte, weit auseinanderliegende Stellen statt einer, dazu ein Balkendiagramm der je Iteration kombinierten Züge) → **Ergebnis** (beste Tour im gewählten Modus neben demselben Vergleich im jeweils anderen Modus, gleiches Budget).
2. **Was die Suche gefunden hat:** Abstand zur Schranke im gewählten Modus, im jeweils anderen Modus (gleiches Budget) und für einen einzelnen Abstieg; Urteil (`beats_hc` → `comparable` → `hc_wins`, relativ zum gewählten Modus formuliert), Detailtabellen.
3. **📐 Sweeps** über Budget, Stopps und Gruppen (feste Instanzen ab 100000, drei Ketten je Instanz), gewählter Modus gegen den jeweils anderen.
4. **🔬 Experimente auf Abruf:** **ein Abstieg: Effizienz** (Bewertungen bis Konvergenz UND die Güte-Umkehr mit der Instanzgröße - der zentrale ehrliche Befund); Budget von 10 Tausend bis 2 Millionen; Streuung über 20 Ketten; Skalierung von 20 bis 200 Stopps bei festem Budget.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (Instanzgröße, genug Budget für mehr als einen Neustart, Depot-Ausnahme in der DP, keine Or-opt-Kombination).

Regler: Stopps (10–200), Anteil der Stopps in Gruppen, **Modus** (Dynasearch / Sequentiell - der direkte Umschalter zwischen den zwei Mechanismen), **Budget** (10 Tausend bis 2 Millionen bewertete Nachbarn, voller Rescan wie Tabu Search/GRASP), Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲, steuert nur die zufällige Startlösung - beide Verfahren sind sonst deterministisch, kein Zufall im Kern).
Keine Nachbarschaftswahl (fest 2-opt, wie die Literatur - Or-opt-Kompatibilität wäre eine eigenständige Herleitung, bewusst nicht umgesetzt, siehe Grenzen). Kein Kandidatenlisten-Vergleich (dieselbe Begründung wie Tabu Search/GRASP: Dynasearch ist nicht Don't-Look-Bits-nativ).

## Messwerte der Presets (Instanz-Seed 35, Ketten-Seed 0; sie prüfen sich mit Urteil-Bändern selbst)

Die einzelne Standardinstanz liegt hier bei 2.16 % gegen 4.21 % (Dynasearch gegen sequentiell) - näher am Optimum als der Mittelwert über fünf Instanzen (3.75 % gegen 6.33 %); die Mittelwerte oben in der Tabelle sind die belastbaren Zahlen.

| Preset | Urteil (Band über Instanzen × Ketten) |
|---|---|
| Standardfall (Voreinstellung) | beats_hc / comparable / hc_wins |
| Sequentiell statt Dynasearch | beats_hc / comparable / hc_wins |
| Zu kleines Budget (10 Tausend) | beats_hc / comparable / hc_wins |
| Großes Budget (1 Million) | beats_hc / comparable / hc_wins |
| Große Instanz (200 Stopps) | beats_hc / comparable / hc_wins |

Die "Große Instanz"-Voreinstellung ist bewusst ein **Negativbefund-Preset**: bei 200 Stopps und 200 Tausend Vorschlägen liegt sequentiell vorn (9.56 % gegen 10.03 % für Dynasearch).

## Modell und Verfahren

- **Instanz, Nachbarschaften, Abstieg, Schranke** (`dyna_scenario.py`, `dyna_tour.py`): wortgleiche Kopien aus der [hill-climbing-demo](../hill-climbing-demo). Die sequentielle Vergleichsgröße ist direkt `dyna_tour.descend(D, tour, "2opt", "best", max_evaluations=budget)` - kein neuer Code, nur Wiederverwendung.
- **Dynasearch-Schritt** (`dyna_algorithm.dynasearch_step`): sammelt alle gültigen, verbessernden 2-opt-Kandidaten aus derselben vektorisierten Delta-Matrix wie der sequentielle Abstieg (`dyna_tour._delta_2opt`, kein eigenständig hergeleitetes Delta-Risiko), löst das gewichtete Intervall-Scheduling-Problem (maximale Gesamtsumme paarweise unabhängiger Züge) mit einer einzigen Vorwärts-DP über die Tour-Positionen, wendet die gewählte Menge gleichzeitig an. Zwei Züge (i₁,j₁), (i₂,j₂) mit i₁<i₂ sind unabhängig genau dann, wenn i₂ ≥ j₁+2 (ihr voller Fußabdruck [i,j+1] - vier referenzierte Positionen, nicht nur das umgekehrte Stück - überlappt sich nicht). Wie im Original-Paper ("moves that do not involve city 1") werden Züge ausgeschlossen, die Position 0 (das Depot) referenzieren.
- **Dynasearch-Abstieg** (`dyna_algorithm.dynasearch_descend`): wiederholt den Schritt bis zur Konvergenz oder zum Bewertungsbudget, dann EIN abschließender sequentieller 2-opt-Aufräum-Durchgang (volle Nachbarschaft inkl. Depot) - da die DP depot-berührende Züge ausschließt, garantiert die DP-Konvergenz allein nur ein 2-opt-Lokaloptimum UNTER dieser Einschränkung; der Aufräum-Durchgang schließt die Lücke zu einem echten 2-opt-Lokaloptimum.
- **Modus-Umschalter**: ein direkter Vergleich zweier Mechanismen bei sonst identischer Instanz/Budget/Startlösung (Vorbild: ILS' "Lokale Suche"-Umschalter DLB/voller Rescan).
- **Unabhängige Neustarts** (`dyna_evaluation.restarts`): derselbe Abstieg im gewählten Modus, aber mit Neustarts statt Kicks - der erste Neustart läuft immer bis zur Konvergenz.
- **Auswertung** (`dyna_evaluation.py`): Kennzahlen, Urteil, Sweeps über feste Instanzen × Ketten, Effizienz-Vergleich eines einzelnen Abstiegs, Streuung, Skalierung.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "Dynasearch braucht weniger Bewertungen bis zur Konvergenz als sequentielles bestes-Verbesserung-2-opt"** – **bestätigt, klar und ausnahmslos**: bei JEDER gemessenen Instanzgröße (20 bis 200 Stopps) braucht ein einzelner Dynasearch-Abstieg deutlich weniger Bewertungen bis zur Konvergenz als ein sequentieller (1.1x bis 3.9x, wachsend mit n) - erklärt durch die mittlere Zahl gleichzeitig kombinierter Züge je Rescan (2.0 bei n=20, 4.0 bei n=60, 8.9 bei n=200: je größer die Instanz, desto mehr unabhängige Verbesserungen passen gleichzeitig hinein). Übersetzt sich das auch in eine bessere beste Tour bei Neustarts und festem Budget? **Ja, bei JEDEM gemessenen Budget** (10 Tausend bis 2 Millionen) - die klarste, am wenigsten von Umschlagpunkten unterbrochene Budget-Geschichte der ganzen Linie.
- **Aber: die GÜTE eines einzelnen Abstiegs dreht sich mit der Instanzgröße um** (ehrlicher, nicht-monotoner Zweitbefund, NICHT vorab erwartet). Bei n≤60 findet Dynasearch das bessere 2-opt-Lokaloptimum (0.60 % gegen 1.85 % bei n=20); ab n≈80 ist sequentiell leicht genauer (10.03 % gegen 9.56 % bei n=200) - Dynasearchs "gierige Kombination möglichst vieler Züge auf einmal" ist bei großen Instanzen offenbar eine leicht schlechtere Suchreihenfolge als "immer nur den einen besten Zug, aber öfter". Das erklärt auch, warum sich der Neustart-basierte Budget-Vorteil bei FESTEM Budget und großen, budgetknappen Instanzen umkehrt: bei so wenig Budget pro Stopp passt für beide kaum mehr als ein Neustart, und dann zählt nur noch die (bei großem n leicht schlechtere) Güte des einzelnen Abstiegs, nicht mehr die Effizienz-Ersparnis von Dynasearch.
- **Depot bewusst ausgenommen** (wie im Original-Paper): die DP schließt Züge, die Position 0 betreffen, komplett aus - ohne den anschließenden sequentiellen Aufräum-Durchgang wäre die Konvergenz-Tour KEIN echtes 2-opt-Lokaloptimum (empirisch verifiziert: 59 von 60 "nicht konvergierten" Prüf-Läufen hatten genau EINEN übrig gebliebenen, depot-berührenden verbessernden Zug). Der Aufräum-Durchgang behebt das vollständig (80 von 80 Läufen danach ein echtes Lokaloptimum, gegen `is_local_optimum` geprüft, unabhängig von der eigenen DP-Logik).
- **Nur 2-opt-Züge, keine Or-opt-Kombination.** Die Unabhängigkeits-DP ist nur für 2-opt-Segmentumkehrungen hergeleitet (der Fußabdruck-Beweis nutzt explizit, dass ein 2-opt-Zug genau vier Positionen referenziert); eine Erweiterung auf Or-opt-Züge bräuchte eine eigene Kompatibilitäts-Definition - bewusst nicht umgesetzt.
- **Synthetische Instanzen:** euklidisch, gleichverteilt oder in fünf Gruppen, ein Fahrzeug, keine Kapazitäten oder Zeitfenster. Zeiten hängen vom Rechner und der Python-Version ab (die Tests prüfen nur Größenordnungen).

## Verifikation

- **Zentrale Korrektheit der DP:** das Ergebnis von `dynasearch_step` gegen eine unabhängige Brute-Force-Suche über alle Teilmengen der Kandidaten (25 parametrisierte Zufallsinstanzen, dieselbe i₂≥j₁+2-Unabhängigkeitsregel, aber nicht dieselbe DP-Logik) - deckte zwei echte Fehler auf: eine zu lockere Unabhängigkeitsbedingung (nur das umgekehrte Stück statt des vollen Vier-Positionen-Fußabdrucks) und eine Off-by-one-Gruppierung in der DP-Rückverfolgung.
- **Additivität der Deltas:** die angewandte Gesamtlängenänderung einer kombinierten Menge entspricht empirisch exakt der Summe der Einzeldeltas (30 Zufallsläufe) - die Eigenschaft, die die gesamte DP-Konstruktion voraussetzt.
- **Konvergenz-ist-2opt-Optimum:** `dynasearch_descend` erreicht in jedem getesteten Lauf (15 Zufallsläufe) ein echtes 2-opt-Lokaloptimum, unabhängig gegen `dyna_tour.is_local_optimum` geprüft (nicht die eigene DP-Logik) - nur MIT dem abschließenden Aufräum-Durchgang; ohne ihn schlägt die Prüfung in 59 von 60 Läufen fehl (Depot-Ausnahme, siehe Grenzen).
- Übernommener Kern: 2-opt gegen Brute-Force, Abstieg strikt monoton und im lokalen Optimum, Bewertungsbudget, 1-Baum-Schranke; Instanz gegen eingefrorene Werte (aus der Hill-Climbing-Demo).
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Budget-, Effizienz-, Güte-Umkehr- und Skalierungs-Aussagen; jeweils Mittel über die festen Sweep-Instanzen × Ketten; positive **und** negative Aussagen; Rechenzeiten nur als Größenordnung), über dieselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/`ev.single_descent_comparison`), NIE über ein Ad-hoc-Skript mit abweichender Zufalls-Bindung (die Lehre aus der [lin-kernighan-demo](../lin-kernighan-demo) dieser Linie); alle 5 Presets über mehrere Instanzen und Ketten in Urteil-Bändern; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt bei 20 Stopps in beiden Modi, Iterations-Regler, ▶️ Abspielen und ▶️ Abstieg abspielen ohne doppelte Schlüssel, Würfel-Knöpfe, Permalink-Grenzen, Extremwerte, Experimente auf Abruf, Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Effizienz, Budget, Streuung, Skalierung), 🚧 Grenzen, Mathe |
| `dyna_algorithm.py` | Die Dynasearch-DP (`dynasearch_step`) und der Dynasearch-Abstieg mit Aufräum-Durchgang (`dynasearch_descend`) |
| `dyna_tour.py` | Nachbarschaften, Abstieg (mit Bewertungsbudget), Kreuzungen, 1-Baum-Schranke (aus der Hill-Climbing-Demo) |
| `dyna_scenario.py`, `dyna_constants.py` | Instanzen; Konstanten, Presets |
| `dyna_evaluation.py` | Analyse, Urteil, Neustarts, Sweeps, Effizienz-Vergleich, Streuung, Skalierung |
| `dyna_presets.py`, `dyna_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Übernommener Kern, Dynasearch-DP (Brute-Force-Vergleich, Additivität, Konvergenz), Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
