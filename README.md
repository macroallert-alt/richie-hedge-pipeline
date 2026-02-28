\# richie-hedge-pipeline



Baldur Creek Capital — Daily Pipeline

V16 Daily Runner → dashboard.json → Vercel Frontend



\## Struktur

\- V16\_DAILY\_RUNNER.py — Holt Daten aus V16 Sheet, berechnet heutige Allokation, generiert dashboard.json

\- data/dashboard/latest.json — Aktuelle Dashboard-Daten (wird täglich überschrieben)

\- .github/workflows/daily\_v16.yml — GitHub Actions (täglich 06:00 UTC)

