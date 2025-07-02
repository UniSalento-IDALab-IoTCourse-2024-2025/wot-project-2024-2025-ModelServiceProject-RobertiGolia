# Taxi AI

Taxi AI è un progetto Python che utilizza tecniche di Natural Language Processing (NLP) per riconoscere richieste di prenotazione taxi in linguaggio naturale, estrarre date e orari, e verificare la disponibilità degli autisti su un database CSV.

## Funzionalità principali

- Riconoscimento di entità come date e orari da messaggi testuali in italiano.
- Addestramento di un modello NLP personalizzato (basato su spaCy) per migliorare il riconoscimento.
- Interfaccia chatbot per interagire con l'utente e rispondere alle richieste di prenotazione.

## Struttura del progetto

- `chatbot_app.py`: applicazione principale per l'interazione con l'utente.
- `train_model.py`: script per addestrare il modello NLP sui dati di esempio.
- `training_examples.py`: contiene la funzione per generare dati di training.
- `data/drivers.csv`: dataset degli autisti e delle loro disponibilità.
- `taxi_nlp_model/` o `nlp_model/`: cartella dove viene salvato il modello NLP addestrato.
- `venv/`: ambiente virtuale Python. (utilizzato per test)

## Dipendenze principali

- Python 3.10+
- spaCy
- pandas

Per installare le dipendenze principali:

```bash
pip install spacy pandas
```

## Come usare il progetto

1. **Addestra il modello NLP** (solo la prima volta o quando aggiorni i dati di training):

   ```bash
   python train_model.py
   ```

   Questo genererà la cartella `taxi_nlp_model` con il modello addestrato.

2. **Avvia il chatbot**:
   ```bash
   python chatbot_app.py
   ```
   Segui le istruzioni a schermo per inviare richieste in linguaggio naturale (es: "Taxi per domani alle 10").

## Note

- Il file `drivers.csv` deve essere formattato correttamente con le colonne richieste (nome, data, orario_inizio, orario_fine).
- Il progetto è pensato per la lingua italiana.
