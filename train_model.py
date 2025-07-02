
import spacy
import pandas as pd
from datetime import datetime, timedelta
from spacy.training.example import Example
import random
from spacy.util import minibatch, compounding
import warnings
import re
import os 

# Importa la funzione dal modulo training_examples
# fatto solo per un codice più pulito
# perchè la funz era troppo grande
from training_examples import create_training_data

warnings.filterwarnings('ignore', category=UserWarning, module='spacy')


#Caricamento dati en verifica
def load_data():
    try:
        df = pd.read_csv('data/drivers.csv', parse_dates=['data'])
        df['orario_inizio'] = pd.to_datetime(df['orario_inizio'], format='%H:%M', errors='coerce').dt.time
        df['orario_fine'] = pd.to_datetime(df['orario_fine'], format='%H:%M', errors='coerce').dt.time
        df = df.dropna(subset=['orario_inizio', 'orario_fine'])
        print(f"Caricati {len(df)} record validi dal database")
        return df
    except Exception as e:
        print(f"Errore caricamento dati: {e}")
        return pd.DataFrame(columns=['nome', 'data', 'orario_inizio', 'orario_fine'])

# Funzione di training
def train_model(base_nlp_model, train_data, n_iter=500):
    print(f"\nInizio training del NER per {n_iter} iterazioni...")
    
    if "ner" not in base_nlp_model.pipe_names:
        ner_for_training = base_nlp_model.add_pipe("ner", last=True)
    else:
        ner_for_training = base_nlp_model.get_pipe("ner")

    # Aggiungi le label se non esistono già
    if "DATE" not in ner_for_training.labels:
        ner_for_training.add_label("DATE")
    if "TIME" not in ner_for_training.labels:
        ner_for_training.add_label("TIME")
        
    optimizer = base_nlp_model.begin_training()
    
    other_pipes = [pipe for pipe in base_nlp_model.pipe_names if pipe != "ner"]
    with base_nlp_model.disable_pipes(*other_pipes): 
        sizes = compounding(1.0, 4.0, 1.001)
        for itn in range(n_iter):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=sizes)
            losses = {}
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    examples.append(Example.from_dict(base_nlp_model.make_doc(text), annotations))
                base_nlp_model.update(examples, drop=0.5, losses=losses, sgd=optimizer)
            print(f"Iterazione {itn + 1}/{n_iter} - Perdita NER: {losses.get('ner', 0.0):.2f}")
    print("Training completato.")
    return base_nlp_model

#main per il training 
if __name__ == "__main__":
    # carica il modello base  di spacy (it_core_news_sm)
    try:
        nlp = spacy.load("it_core_news_sm")
        print("Modello 'it_core_news_sm' caricato per l'addestramento.")
    except OSError:
        print("Download del modello 'it_core_news_sm'...")
        spacy.cli.download("it_core_news_sm")
        nlp = spacy.load("it_core_news_sm")
        print("Modello 'it_core_news_sm' scaricato e caricato per l'addestramento.")

    train_data = create_training_data()
    
    nlp_trained = train_model(nlp, train_data)
    
    # Salva il modello addestrato in una nuova cartella
    output_dir = "taxi_nlp_model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nlp_trained.to_disk(output_dir)
    print(f"\nModello addestrato salvato in: {output_dir}")
    print("Puoi ora eseguire 'chatbot_app.py' per usare il modello salvato.")