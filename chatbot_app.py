
import spacy
import pandas as pd
from datetime import datetime, timedelta
import re
import os 

MODEL_PATH = "taxi_nlp_model"

nlp = None # Inizializza del modello a None, sarà caricato nel blocco del main

#Caricamento dati con verifica (riutilizzata)
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

#Funzioni di parsing  
def parse_date(date_str, reference_date=None):
    if not date_str:
        return None

    if reference_date is None:
        reference_date = datetime.now() 

    date_str = date_str.lower().strip()

    if date_str == "domani":
        return reference_date + timedelta(days=1)
    elif date_str == "oggi":
        return reference_date
    elif date_str == "dopodomani":
        return reference_date + timedelta(days=2)

    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
        "%d/%m/%y", "%d %B %Y", "%d %b %Y",
        "%d-%m-%y", "%Y/%m/%d"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def parse_time(time_str):
    if not time_str:
        return None
        
    time_str = str(time_str).lower().strip()
    
    if time_str.isdigit():
        try:
            hour = int(time_str)
            if 0 <= hour <= 23:
                return datetime.strptime(f"{hour:02d}:00", "%H:%M").time()
        except ValueError:
            pass
            
    formats = ["%H:%M", "%H"] 
    
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt).time()
        except ValueError:
            continue
    return None

#query database (riutilizzata)
def query_database(date_obj, time_obj=None):
    try:
        df = load_data()
        if df.empty:
            return [], 0
            
        available = df[df['data'].dt.date == date_obj.date()]
        
        if time_obj:
            available = available[
                (available['orario_inizio'] <= time_obj) & 
                (available['orario_fine'] >= time_obj)
            ]
        
        return available['nome'].tolist(), len(available)
    except Exception as e:
        print(f"Errore query: {e}")
        return [], 0

#per pilizia (riutilizzata)
def clean_text(text):
    corrections = {
        "disponobilità": "disponibilità",
        "dispobilita": "disponibilità",
        "disponibilita": "disponibilità"
    }
    text = text.lower()
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

# eventuale fallback (riutilizzata)
def process_message(message):
    global nlp 
    if nlp is None:
        raise RuntimeError("Il modello NLP non è stato caricato. Eseguire prima il training con 'train_model.py'.")

    cleaned_msg = clean_text(message)
    doc = nlp(cleaned_msg) 

    print(f"\n[DEBUG] Messaggio pulito: '{cleaned_msg}'") 
    print("[DEBUG] Entità riconosciute da SpaCy:")      
    
    spacy_dates = []
    spacy_times = []
    for ent in doc.ents:                                
        print(f"  - Testo: '{ent.text}', Label: '{ent.label_}', Start: {ent.start_char}, End: {ent.end_char}")
        if ent.label_ == "DATE":
            spacy_dates.append(ent.text)
        elif ent.label_ == "TIME":
            spacy_times.append(ent.text)

    date_obj = None
    time_obj = None

    if spacy_dates:
        date_obj = parse_date(spacy_dates[0])
    
    if not date_obj:
        date_patterns = [
            r"\d{2}/\d{2}/\d{4}",  
            r"\d{2}-\d{2}-\d{4}",  
            r"\d{4}-\d{2}-\d{2}",  
            r"\d{4}/\d{2}/\d{2}",  
        ]
        for pattern in date_patterns:
            match = re.search(pattern, cleaned_msg)
            if match:
                date_obj = parse_date(match.group())
                if date_obj:
                    print(f"  [DEBUG] Data trovata tramite RegEx fallback: '{match.group()}'")
                    break
    
    if spacy_times:
        time_obj = parse_time(spacy_times[0])

    if not time_obj:
        time_patterns = re.search(r'(?:alle|dalle)\s*(\d{1,2}(?::\d{2})?)', cleaned_msg)
        if time_patterns:
            extracted_time_str = time_patterns.group(1)
            parsed_time = parse_time(extracted_time_str)
            if parsed_time:
                time_obj = parsed_time
                print(f"  [DEBUG] Ora trovata tramite RegEx fallback: '{extracted_time_str}'")
        
        if not time_obj:
            potential_hours = re.findall(r'\b(0?[1-9]|1[0-9]|2[0-3])\b', cleaned_msg)
            if date_obj:
                parsed_date_parts = date_obj.strftime("%d %m %Y").split()
                potential_hours = [h for h in potential_hours if h not in parsed_date_parts]
            
            if potential_hours:
                try:
                    hour_val = int(potential_hours[0])
                    if 0 <= hour_val <= 23:
                        time_obj = datetime.strptime(f"{hour_val:02d}:00", "%H:%M").time()
                        print(f"  [DEBUG] Ora trovata tramite fallback numerico generico: '{hour_val}'")
                except ValueError:
                    pass

    if date_obj:
        drivers, count = query_database(date_obj, time_obj)
        
        if count > 0:
            return {
                "status": "success",
                "date": date_obj.strftime("%d/%m/%Y"),
                "time": time_obj.strftime("%H:%M") if time_obj else "tutto il giorno",
                "drivers": drivers,
                "count": count
            }
        else:
            return {
                "status": "no_drivers",
                "message": f"Nessun autista trovato per il {date_obj.strftime('%d/%m/%Y')}" + 
                          (f" alle {time_obj.strftime('%H:%M')}" if time_obj else "")
            }
    else:
        return {
            "status": "error",
            "message": "Specifica una data valida (es. 'domani', '15/03/2025')"
        }

#Test automatici utili per la demo
def run_tests():
    test_cases = [
        ("Taxi per domani", True),
        ("Disponibilità il 25/12/2025", True),
        ("Autista oggi alle 14:00", True),
        ("Prenota per 30-03-2025", True),
        ("Disponobilità per il 17/04/2026", True),
        ("Dispobilita per il 2026-04-17", True),
        ("Chi è libero?", False), 
        ("Qual è il tuo nome?", False), 
        ("Il 02/10/2025 alle 10:00", True),

    ]
    
    print("\n=== TEST AUTOMATICI ===")
    for i, (msg, should_pass) in enumerate(test_cases, 1):
        result = process_message(msg)
        passed = (result["status"] in ("success", "no_drivers")) if should_pass else (result["status"] == "error")
        print(f"Test {i}: {'✅' if passed else '❌'} '{msg}' => Status: {result['status']}" +
              (f", Data: {result.get('date')}, Ora: {result.get('time')}" if passed else ""))







# --------------- Parte principale ---------------
if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        try:
            nlp = spacy.load(MODEL_PATH)
            print(f"Modello NLP caricato da: {MODEL_PATH}")
        except Exception as e:
            print(f"Errore caricamento modello da {MODEL_PATH}: {e}")
            print("Assicurati di aver eseguito 'python train_model.py' almeno una volta.")
            exit() 
    else:
        print(f"La directory del modello '{MODEL_PATH}' non trovata.")
        print("Esegui 'python train_model.py' per addestrare e salvare il modello.")
        exit() # Esce se la directory del modello non esiste

    drivers_df = load_data() 

    # Esegui test (da commentare se non si devono fare i test automatici)
    #run_tests()
    
    # Modalità interattiva
    """
    #giusto per avere una linea guida
    print("\n=== MODALITÀ INTERATTIVA ===")
    print("Scrivi una richiesta o 'exit' per uscire")
    print("Esempi validi:")
    print("- Taxi per domani")
    print("- Disponibilità il 25/12/2025")
    print("- Autista oggi alle 14:00")
    print("- Disponobilità per il 17/04/2026")
    print("- Il 02/10/2025 alle 10:00")
    """

    while True:
        try:
            msg = input("\n> Richiesta (TU 👤): ").strip()
            if msg.lower() in ("exit", "quit"):
                break
                
            response = process_message(msg)
            
            if response["status"] == "success":
                print(f"\n(CHATBOT ✨)🔹 {response['count']} autisti disponibili il {response['date']} {response['time']}:\n")
                for driver in response["drivers"]:
                    print(f"- {driver}")
            elif response["status"] == "no_drivers":
                print(f"\n(CHATBOT ✨)🔸 {response['message']}")
            else:
                print(f"\n(CHATBOT ✨)❌ {response['message']}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n⚠ Errore: {str(e)}")

    print("\nApplicazione terminata.")