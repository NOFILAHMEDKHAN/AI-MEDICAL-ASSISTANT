from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import urllib.parse

app = Flask(__name__)

# 94 symptoms
l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
      'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
      'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
      'distention_of_abdomen', 'history_of_alcohol_consumption', 'blood_in_sputum',
      'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
      'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
      'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

# Contradictory symptoms - pairs that cannot logically occur together
contradictory_symptoms = {
    # Bowel movements
    'constipation': ['diarrhoea'],
    'diarrhoea': ['constipation'],

    # Appetite
    'excessive_hunger': ['increased_appetite'],
    'increased_appetite': ['excessive_hunger'],

    # Consciousness - if in coma, cannot show other active symptoms
    'coma': ['slurred_speech', 'altered_sensorium', 'irritability', 'depression'],
    'slurred_speech': ['coma'],
    'altered_sensorium': ['coma'],
    'irritability': ['coma'],
    'depression': ['coma'],

    # Eyes - can't have both redness and yellowing simultaneously
    'redness_of_eyes': ['yellowing_of_eyes'],
    'yellowing_of_eyes': ['redness_of_eyes'],

    # Movement - stiffness and loss of balance are opposite ends
    'movement_stiffness': ['spinning_movements'],
    'spinning_movements': ['movement_stiffness'],

    # Urine smell
    'yellow_urine': ['foul_smell_of urine'],
    'foul_smell_of urine': ['yellow_urine'],

    # Weight
    'obesity': ['excessive_hunger'],
    'excessive_hunger': ['obesity'],
}

# Disease map
nmap = {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
    'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
    'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
    'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24,
    'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28,
    'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32,
    'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
    '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
    'Psoriasis': 39, 'Impetigo': 40
}
disease = sorted(nmap, key=lambda x: nmap[x])

# Disease to specialist mapping - specific specialist per disease
disease_to_specialist = {
    'Fungal infection':                        'Dermatologist',
    'Allergy':                                 'Allergist',
    'GERD':                                    'Gastroenterologist',
    'Chronic cholestasis':                     'Hepatologist',
    'Drug Reaction':                           'Allergist',
    'Peptic ulcer diseae':                     'Gastroenterologist',
    'AIDS':                                    'Infectious Disease Specialist',
    'Diabetes ':                               'Endocrinologist',
    'Gastroenteritis':                         'Gastroenterologist',
    'Bronchial Asthma':                        'Pulmonologist',
    'Hypertension ':                           'Cardiologist',
    'Migraine':                                'Neurologist',
    'Cervical spondylosis':                    'Orthopedic Surgeon',
    'Paralysis (brain hemorrhage)':            'Neurologist',
    'Jaundice':                                'Hepatologist',
    'Malaria':                                 'Infectious Disease Specialist',
    'Chicken pox':                             'General Physician',
    'Dengue':                                  'General Physician',
    'Typhoid':                                 'Infectious Disease Specialist',
    'hepatitis A':                             'Hepatologist',
    'Hepatitis B':                             'Hepatologist',
    'Hepatitis C':                             'Hepatologist',
    'Hepatitis D':                             'Hepatologist',
    'Hepatitis E':                             'Hepatologist',
    'Alcoholic hepatitis':                     'Hepatologist',
    'Tuberculosis':                            'Pulmonologist',
    'Common Cold':                             'General Physician',
    'Pneumonia':                               'Pulmonologist',
    'Dimorphic hemmorhoids(piles)':            'Proctologist',
    'Heart attack':                            'Cardiologist',
    'Varicose veins':                          'Vascular Surgeon',
    'Hypothyroidism':                          'Endocrinologist',
    'Hyperthyroidism':                         'Endocrinologist',
    'Hypoglycemia':                            'Endocrinologist',
    'Osteoarthristis':                         'Orthopedic Surgeon',
    'Arthritis':                               'Rheumatologist',
    '(vertigo) Paroymsal  Positional Vertigo': 'ENT Specialist',
    'Acne':                                    'Dermatologist',
    'Urinary tract infection':                 'Urologist',
    'Psoriasis':                               'Dermatologist',
    'Impetigo':                                'Dermatologist'
}

# ML models
dt_model = rf_model = nb_model = None

def train_models():
    global dt_model, rf_model, nb_model
    df = pd.read_csv("C:/Users/PMLS/Downloads/Training.csv")
    tr = pd.read_csv("C:/Users/PMLS/Downloads/Testing.csv")
    df.replace({'prognosis': nmap}, inplace=True)
    tr.replace({'prognosis': nmap}, inplace=True)
    X, y = df[l1], df['prognosis']

    dt_model = DecisionTreeClassifier().fit(X, y)
    rf_model = RandomForestClassifier().fit(X, y)
    nb_model = GaussianNB().fit(X, y)

train_models()


# Direct Marham specialty slugs (exact pages that work on marham.pk)
marham_slugs = {
    "Dermatologist":                  "dermatologist",
    "Allergist":                       "allergy-specialist",
    "Gastroenterologist":              "gastroenterologist",
    "Hepatologist":                    "hepatologist",
    "Infectious Disease Specialist":   "infectious-diseases-specialist",
    "Endocrinologist":                 "endocrinologist",
    "Pulmonologist":                   "pulmonologist",
    "Cardiologist":                    "cardiologist",
    "Neurologist":                     "neurologist",
    "Orthopedic Surgeon":              "orthopedic-surgeon",
    "General Physician":               "general-physician",
    "Proctologist":                    "proctologist",
    "Vascular Surgeon":                "vascular-surgeon",
    "Rheumatologist":                  "rheumatologist",
    "ENT Specialist":                  "ent-specialist",
    "Urologist":                       "urologist",
}

def get_doctor_links(disease_name):
    """Returns direct clickable links to doctor directories. No scraping, no external requests."""
    specialist = disease_to_specialist.get(disease_name, "General Physician")
    specialist_encoded = urllib.parse.quote(specialist)
    marham_slug = marham_slugs.get(specialist, specialist.lower().replace(" ", "-"))

    return [
        {
            "title": f"Find {specialist} on Marham.pk",
            "url": f"https://www.marham.pk/doctors/{marham_slug}"
        },
        {
            "title": f"Find {specialist} on OlaDoc",
            "url": f"https://oladoc.com/pakistan/search?q={specialist_encoded}"
        },
        {
            "title": f"Find {specialist} on Healthwire",
            "url": f"https://healthwire.pk/doctors/{marham_slug}"
        },
    ]


@app.route('/', methods=['GET', 'POST'])
def index():
    original = {}
    enhanced = {}
    error_message = None
    selected_symptoms = []

    if request.method == 'POST':
        selected_symptoms = [request.form.get(f'symptom{i}') for i in range(1, 6)
                             if request.form.get(f'symptom{i}') and request.form.get(f'symptom{i}') != "None"]

        has_contradiction = False
        for symptom in selected_symptoms:
            if symptom in contradictory_symptoms:
                for contradictory in contradictory_symptoms[symptom]:
                    if contradictory in selected_symptoms:
                        error_message = f"Contradictory symptoms: '{symptom}' and '{contradictory}' cannot occur together."
                        has_contradiction = True
                        break
            if has_contradiction:
                break

        if len(selected_symptoms) != len(set(selected_symptoms)):
            error_message = "Duplicate symptoms selected. Please select different symptoms."
            has_contradiction = True

        if not has_contradiction and len(selected_symptoms) < 3:
            error_message = "Please select at least 3 symptoms for an accurate prediction."
            has_contradiction = True

        if not has_contradiction and selected_symptoms:
            l2 = [1 if s in selected_symptoms else 0 for s in l1]

            dt_idx = dt_model.predict([l2])[0]
            rf_idx = rf_model.predict([l2])[0]
            nb_idx = nb_model.predict([l2])[0]

            dt_conf = round(dt_model.predict_proba([l2])[0][dt_idx] * 100, 1)
            rf_conf = round(rf_model.predict_proba([l2])[0][rf_idx] * 100, 1)
            nb_conf = round(nb_model.predict_proba([l2])[0][nb_idx] * 100, 1)

            original = {
                'Decision Tree': disease[dt_idx],
                'Random Forest': disease[rf_idx],
                'Naive Bayes': disease[nb_idx]
            }

            for model_name, idx, conf in [
                ('Decision Tree', dt_idx, dt_conf),
                ('Random Forest', rf_idx, rf_conf),
                ('Naive Bayes', nb_idx, nb_conf)
            ]:
                dis = disease[idx]
                enhanced[model_name] = {
                    'disease': dis,
                    'confidence': conf,
                    'doctors': get_doctor_links(dis)
                }
        elif not selected_symptoms:
            error_message = "Please select at least 3 symptoms for an accurate prediction."

    return render_template(
        'index.html',
        symptoms=l1,
        original=original,
        enhanced=enhanced,
        error_message=error_message,
        selected_symptoms=selected_symptoms
    )


if __name__ == '__main__':
    app.run(debug=True)