RAW_WEATHER_COLS = [
    "Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
]

ENGINEERED_WEATHER_COLS = [
    "heat_index",
    "humidex",
    "apparent_temperature",
    "thi",
]

FULL_WEATHER_COLS = RAW_WEATHER_COLS + ENGINEERED_WEATHER_COLS

DEMOGRAPHIC_COLS = ["Age", "Gender"]

HISTORY_COLS = [
    "asthma_history",
    "high_cholesterol",
    "diabetes",
    "obesity",
    "hiv_aids",
    "nasal_polyps",
    "high_blood_pressure",
]

WEATHER_DEMOGRAPHIC_COLS = DEMOGRAPHIC_COLS + FULL_WEATHER_COLS
PRE_EVENT_CONTEXT_COLS = DEMOGRAPHIC_COLS + FULL_WEATHER_COLS + HISTORY_COLS

CORE_SYMPTOMS = [
    "nausea",
    "joint_pain",
    "abdominal_pain",
    "high_fever",
    "chills",
    "fatigue",
    "runny_nose",
    "dizziness",
    "headache",
    "chest_pain",
    "vomiting",
    "cough",
    "shivering",
    "asthma",
    "severe_headache",
    "weakness",
    "trouble_seeing",
    "fever",
    "body_aches",
    "sore_throat",
    "sneezing",
    "diarrhea",
    "rapid_breathing",
    "rapid_heart_rate",
    "pain_behind_eyes",
    "swollen_glands",
    "rashes",
    "sinus_headache",
    "facial_pain",
    "shortness_of_breath",
    "reduced_smell_and_taste",
    "skin_irritation",
    "itchiness",
    "throbbing_headache",
    "confusion",
    "back_pain",
    "knee_ache",
]

MODELED_SYMPTOMS = [
    symptom for symptom in CORE_SYMPTOMS if symptom != "shivering"
]

TARGET_COL = "prognosis"
SOURCE_DUPLICATE_SYMPTOM = "pain_behind_the_eyes"
