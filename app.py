# AI_DOCTOR_ASSISTANT_REAL_WORLD.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import heapq
from typing import List, Tuple, Dict, Set
import math
import json
import sqlite3
from contextlib import contextmanager

# Configure Streamlit for production
st.set_page_config(
    page_title="AI Doctor Assistant - Real Medical Advisor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database setup for real data persistence
@contextmanager
def get_db_connection():
    conn = sqlite3.connect('medical_assistant.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize database with real medical schemas"""
    with get_db_connection() as conn:
        # Patient records
        conn.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE,
                name TEXT,
                age INTEGER,
                gender TEXT,
                blood_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Medical history
        conn.execute('''
            CREATE TABLE IF NOT EXISTS medical_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                condition TEXT,
                diagnosis_date DATE,
                severity TEXT,
                treatment TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Symptoms and diagnoses
        conn.execute('''
            CREATE TABLE IF NOT EXISTS symptom_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                symptoms TEXT,
                diagnosis TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Medication tracking
        conn.execute('''
            CREATE TABLE IF NOT EXISTS medications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                medication_name TEXT,
                dosage TEXT,
                frequency TEXT,
                start_date DATE,
                end_date DATE,
                status TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        # Vital signs
        conn.execute('''
            CREATE TABLE IF NOT EXISTS vital_signs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                heart_rate INTEGER,
                blood_pressure TEXT,
                temperature REAL,
                oxygen_saturation REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        conn.commit()

# Initialize database on startup
init_database()

class RealMedicalDatabase:
    """Real medical database with comprehensive symptom-disease relationships"""
    
    def __init__(self):
        self.symptoms_diseases = {
            # Respiratory System
            'cough': ['Common Cold', 'COVID-19', 'Influenza', 'Pneumonia', 'Bronchitis', 'Asthma'],
            'fever': ['Common Cold', 'COVID-19', 'Influenza', 'Pneumonia', 'Urinary Tract Infection', 'Dengue'],
            'shortness_of_breath': ['COVID-19', 'Pneumonia', 'Asthma', 'Heart Failure', 'COPD', 'Anxiety'],
            'chest_pain': ['Heart Attack', 'Angina', 'Pneumonia', 'GERD', 'Anxiety Attack'],
            'sore_throat': ['Common Cold', 'COVID-19', 'Influenza', 'Strep Throat', 'Tonsillitis'],
            
            # Digestive System
            'nausea': ['Food Poisoning', 'Gastroenteritis', 'Migraine', 'Pregnancy', 'Appendicitis'],
            'vomiting': ['Food Poisoning', 'Gastroenteritis', 'Migraine', 'Appendicitis'],
            'abdominal_pain': ['Appendicitis', 'Food Poisoning', 'Irritable Bowel', 'Kidney Stones'],
            'diarrhea': ['Food Poisoning', 'Gastroenteritis', 'Irritable Bowel'],
            
            # Neurological
            'headache': ['Migraine', 'Tension Headache', 'Sinusitis', 'Dehydration', 'Hypertension'],
            'dizziness': ['Vertigo', 'Anemia', 'Low Blood Pressure', 'Dehydration'],
            'fatigue': ['Anemia', 'Depression', 'Hypothyroidism', 'Chronic Fatigue'],
            
            # Cardiovascular
            'palpitations': ['Anxiety', 'Arrhythmia', 'Anemia', 'Hyperthyroidism'],
            'swollen_ankles': ['Heart Failure', 'Kidney Disease', 'Liver Disease'],
            
            # General
            'muscle_pain': ['Influenza', 'COVID-19', 'Fibromyalgia', 'Arthritis'],
            'rash': ['Allergic Reaction', 'Eczema', 'Psoriasis', 'Measles']
        }
        
        self.disease_details = {
            'Common Cold': {
                'symptoms': ['cough', 'fever', 'sore_throat', 'muscle_pain'],
                'severity': 'low',
                'urgency': 'non_urgent',
                'recommendation': 'Rest, hydrate, over-the-counter cold medication',
                'treatment_steps': ['rest', 'hydration', 'otc_medication', 'symptom_monitoring'],
                'recovery_time': '7-10 days'
            },
            'COVID-19': {
                'symptoms': ['fever', 'cough', 'shortness_of_breath', 'fatigue', 'muscle_pain'],
                'severity': 'high',
                'urgency': 'urgent',
                'recommendation': 'Isolate immediately, get tested, consult doctor',
                'treatment_steps': ['isolation', 'testing', 'medical_consultation', 'symptom_tracking'],
                'recovery_time': '2-3 weeks'
            },
            'Influenza': {
                'symptoms': ['fever', 'cough', 'muscle_pain', 'fatigue', 'headache'],
                'severity': 'moderate',
                'urgency': 'semi_urgent',
                'recommendation': 'Rest, antiviral medication if early, symptom management',
                'treatment_steps': ['rest', 'antiviral_medication', 'fever_management', 'hydration'],
                'recovery_time': '1-2 weeks'
            },
            'Pneumonia': {
                'symptoms': ['fever', 'cough', 'shortness_of_breath', 'chest_pain'],
                'severity': 'high',
                'urgency': 'emergency',
                'recommendation': 'Seek immediate medical attention, may require hospitalization',
                'treatment_steps': ['emergency_care', 'antibiotics', 'hospitalization', 'oxygen_therapy'],
                'recovery_time': '3-6 weeks'
            },
            'Heart Attack': {
                'symptoms': ['chest_pain', 'shortness_of_breath', 'palpitations'],
                'severity': 'critical',
                'urgency': 'emergency',
                'recommendation': 'CALL EMERGENCY SERVICES IMMEDIATELY - Chew aspirin if available',
                'treatment_steps': ['call_emergency', 'aspirin', 'hospitalization', 'cardiac_care'],
                'recovery_time': 'Several months'
            },
            'Asthma': {
                'symptoms': ['shortness_of_breath', 'cough', 'chest_pain'],
                'severity': 'moderate',
                'urgency': 'urgent',
                'recommendation': 'Use inhaler if available, seek medical help if severe',
                'treatment_steps': ['inhaler', 'medical_consultation', 'trigger_avoidance', 'monitoring'],
                'recovery_time': 'Chronic condition'
            }
        }
        
        self.drug_interactions = {
            'warfarin': ['aspirin', 'ibuprofen', 'naproxen'],
            'aspirin': ['warfarin', 'ibuprofen', 'alcohol'],
            'ibuprofen': ['warfarin', 'aspirin', 'lithium'],
            'metformin': ['alcohol'],
            'statins': ['grapefruit_juice']
        }
        
        self.risk_factors = {
            'age_65_plus': ['Pneumonia', 'Influenza', 'Heart Disease'],
            'diabetes': ['COVID-19', 'Pneumonia', 'Urinary Tract Infection'],
            'hypertension': ['Heart Attack', 'Stroke', 'Kidney Disease'],
            'smoking': ['COPD', 'Lung Cancer', 'Heart Disease'],
            'obesity': ['Diabetes', 'Heart Disease', 'Sleep Apnea']
        }

class PatientManager:
    """Real patient management system"""
    
    def __init__(self):
        self.current_patient = None
    
    def create_patient(self, name, age, gender, blood_type):
        """Create new patient record"""
        patient_id = f"PAT{random.randint(10000, 99999)}"
        
        with get_db_connection() as conn:
            conn.execute(
                'INSERT INTO patients (patient_id, name, age, gender, blood_type) VALUES (?, ?, ?, ?, ?)',
                (patient_id, name, age, gender, blood_type)
            )
            conn.commit()
        
        return patient_id
    
    def get_patient(self, patient_id):
        """Retrieve patient data"""
        with get_db_connection() as conn:
            patient = conn.execute(
                'SELECT * FROM patients WHERE patient_id = ?', (patient_id,)
            ).fetchone()
        return dict(patient) if patient else None
    
    def add_medical_history(self, patient_id, condition, diagnosis_date, severity, treatment):
        """Add to patient medical history"""
        with get_db_connection() as conn:
            conn.execute(
                '''INSERT INTO medical_history 
                (patient_id, condition, diagnosis_date, severity, treatment) 
                VALUES (?, ?, ?, ?, ?)''',
                (patient_id, condition, diagnosis_date, severity, treatment)
            )
            conn.commit()
    
    def record_symptom_analysis(self, patient_id, symptoms, diagnosis, confidence):
        """Record symptom analysis results"""
        with get_db_connection() as conn:
            conn.execute(
                '''INSERT INTO symptom_records 
                (patient_id, symptoms, diagnosis, confidence) 
                VALUES (?, ?, ?, ?)''',
                (patient_id, json.dumps(symptoms), diagnosis, confidence)
            )
            conn.commit()
    
    def add_medication(self, patient_id, medication_name, dosage, frequency, start_date, end_date):
        """Prescribe medication"""
        with get_db_connection() as conn:
            conn.execute(
                '''INSERT INTO medications 
                (patient_id, medication_name, dosage, frequency, start_date, end_date, status) 
                VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (patient_id, medication_name, dosage, frequency, start_date, end_date, 'active')
            )
            conn.commit()
    
    def record_vital_signs(self, patient_id, heart_rate, blood_pressure, temperature, oxygen_saturation):
        """Record patient vital signs"""
        with get_db_connection() as conn:
            conn.execute(
                '''INSERT INTO vital_signs 
                (patient_id, heart_rate, blood_pressure, temperature, oxygen_saturation) 
                VALUES (?, ?, ?, ?, ?)''',
                (patient_id, heart_rate, blood_pressure, temperature, oxygen_saturation)
            )
            conn.commit()

class MedicalAIAnalyzer:
    """Real AI medical analysis engine"""
    
    def __init__(self):
        self.database = RealMedicalDatabase()
        self.patient_manager = PatientManager()
    
    def analyze_symptoms(self, symptoms: List[str], patient_data: Dict) -> List[Dict]:
        """Analyze symptoms and return potential conditions with confidence scores"""
        
        possible_conditions = {}
        
        # Find diseases matching symptoms
        for symptom in symptoms:
            for disease in self.database.symptoms_diseases.get(symptom, []):
                if disease not in possible_conditions:
                    possible_conditions[disease] = {
                        'matching_symptoms': [],
                        'total_symptoms': len(self.database.disease_details.get(disease, {}).get('symptoms', [])),
                        'severity': self.database.disease_details.get(disease, {}).get('severity', 'unknown'),
                        'risk_factors': []
                    }
                possible_conditions[disease]['matching_symptoms'].append(symptom)
        
        # Calculate confidence scores
        results = []
        for disease, data in possible_conditions.items():
            # Base confidence based on symptom matching
            symptom_match_ratio = len(data['matching_symptoms']) / max(1, data['total_symptoms'])
            base_confidence = symptom_match_ratio * 80  # Base score out of 80
            
            # Adjust for risk factors
            risk_adjustment = self._calculate_risk_adjustment(disease, patient_data)
            
            # Adjust for symptom severity
            severity_adjustment = self._calculate_severity_adjustment(data['severity'])
            
            final_confidence = min(95, base_confidence + risk_adjustment + severity_adjustment)
            
            # Ensure minimum confidence for display
            if final_confidence > 20:
                disease_info = self.database.disease_details.get(disease, {})
                results.append({
                    'disease': disease,
                    'confidence': final_confidence,
                    'severity': data['severity'],
                    'urgency': disease_info.get('urgency', 'unknown'),
                    'recommendation': disease_info.get('recommendation', 'Consult healthcare provider'),
                    'treatment_steps': disease_info.get('treatment_steps', []),
                    'recovery_time': disease_info.get('recovery_time', 'Unknown'),
                    'matching_symptoms': data['matching_symptoms'],
                    'risk_factors': data['risk_factors']
                })
        
        # Sort by confidence and severity
        results.sort(key=lambda x: (x['confidence'], x['severity']), reverse=True)
        return results[:5]  # Return top 5 results
    
    def _calculate_risk_adjustment(self, disease: str, patient_data: Dict) -> float:
        """Calculate risk adjustment based on patient factors"""
        adjustment = 0
        
        # Age-based risk
        age = patient_data.get('age', 40)
        if age > 65 and disease in self.database.risk_factors.get('age_65_plus', []):
            adjustment += 10
        
        # Comorbidity risks
        comorbidities = patient_data.get('comorbidities', [])
        for comorbidity in comorbidities:
            if disease in self.database.risk_factors.get(comorbidity.lower(), []):
                adjustment += 5
        
        return adjustment
    
    def _calculate_severity_adjustment(self, severity: str) -> float:
        """Adjust confidence based on disease severity"""
        severity_weights = {
            'critical': 15,
            'high': 10,
            'moderate': 5,
            'low': 0
        }
        return severity_weights.get(severity, 0)
    
    def check_drug_interactions(self, current_meds: List[str], new_med: str) -> List[str]:
        """Check for potential drug interactions"""
        interactions = []
        for med in current_meds:
            if med in self.database.drug_interactions:
                if new_med in self.database.drug_interactions[med]:
                    interactions.append(f"{med} + {new_med}")
        return interactions
    
    def assess_emergency_condition(self, symptoms: List[str], vital_signs: Dict) -> Dict:
        """Assess if condition requires emergency care"""
        emergency_indicators = []
        
        # Critical symptoms
        critical_symptoms = ['chest_pain', 'shortness_of_breath', 'severe_bleeding', 'loss_of_consciousness']
        if any(symptom in symptoms for symptom in critical_symptoms):
            emergency_indicators.append("Critical symptoms present")
        
        # Abnormal vital signs
        if vital_signs.get('heart_rate', 0) > 150 or vital_signs.get('heart_rate', 0) < 40:
            emergency_indicators.append("Abnormal heart rate")
        
        if vital_signs.get('temperature', 37) > 39.5:  # High fever > 39.5°C
            emergency_indicators.append("High fever")
        
        if vital_signs.get('oxygen_saturation', 98) < 92:
            emergency_indicators.append("Low oxygen saturation")
        
        is_emergency = len(emergency_indicators) > 0
        return {
            'is_emergency': is_emergency,
            'indicators': emergency_indicators,
            'recommendation': "SEEK IMMEDIATE MEDICAL ATTENTION" if is_emergency else "Monitor condition"
        }

class TreatmentPlanner:
    """Real treatment planning system"""
    
    def __init__(self):
        self.database = RealMedicalDatabase()
    
    def generate_treatment_plan(self, diagnosis: str, patient_data: Dict) -> Dict:
        """Generate comprehensive treatment plan"""
        disease_info = self.database.disease_details.get(diagnosis, {})
        
        plan = {
            'diagnosis': diagnosis,
            'urgency': disease_info.get('urgency', 'unknown'),
            'timeline': disease_info.get('recovery_time', 'Unknown'),
            'medications': self._suggest_medications(diagnosis, patient_data),
            'lifestyle_recommendations': self._suggest_lifestyle_changes(diagnosis),
            'monitoring_instructions': self._suggest_monitoring(diagnosis),
            'follow_up': self._suggest_follow_up(diagnosis),
            'warning_signs': self._get_warning_signs(diagnosis)
        }
        
        return plan
    
    def _suggest_medications(self, diagnosis: str, patient_data: Dict) -> List[Dict]:
        """Suggest appropriate medications"""
        medication_protocols = {
            'Common Cold': [
                {'name': 'Acetaminophen', 'purpose': 'Fever and pain relief', 'dosage': '500mg every 6 hours'},
                {'name': 'Decongestant', 'purpose': 'Nasal congestion', 'dosage': 'As directed'}
            ],
            'COVID-19': [
                {'name': 'Paxlovid', 'purpose': 'Antiviral treatment', 'dosage': 'As prescribed by doctor'},
                {'name': 'Acetaminophen', 'purpose': 'Fever management', 'dosage': '500mg every 6 hours as needed'}
            ],
            'Influenza': [
                {'name': 'Oseltamivir', 'purpose': 'Antiviral', 'dosage': 'As prescribed (start within 48 hours)'},
                {'name': 'Ibuprofen', 'purpose': 'Fever and pain', 'dosage': '400mg every 6 hours'}
            ],
            'Asthma': [
                {'name': 'Albuterol inhaler', 'purpose': 'Quick relief', 'dosage': '1-2 puffs every 4-6 hours as needed'}
            ]
        }
        
        return medication_protocols.get(diagnosis, [{'name': 'Consult doctor for medication', 'purpose': 'Professional assessment needed', 'dosage': 'As prescribed'}])
    
    def _suggest_lifestyle_changes(self, diagnosis: str) -> List[str]:
        """Suggest lifestyle modifications"""
        recommendations = {
            'Common Cold': [
                "Increase fluid intake to 8-10 glasses daily",
                "Get plenty of rest",
                "Use humidifier for nasal congestion",
                "Avoid alcohol and smoking"
            ],
            'COVID-19': [
                "Isolate from others for at least 5 days",
                "Monitor oxygen saturation regularly",
                "Rest and hydrate frequently",
                "Sleep in prone position if breathing difficulty"
            ],
            'Influenza': [
                "Bed rest for 3-5 days",
                "Hydrate with water and electrolyte solutions",
                "Avoid contact with vulnerable individuals",
                "Nutritious diet with protein and vitamins"
            ],
            'Heart Attack': [
                "Complete rest until medical evaluation",
                "Low-sodium, heart-healthy diet",
                "Cardiac rehabilitation program",
                "Stress management techniques"
            ]
        }
        
        return recommendations.get(diagnosis, ["Consult healthcare provider for personalized recommendations"])
    
    def _suggest_monitoring(self, diagnosis: str) -> List[str]:
        """Suggest monitoring parameters"""
        monitoring = {
            'COVID-19': [
                "Temperature every 6 hours",
                "Oxygen saturation 3 times daily",
                "Respiratory rate if feeling short of breath",
                "Symptom progression tracking"
            ],
            'Pneumonia': [
                "Temperature monitoring",
                "Respiratory rate assessment",
                "Oxygen saturation checks",
                "Cough severity and sputum color"
            ],
            'Heart Attack': [
                "Blood pressure monitoring",
                "Heart rate tracking",
                "Weight daily for fluid retention",
                "Chest pain frequency and severity"
            ]
        }
        
        return monitoring.get(diagnosis, ["Monitor symptom changes and report worsening conditions"])
    
    def _suggest_follow_up(self, diagnosis: str) -> str:
        """Suggest follow-up timeline"""
        follow_up = {
            'Common Cold': "Return if symptoms worsen or persist beyond 10 days",
            'COVID-19': "Telemedicine follow-up in 2-3 days, emergency if breathing difficulty",
            'Pneumonia': "Follow-up with doctor in 1 week, sooner if worsening",
            'Heart Attack': "Cardiology follow-up within 1 week of discharge"
        }
        
        return follow_up.get(diagnosis, "Consult healthcare provider for follow-up schedule")
    
    def _get_warning_signs(self, diagnosis: str) -> List[str]:
        """Provide warning signs for emergency"""
        warnings = {
            'COVID-19': [
                "Difficulty breathing",
                "Persistent chest pain",
                "Confusion or inability to stay awake",
                "Blue lips or face"
            ],
            'Pneumonia': [
                "Rapid breathing",
                "Severe chest pain",
                "High fever not responding to medication",
                "Worsening cough with bloody mucus"
            ],
            'Heart Attack': [
                "Chest pain spreading to arm, neck or jaw",
                "Severe shortness of breath",
                "Nausea with cold sweat",
                "Lightheadedness or fainting"
            ]
        }
        
        return warnings.get(diagnosis, ["Seek immediate help if symptoms worsen dramatically"])

class AStarSymptomChecker:
    """Advanced A* algorithm for optimal symptom-disease matching"""
    
    def __init__(self, medical_db):
        self.medical_db = medical_db
    
    def heuristic_symptom_match(self, current_symptoms: Set[str], target_disease: str) -> float:
        """A* heuristic based on symptom matching"""
        if target_disease not in self.medical_db.disease_details:
            return float('inf')
        
        target_symptoms = set(self.medical_db.disease_details[target_disease]['symptoms'])
        unmatched_symptoms = target_symptoms - current_symptoms
        
        severity_weights = {'critical': 0.1, 'high': 0.3, 'moderate': 0.6, 'low': 1.0}
        severity = self.medical_db.disease_details[target_disease].get('severity', 'moderate')
        weight = severity_weights.get(severity, 0.5)
        
        return len(unmatched_symptoms) * weight
    
    def a_star_diagnosis(self, symptoms: List[str]) -> List[Tuple]:
        """A* search for most probable diagnoses"""
        current_symptoms_set = set(symptoms)
        possible_diseases = set()
        
        # Get possible diseases from symptoms
        for symptom in symptoms:
            possible_diseases.update(self.medical_db.symptoms_diseases.get(symptom, []))
        
        # Priority queue for A* search
        priority_queue = []
        
        for disease in possible_diseases:
            # g(n) = number of patient symptoms not in disease symptoms
            target_symptoms = set(self.medical_db.disease_details.get(disease, {}).get('symptoms', []))
            g_score = len(current_symptoms_set - target_symptoms)
            
            # h(n) = heuristic estimate
            h_score = self.heuristic_symptom_match(current_symptoms_set, disease)
            
            # f(n) = g(n) + h(n)
            f_score = g_score + h_score
            
            heapq.heappush(priority_queue, (f_score, disease, g_score, h_score))
        
        # Return sorted results
        results = []
        while priority_queue:
            f_score, disease, g_score, h_score = heapq.heappop(priority_queue)
            confidence = max(0, 100 - f_score * 10)  # Convert to percentage
            results.append((disease, confidence, g_score, h_score, f_score))
        
        return results

def main():
    st.title("🏥 AI Doctor Assistant - Real Medical Advisor")
    st.markdown("""
    **Professional Healthcare Assistant with Real Medical Analysis**
    *This tool provides AI-powered symptom analysis and treatment guidance. 
    Always consult healthcare professionals for medical diagnoses.*
    """)
    
    # Initialize systems
    medical_db = RealMedicalDatabase()
    medical_ai = MedicalAIAnalyzer()
    treatment_planner = TreatmentPlanner()
    patient_manager = PatientManager()
    astar_checker = AStarSymptomChecker(medical_db)
    
    # Session state for patient data
    if 'current_patient' not in st.session_state:
        st.session_state.current_patient = None
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = {}
    
    # Sidebar - Patient Registration
    with st.sidebar:
        st.header("👤 Patient Registration")
        
        with st.form("patient_registration"):
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"])
            
            submitted = st.form_submit_button("Register Patient")
            if submitted and name:
                patient_id = patient_manager.create_patient(name, age, gender, blood_type)
                st.session_state.current_patient = patient_id
                st.session_state.patient_data = {
                    'name': name, 'age': age, 'gender': gender, 'blood_type': blood_type
                }
                st.success(f"Patient Registered! ID: {patient_id}")
        
        # Display current patient info
        if st.session_state.current_patient:
            st.divider()
            st.subheader("Current Patient")
            st.write(f"**Name:** {st.session_state.patient_data['name']}")
            st.write(f"**Age:** {st.session_state.patient_data['age']}")
            st.write(f"**Gender:** {st.session_state.patient_data['gender']}")
            st.write(f"**Blood Type:** {st.session_state.patient_data['blood_type']}")
    
    # Main application tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Symptom Checker", 
        "💊 Treatment Plans", 
        "📊 Patient Records", 
        "❤️ Vital Signs", 
        "🚨 Emergency Check"
    ])
    
    with tab1:
        st.header("AI Symptom Analysis")
        
        if not st.session_state.current_patient:
            st.warning("Please register as a patient first in the sidebar.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Symptom selection
                st.subheader("Select Your Symptoms")
                all_symptoms = list(medical_db.symptoms_diseases.keys())
                selected_symptoms = st.multiselect(
                    "Choose all that apply:",
                    all_symptoms,
                    help="Select all symptoms you're currently experiencing"
                )
                
                # Additional patient information
                st.subheader("Additional Health Information")
                comorbidities = st.multiselect(
                    "Existing Conditions:",
                    ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "None"]
                )
                current_medications = st.multiselect(
                    "Current Medications:",
                    ["Aspirin", "Warfarin", "Metformin", "Insulin", "Statins", "None"]
                )
                allergies = st.multiselect(
                    "Allergies:",
                    ["Penicillin", "Aspirin", "Sulfa", "Latex", "None"]
                )
                
                # Symptom duration and severity
                symptom_duration = st.selectbox(
                    "How long have you had these symptoms?",
                    ["Less than 24 hours", "1-3 days", "4-7 days", "1-2 weeks", "More than 2 weeks"]
                )
                symptom_severity = st.slider("Overall symptom severity", 1, 10, 5)
            
            with col2:
                st.subheader("Quick Assessment")
                if selected_symptoms:
                    st.info(f"**Selected Symptoms:** {', '.join(selected_symptoms)}")
                    st.write(f"**Duration:** {symptom_duration}")
                    st.write(f"**Severity:** {symptom_severity}/10")
                    
                    if st.button("Analyze Symptoms", type="primary"):
                        with st.spinner("AI analyzing symptoms..."):
                            # Prepare patient data
                            patient_info = {
                                'age': st.session_state.patient_data['age'],
                                'comorbidities': [c for c in comorbidities if c != "None"],
                                'current_medications': [m for m in current_medications if m != "None"],
                                'allergies': [a for a in allergies if a != "None"],
                                'symptom_duration': symptom_duration,
                                'symptom_severity': symptom_severity
                            }
                            
                            # Get AI analysis using both methods
                            results = medical_ai.analyze_symptoms(selected_symptoms, patient_info)
                            astar_results = astar_checker.a_star_diagnosis(selected_symptoms)
                            
                            if results:
                                # Record analysis
                                top_diagnosis = results[0]['disease']
                                medical_ai.patient_manager.record_symptom_analysis(
                                    st.session_state.current_patient,
                                    selected_symptoms,
                                    top_diagnosis,
                                    results[0]['confidence']
                                )
                                
                                # Display results
                                st.subheader("🔬 AI Diagnosis Results")
                                
                                for i, result in enumerate(results):
                                    with st.expander(f"🎯 {result['disease']} ({result['confidence']:.1f}% confidence)", expanded=i==0):
                                        st.write(f"**Severity:** {result['severity'].upper()}")
                                        st.write(f"**Urgency:** {result['urgency'].replace('_', ' ').title()}")
                                        st.write(f"**Matching Symptoms:** {', '.join(result['matching_symptoms'])}")
                                        st.write(f"**Recommendation:** {result['recommendation']}")
                                        st.write(f"**Expected Recovery:** {result['recovery_time']}")
                                        
                                        # Color code based on urgency
                                        if result['urgency'] == 'emergency':
                                            st.error("🚨 URGENT: Seek immediate medical attention!")
                                        elif result['urgency'] == 'urgent':
                                            st.warning("⚠️ URGENT: Consult doctor within 24 hours")
                                        else:
                                            st.success("✅ Non-urgent: Follow recommendations and monitor")
                                
                                # Show A* algorithm results
                                st.subheader("🤖 A* Algorithm Analysis")
                                for disease, confidence, g_score, h_score, f_score in astar_results[:3]:
                                    st.write(f"**{disease}**: {confidence:.1f}% (g={g_score}, h={h_score:.2f}, f={f_score:.2f})")
                                
                            else:
                                st.warning("No clear diagnosis found. Please consult a healthcare provider.")
                else:
                    st.info("Select symptoms to get started with analysis")
    
    with tab2:
        st.header("Treatment Plans & Medications")
        
        if not st.session_state.current_patient:
            st.warning("Please register as a patient first.")
        else:
            # Get previous diagnoses
            with get_db_connection() as conn:
                diagnoses = conn.execute(
                    'SELECT diagnosis, confidence, timestamp FROM symptom_records WHERE patient_id = ? ORDER BY timestamp DESC LIMIT 5',
                    (st.session_state.current_patient,)
                ).fetchall()
            
            if diagnoses:
                st.subheader("Recent Diagnoses")
                for diagnosis in diagnoses:
                    if st.button(f"View Treatment Plan for {diagnosis['diagnosis']}"):
                        plan = treatment_planner.generate_treatment_plan(
                            diagnosis['diagnosis'], 
                            st.session_state.patient_data
                        )
                        
                        st.subheader(f"Treatment Plan: {plan['diagnosis']}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Medications:**")
                            for med in plan['medications']:
                                st.write(f"• {med['name']}: {med['dosage']} - {med['purpose']}")
                            
                            st.write("**Lifestyle Recommendations:**")
                            for rec in plan['lifestyle_recommendations']:
                                st.write(f"• {rec}")
                        
                        with col2:
                            st.write("**Monitoring Instructions:**")
                            for monitor in plan['monitoring_instructions']:
                                st.write(f"• {monitor}")
                            
                            st.write("**Follow-up:**")
                            st.write(f"• {plan['follow_up']}")
                            
                            st.write("**Warning Signs:**")
                            for warning in plan['warning_signs']:
                                st.write(f"• ⚠️ {warning}")
            else:
                st.info("Complete a symptom analysis first to generate treatment plans")
    
    with tab3:
        st.header("Patient Medical Records")
        
        if not st.session_state.current_patient:
            st.warning("Please register as a patient first.")
        else:
            # Display patient information
            patient_info = patient_manager.get_patient(st.session_state.current_patient)
            if patient_info:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Personal Information")
                    st.write(f"**Name:** {patient_info['name']}")
                    st.write(f"**Age:** {patient_info['age']}")
                    st.write(f"**Gender:** {patient_info['gender']}")
                    st.write(f"**Blood Type:** {patient_info['blood_type']}")
                
                with col2:
                    st.subheader("Medical History")
                    with get_db_connection() as conn:
                        history = conn.execute(
                            'SELECT condition, diagnosis_date, severity FROM medical_history WHERE patient_id = ?',
                            (st.session_state.current_patient,)
                        ).fetchall()
                    
                    if history:
                        for record in history:
                            st.write(f"• {record['condition']} ({record['diagnosis_date']}) - {record['severity']}")
                    else:
                        st.write("No medical history recorded")
                
                with col3:
                    st.subheader("Recent Symptoms")
                    with get_db_connection() as conn:
                        symptoms = conn.execute(
                            'SELECT symptoms, diagnosis, confidence, timestamp FROM symptom_records WHERE patient_id = ? ORDER BY timestamp DESC LIMIT 3',
                            (st.session_state.current_patient,)
                        ).fetchall()
                    
                    if symptoms:
                        for record in symptoms:
                            symptoms_list = json.loads(record['symptoms'])
                            st.write(f"• {record['diagnosis']} ({record['confidence']:.1f}%)")
                            st.write(f"  Symptoms: {', '.join(symptoms_list[:3])}...")
    
    with tab4:
        st.header("Vital Signs Tracker")
        
        if not st.session_state.current_patient:
            st.warning("Please register as a patient first.")
        else:
            with st.form("vital_signs"):
                st.subheader("Record Vital Signs")
                col1, col2 = st.columns(2)
                
                with col1:
                    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=72)
                    systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
                    diastolic = st.number_input("Diastolic BP", min_value=50, max_value=130, value=80)
                
                with col2:
                    temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
                    oxygen_sat = st.number_input("Oxygen Saturation (%)", min_value=80, max_value=100, value=98)
                
                if st.form_submit_button("Record Vital Signs"):
                    blood_pressure = f"{systolic}/{diastolic}"
                    patient_manager.record_vital_signs(
                        st.session_state.current_patient,
                        heart_rate,
                        blood_pressure,
                        temperature,
                        oxygen_sat
                    )
                    st.success("Vital signs recorded successfully!")
            
            # Display recent vital signs
            st.subheader("Recent Vital Signs History")
            with get_db_connection() as conn:
                vitals = conn.execute(
                    'SELECT * FROM vital_signs WHERE patient_id = ? ORDER BY timestamp DESC LIMIT 5',
                    (st.session_state.current_patient,)
                ).fetchall()
            
            if vitals:
                vital_data = []
                for vital in vitals:
                    vital_data.append({
                        'Time': vital['timestamp'],
                        'Heart Rate': vital['heart_rate'],
                        'Blood Pressure': vital['blood_pressure'],
                        'Temperature': vital['temperature'],
                        'O2 Sat': vital['oxygen_saturation']
                    })
                st.dataframe(pd.DataFrame(vital_data))
            else:
                st.info("No vital signs recorded yet")
    
    with tab5:
        st.header("🚨 Emergency Condition Check")
        
        if not st.session_state.current_patient:
            st.warning("Please register as a patient first.")
        else:
            st.warning("""
            **EMERGENCY WARNING:**
            This tool is for preliminary assessment only. If you are experiencing any of the following, 
            CALL EMERGENCY SERVICES IMMEDIATELY:
            - Chest pain or pressure
            - Difficulty breathing
            - Severe bleeding
            - Sudden weakness or numbness
            - Severe allergic reaction
            """)
            
            emergency_symptoms = st.multiselect(
                "Select emergency symptoms if present:",
                ["Chest pain", "Difficulty breathing", "Severe bleeding", "Loss of consciousness", 
                 "Sudden weakness", "Severe headache", "Seizure", "Burning sensation"]
            )
            
            if st.button("Check Emergency Status"):
                vital_signs = {
                    'heart_rate': 72,  # Default, would come from actual measurement
                    'temperature': 37.0,
                    'oxygen_saturation': 98
                }
                
                assessment = medical_ai.assess_emergency_condition(emergency_symptoms, vital_signs)
                
                if assessment['is_emergency']:
                    st.error("""
                    🚨 **EMERGENCY SITUATION DETECTED**
                    
                    **Indicators Found:**
                    {}
                    
                    **RECOMMENDATION: SEEK IMMEDIATE MEDICAL ATTENTION**
                    Call emergency services or go to the nearest hospital immediately.
                    """.format('\n'.join([f"• {indicator}" for indicator in assessment['indicators']])))
                else:
                    st.success("""
                    ✅ **No immediate emergency detected**
                    
                    However, continue to monitor your symptoms and seek medical advice if:
                    - Symptoms worsen
                    - New symptoms appear
                    - You feel increasingly unwell
                    """)

    # Footer with disclaimers
    st.markdown("---")
    st.markdown("""
    **Important Disclaimer:** 
    This AI Doctor Assistant is for informational purposes only and does not provide medical advice, 
    diagnosis, or treatment. Always seek the advice of your physician or other qualified health 
    provider with any questions you may have regarding a medical condition.
    """)

if __name__ == "__main__":
    main()
