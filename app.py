# AI_Doctor_Personal_Health_Assistant_COMPLETE.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import heapq
from typing import List, Tuple, Dict, Set
import math

# Configure Streamlit page
st.set_page_config(
    page_title="AI Doctor Helper - Complete",
    page_icon="🏥",
    layout="wide"
)

class HazardDetectionModule:
    """Hazard detection similar to planetary rover's cliff/trap detection"""
    
    def __init__(self):
        self.hazardous_conditions = {
            'critical_vitals': ['heart_attack', 'stroke', 'septic_shock'],
            'drug_interactions': ['warfarin_aspirin', 'beta_blocker_asthma'],
            'allergy_risks': ['penicillin_allergy', 'contrast_allergy'],
            'contraindications': ['pregnancy_medications', 'renal_impairment']
        }
        self.safe_state_history = []
    
    def detect_hazards(self, patient_condition: Dict, treatment: str) -> Tuple[bool, List[str]]:
        """Detect hazardous medical conditions"""
        hazards_detected = []
        
        # Check critical vitals
        if patient_condition.get('heart_rate', 0) > 150 or patient_condition.get('heart_rate', 0) < 40:
            hazards_detected.append("critical_vitals")
        
        # Check drug interactions
        current_meds = patient_condition.get('current_medications', [])
        if treatment in ['aspirin', 'ibuprofen'] and 'warfarin' in current_meds:
            hazards_detected.append("drug_interactions")
        
        # Check allergies
        allergies = patient_condition.get('allergies', [])
        if treatment in ['penicillin', 'amoxicillin'] and 'penicillin' in allergies:
            hazards_detected.append("allergy_risks")
        
        # Check age-related risks
        age = patient_condition.get('age', 40)
        if age > 65 and treatment in ['strong_medication', 'surgery']:
            hazards_detected.append("age_risk")
        
        return len(hazards_detected) > 0, hazards_detected
    
    def emergency_stop(self, current_state: Dict) -> Dict:
        """Execute emergency stop and backtrack to last safe state"""
        if self.safe_state_history:
            safe_state = self.safe_state_history[-1]
            return safe_state
        else:
            # Return minimal safe state
            return {'age': 40, 'allergies': [], 'current_medications': [], 'vitals_stable': True}
    
    def update_safe_state(self, patient_state: Dict):
        """Update safe state history"""
        self.safe_state_history.append(patient_state.copy())
        if len(self.safe_state_history) > 5:
            self.safe_state_history.pop(0)

class RiskAssessmentModule:
    """Risk assessment similar to rover's terrain cost assessment"""
    
    def __init__(self):
        self.risk_costs = {
            'low_risk': 1.0,
            'moderate_risk': 5.0,
            'high_risk': 15.0,
            'critical_risk': 1000.0
        }
        
    def calculate_treatment_risk(self, treatment_plan: List[str], patient_condition: Dict) -> float:
        """Calculate total risk cost of treatment plan"""
        total_risk = 0
        
        for treatment in treatment_plan:
            # Base risk based on treatment type
            if treatment in ['rest', 'hydrate', 'monitor']:
                base_cost = self.risk_costs['low_risk']
            elif treatment in ['medication', 'consult_doctor']:
                base_cost = self.risk_costs['moderate_risk']
            elif treatment in ['antibiotics', 'hospitalization']:
                base_cost = self.risk_costs['high_risk']
            else:
                base_cost = self.risk_costs['moderate_risk']
            
            # Adjust based on patient condition
            condition_modifier = self._get_condition_modifier(patient_condition)
            adjusted_cost = base_cost * condition_modifier
            
            total_risk += adjusted_cost
        
        return total_risk
    
    def _get_condition_modifier(self, condition: Dict) -> float:
        """Modify risk based on patient condition"""
        modifier = 1.0
        
        # Age factor
        age = condition.get('age', 40)
        if age > 65:
            modifier *= 1.5
        elif age < 18:
            modifier *= 1.3
        
        # Comorbidities
        comorbidities = condition.get('comorbidities', [])
        modifier *= (1 + len(comorbidities) * 0.2)
        
        return modifier

class AStarSymptomChecker:
    """A* Search implementation for symptom checking"""
    
    def __init__(self, symptoms_db, diseases_db):
        self.symptoms_db = symptoms_db
        self.diseases_db = diseases_db
        self.hazard_detector = HazardDetectionModule()
        self.risk_assessor = RiskAssessmentModule()
        
    def heuristic_manhattan(self, current_symptoms: Set[str], target_disease: str) -> float:
        """Heuristic 1: Manhattan distance based on symptom matching"""
        if target_disease not in self.diseases_db:
            return float('inf')
        
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        unmatched_symptoms = target_symptoms - current_symptoms
        
        return len(unmatched_symptoms)

    def heuristic_euclidean(self, current_symptoms: Set[str], target_disease: str) -> float:
        """Heuristic 2: Euclidean distance in symptom space"""
        if target_disease not in self.diseases_db:
            return float('inf')
        
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        matched = len(current_symptoms.intersection(target_symptoms))
        total_target = len(target_symptoms)
        
        return math.sqrt((total_target - matched) ** 2)

    def heuristic_symptom_frequency(self, current_symptoms: Set[str], target_disease: str) -> float:
        """Heuristic 3: Inverse of symptom frequency"""
        if target_disease not in self.diseases_db:
            return float('inf')
        
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        common_symptoms = current_symptoms.intersection(target_symptoms)
        
        frequency_score = 0
        for symptom in common_symptoms:
            disease_count = len(self.symptoms_db.get(symptom, []))
            frequency_score += 1 / (disease_count + 1)
        
        return -frequency_score if frequency_score > 0 else float('inf')

    def heuristic_severity_weighted(self, current_symptoms: Set[str], target_disease: str) -> float:
        """Heuristic 4: Weighted by disease severity"""
        if target_disease not in self.diseases_db:
            return float('inf')
        
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        unmatched_symptoms = target_symptoms - current_symptoms
        
        severity_weights = {
            'emergency': 3.0, 'high': 2.0, 'moderate': 1.5, 'low': 1.0
        }
        
        severity = self.diseases_db[target_disease].get('severity', 'moderate')
        weight = severity_weights.get(severity, 1.0)
        
        return len(unmatched_symptoms) / weight

    def heuristic_risk_aware(self, current_symptoms: Set[str], target_disease: str, patient_condition: Dict = None) -> float:
        """COMPULSORY: Risk-aware heuristic considering patient safety"""
        if target_disease not in self.diseases_db:
            return float('inf')
        
        # Get base score from severity-weighted heuristic
        base_score = self.heuristic_severity_weighted(current_symptoms, target_disease)
        
        # Initialize risk cost
        risk_cost = 0
        
        # Calculate risk cost if patient condition is provided
        if patient_condition is not None:
            treatment_plan = self.diseases_db[target_disease].get('treatment_steps', [])
            risk_cost = self.risk_assessor.calculate_treatment_risk(treatment_plan, patient_condition)
            
            # Hazard detection penalty
            is_hazardous, hazards = self.hazard_detector.detect_hazards(patient_condition, target_disease)
            if is_hazardous:
                risk_cost *= 10  # Heavy penalty for hazards
        
        return base_score + risk_cost

    def a_star_search(self, selected_symptoms: List[str], heuristic_name: str, patient_condition: Dict = None) -> List[Tuple]:
        """A* search implementation that handles all heuristics"""
        current_symptoms_set = set(selected_symptoms)
        possible_diseases = set()
        
        # Get all possible diseases from selected symptoms
        for symptom in selected_symptoms:
            possible_diseases.update(self.symptoms_db.get(symptom, []))
        
        # Map heuristic names to functions
        heuristic_functions = {
            "Manhattan": self.heuristic_manhattan,
            "Euclidean": self.heuristic_euclidean,
            "Symptom Frequency": self.heuristic_symptom_frequency,
            "Severity Weighted": self.heuristic_severity_weighted,
            "Risk-Aware": self.heuristic_risk_aware
        }
        
        heuristic_func = heuristic_functions[heuristic_name]
        
        # Priority queue for A* search
        priority_queue = []
        
        for disease in possible_diseases:
            # g(n) = number of unmatched symptoms from selected ones
            target_symptoms = set(self.diseases_db.get(disease, {}).get('symptoms', []))
            g_score = len(current_symptoms_set - target_symptoms)
            
            # h(n) = heuristic estimate
            if heuristic_name == "Risk-Aware" and patient_condition is not None:
                h_score = heuristic_func(current_symptoms_set, disease, patient_condition)
            else:
                h_score = heuristic_func(current_symptoms_set, disease)
            
            # f(n) = g(n) + h(n)
            f_score = g_score + h_score
            
            heapq.heappush(priority_queue, (f_score, disease, g_score, h_score))
        
        # Return sorted results
        results = []
        while priority_queue:
            f_score, disease, g_score, h_score = heapq.heappop(priority_queue)
            confidence = max(0, 100 - f_score * 5)  # Convert to percentage
            results.append((disease, confidence, g_score, h_score, f_score))
        
        return results

class ReflexMedicationAgent:
    """Simple reflex agent for medication management"""
    
    def __init__(self):
        self.medication_schedule = {}
        self.adherence_history = {}
        self.hazard_detector = HazardDetectionModule()
    
    def add_medication(self, name: str, dosage: str, times: List[str], patient_condition: Dict):
        """Add medication to schedule with safety check"""
        # Check for hazards
        is_hazardous, hazards = self.hazard_detector.detect_hazards(patient_condition, name)
        
        if is_hazardous:
            return False, f"Hazard detected: {hazards}"
        else:
            self.medication_schedule[name] = {
                'dosage': dosage,
                'times': times,
                'last_taken': None,
                'adherence': []
            }
            self.hazard_detector.update_safe_state(patient_condition)
            return True, "Medication added safely"
    
    def check_medication_time(self, patient_condition: Dict = None) -> List[str]:
        """Reflex agent - condition-action rules"""
        current_time = datetime.now()
        reminders = []
        
        # Check for hazards if patient condition provided
        if patient_condition:
            is_hazardous, hazards = self.hazard_detector.detect_hazards(patient_condition, "medication_check")
            if is_hazardous:
                return [f"EMERGENCY: Hazards detected - {hazards}. Medication check suspended."]
        
        for med, info in self.medication_schedule.items():
            for time_str in info['times']:
                try:
                    med_time = datetime.strptime(time_str, "%H:%M").time()
                    current_time_only = current_time.time()
                    
                    # Check if it's medication time (within 30 minutes)
                    time_diff = abs((current_time_only.hour - med_time.hour) * 60 + 
                                  (current_time_only.minute - med_time.minute))
                    
                    if time_diff <= 30 and info['last_taken'] != current_time.date():
                        reminders.append(f"Time to take {med} - {info['dosage']}")
                except ValueError:
                    continue
        
        return reminders
    
    def mark_taken(self, medication: str):
        """Mark medication as taken"""
        if medication in self.medication_schedule:
            self.medication_schedule[medication]['last_taken'] = datetime.now().date()

class PathPlanningModule:
    """Path planning for treatment recommendation sequence"""
    
    def __init__(self, diseases_db):
        self.diseases_db = diseases_db
        self.hazard_detector = HazardDetectionModule()
    
    def plan_treatment_path(self, disease: str, patient_condition: Dict = None, current_step: str = None) -> List[Tuple[str, float]]:
        """Plan optimal treatment path"""
        if disease not in self.diseases_db:
            return []
        
        treatment_steps = self.diseases_db[disease].get('treatment_steps', [])
        
        # Add costs based on step complexity
        steps_with_costs = []
        for i, step in enumerate(treatment_steps):
            cost = (i + 1) * 5  # Basic cost model
            steps_with_costs.append((step, cost))
        
        # Filter hazardous steps if patient condition provided
        if patient_condition:
            safe_steps = []
            for step, cost in steps_with_costs:
                is_hazardous, hazards = self.hazard_detector.detect_hazards(patient_condition, step)
                if not is_hazardous:
                    safe_steps.append((step, cost))
                else:
                    safe_steps.append((f"AVOID: {step} (hazard)", cost * 10))
            steps_with_costs = safe_steps
        
        if current_step:
            # Find current step and return remaining path
            current_index = next((i for i, (step, _) in enumerate(steps_with_costs) 
                               if step == current_step), 0)
            return steps_with_costs[current_index:]
        
        return steps_with_costs

class AIHealthcareAssistant:
    def __init__(self):
        self.symptoms_db = self._initialize_symptoms_database()
        self.diseases_db = self._initialize_diseases_database()
        self.astar_checker = AStarSymptomChecker(self.symptoms_db, self.diseases_db)
        self.reflex_agent = ReflexMedicationAgent()
        self.path_planner = PathPlanningModule(self.diseases_db)
        
    def _initialize_symptoms_database(self):
        """Symptom-Disease relationships"""
        return {
            'fever': ['flu', 'covid', 'pneumonia'],
            'cough': ['flu', 'covid', 'pneumonia', 'bronchitis'],
            'headache': ['flu', 'migraine', 'covid'],
            'fatigue': ['flu', 'covid', 'anemia'],
            'chest_pain': ['heart_attack', 'pneumonia'],
            'shortness_of_breath': ['covid', 'pneumonia', 'asthma'],
            'nausea': ['food_poisoning', 'migraine'],
            'vomiting': ['food_poisoning', 'migraine'],
            'muscle_pain': ['flu', 'covid'],
            'sore_throat': ['flu', 'covid']
        }
    
    def _initialize_diseases_database(self):
        """Knowledge Base"""
        return {
            'flu': {
                'symptoms': ['fever', 'cough', 'headache', 'fatigue', 'muscle_pain', 'sore_throat'],
                'severity': 'moderate',
                'recommendation': 'Rest, hydrate, take antiviral medication if prescribed',
                'treatment_steps': ['rest', 'hydrate', 'medication', 'monitor']
            },
            'covid': {
                'symptoms': ['fever', 'cough', 'fatigue', 'shortness_of_breath', 'muscle_pain'],
                'severity': 'high',
                'recommendation': 'Isolate, get tested, consult doctor immediately',
                'treatment_steps': ['isolate', 'test', 'consult_doctor', 'monitor_symptoms']
            },
            'pneumonia': {
                'symptoms': ['fever', 'cough', 'chest_pain', 'shortness_of_breath'],
                'severity': 'high',
                'recommendation': 'Emergency care required, antibiotics may be needed',
                'treatment_steps': ['emergency_care', 'antibiotics', 'hospitalization']
            },
            'heart_attack': {
                'symptoms': ['chest_pain', 'shortness_of_breath'],
                'severity': 'emergency',
                'recommendation': 'CALL EMERGENCY SERVICES IMMEDIATELY',
                'treatment_steps': ['call_emergency', 'chew_aspirin', 'hospitalization']
            },
            'migraine': {
                'symptoms': ['headache', 'nausea'],
                'severity': 'moderate',
                'recommendation': 'Rest in dark room, stay hydrated, consider pain relief',
                'treatment_steps': ['dark_room', 'hydration', 'pain_relief', 'rest']
            },
            'food_poisoning': {
                'symptoms': ['nausea', 'vomiting'],
                'severity': 'moderate',
                'recommendation': 'Stay hydrated, rest, avoid solid foods initially',
                'treatment_steps': ['hydrate', 'rest', 'avoid_solids', 'monitor']
            }
        }

def test_risk_aware_heuristic():
    """Test function for Risk-Aware heuristic"""
    st.header("🧪 Risk-Aware Heuristic Test")
    
    # Create test instance
    symptoms_db = {'fever': ['flu', 'covid'], 'cough': ['flu', 'covid']}
    diseases_db = {
        'flu': {'symptoms': ['fever', 'cough'], 'severity': 'moderate', 'treatment_steps': ['rest']},
        'covid': {'symptoms': ['fever', 'cough'], 'severity': 'high', 'treatment_steps': ['hospitalization']}
    }
    
    astar = AStarSymptomChecker(symptoms_db, diseases_db)
    
    # Test cases
    test_symptoms = ['fever', 'cough']
    
    # Test 1: Young healthy patient
    young_patient = {'age': 25, 'allergies': [], 'current_medications': []}
    
    # Test 2: Elderly patient with comorbidities
    elderly_patient = {'age': 75, 'allergies': ['penicillin'], 'current_medications': ['warfarin'], 'comorbidities': ['diabetes']}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Young Healthy Patient")
        results_young = astar.a_star_search(test_symptoms, "Risk-Aware", young_patient)
        for disease, confidence, g, h, f in results_young:
            st.write(f"**{disease}**: {confidence:.1f}% (f-score: {f:.2f})")
    
    with col2:
        st.subheader("Elderly Patient with Risks")
        results_elderly = astar.a_star_search(test_symptoms, "Risk-Aware", elderly_patient)
        for disease, confidence, g, h, f in results_elderly:
            st.write(f"**{disease}**: {confidence:.1f}% (f-score: {f:.2f})")
    
    # Show the difference
    st.subheader("Comparison")
    if results_young and results_elderly:
        young_flu_score = next((f for d, _, _, _, f in results_young if d == 'flu'), 0)
        elderly_flu_score = next((f for d, _, _, _, f in results_elderly if d == 'flu'), 0)
        
        st.write(f"Flu f-score for young patient: {young_flu_score:.2f}")
        st.write(f"Flu f-score for elderly patient: {elderly_flu_score:.2f}")
        st.write(f"Risk penalty: {elderly_flu_score - young_flu_score:.2f}")

def main():
    st.title("🏥 AI Doctor Helper - Complete Implementation")
    st.markdown("""
    **Complete AI Curriculum Implementation with COMPULSORY Risk-Aware Heuristic**
    
    ✅ **A* Search with 5 Heuristics** (Including COMPULSORY Risk-Aware)  
    ✅ **Reflex Agent with Hazard Detection**  
    ✅ **Path Planning with Risk Assessment**  
    ✅ **Planetary Rover-inspired Safety Features**
    """)
    
    # Initialize assistant
    assistant = AIHealthcareAssistant()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Feature",
        ["Symptom Checker (A*)", "Medication Agent", "Treatment Planning", "Heuristic Test", "All Heuristics Comparison"]
    )
    
    if app_mode == "Symptom Checker (A*)":
        st.header("🔍 Symptom Checker with A* Search")
        
        # Symptoms selection
        symptoms = list(assistant.symptoms_db.keys())
        selected_symptoms = st.multiselect("Select your symptoms:", symptoms)
        
        if selected_symptoms:
            # Patient information for Risk-Aware heuristic
            st.subheader("Patient Information (Required for Risk-Aware)")
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=40)
                comorbidities = st.multiselect("Comorbidities", ["diabetes", "hypertension", "asthma", "heart_disease", "none"])
            
            with col2:
                current_meds = st.multiselect("Current Medications", ["warfarin", "insulin", "aspirin", "none"])
                allergies = st.multiselect("Allergies", ["penicillin", "aspirin", "sulfa", "none"])
            
            patient_condition = {
                'age': age,
                'comorbidities': [c for c in comorbidities if c != "none"],
                'current_medications': [m for m in current_meds if m != "none"],
                'allergies': [a for a in allergies if a != "none"]
            }
            
            # Heuristic selection
            heuristic_choice = st.selectbox(
                "Select A* Heuristic:",
                ["Manhattan", "Euclidean", "Symptom Frequency", "Severity Weighted", "Risk-Aware"]
            )
            
            if st.button("Run Diagnosis"):
                st.subheader(f"Results using {heuristic_choice} Heuristic")
                
                # Run A* search
                results = assistant.astar_checker.a_star_search(
                    selected_symptoms, 
                    heuristic_choice, 
                    patient_condition if heuristic_choice == "Risk-Aware" else None
                )
                
                if results:
                    for disease, confidence, g_score, h_score, f_score in results[:5]:
                        if confidence > 10:  # Filter very low confidence results
                            disease_info = assistant.diseases_db.get(disease, {})
                            
                            st.write(f"### {disease} ({confidence:.1f}%)")
                            st.write(f"**Scores**: g(n)={g_score}, h(n)={h_score:.2f}, f(n)={f_score:.2f}")
                            
                            # Hazard detection
                            if heuristic_choice == "Risk-Aware":
                                is_hazardous, hazards = assistant.astar_checker.hazard_detector.detect_hazards(patient_condition, disease)
                                if is_hazardous:
                                    st.error(f"🚨 **Hazards Detected**: {', '.join(hazards)}")
                            
                            st.write(f"**Recommendation**: {disease_info.get('recommendation', 'Consult healthcare provider')}")
                            st.write("---")
                else:
                    st.warning("No matching conditions found. Please check your symptoms or consult a doctor.")
    
    elif app_mode == "Medication Agent":
        st.header("💊 Reflex Medication Agent")
        
        # Patient info
        st.subheader("Patient Profile")
        allergies = st.multiselect("Allergies", ["penicillin", "aspirin", "sulfa", "none"])
        current_meds = st.multiselect("Current Medications", ["warfarin", "insulin", "aspirin", "none"])
        
        patient_condition = {
            'allergies': [a for a in allergies if a != "none"],
            'current_medications': [m for m in current_meds if m != "none"]
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Add New Medication")
            med_name = st.text_input("Medication Name")
            dosage = st.text_input("Dosage")
            times = st.multiselect("Schedule Times", ["08:00", "12:00", "18:00", "22:00"])
            
            if st.button("Add Medication"):
                if med_name and dosage and times:
                    success, message = assistant.reflex_agent.add_medication(med_name, dosage, times, patient_condition)
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
        
        with col2:
            st.subheader("Current Reminders")
            reminders = assistant.reflex_agent.check_medication_time(patient_condition)
            
            if reminders:
                for reminder in reminders:
                    if "EMERGENCY" in reminder:
                        st.error(reminder)
                    else:
                        st.warning(reminder)
                        med_name = reminder.split(" - ")[0].replace("Time to take ", "")
                        if st.button(f"Mark {med_name} as Taken", key=med_name):
                            assistant.reflex_agent.mark_taken(med_name)
                            st.rerun()
            else:
                st.info("No medication reminders at this time")
    
    elif app_mode == "Treatment Planning":
        st.header("🛣️ Treatment Path Planning")
        
        disease = st.selectbox("Select Condition", list(assistant.diseases_db.keys()))
        
        if disease:
            # Patient info for risk assessment
            st.subheader("Patient Risk Profile")
            age = st.slider("Age", 0, 100, 45)
            risk_factors = st.multiselect("Risk Factors", ["smoker", "obese", "sedentary", "family_history"])
            
            patient_condition = {
                'age': age,
                'comorbidities': risk_factors
            }
            
            # Get treatment path
            treatment_path = assistant.path_planner.plan_treatment_path(disease, patient_condition)
            
            st.subheader(f"Treatment Plan for {disease}")
            
            total_risk = assistant.astar_checker.risk_assessor.calculate_treatment_risk(
                [step for step, cost in treatment_path], 
                patient_condition
            )
            
            st.write(f"**Overall Risk Score**: {total_risk:.1f}")
            
            for i, (step, cost) in enumerate(treatment_path):
                if "AVOID" in step:
                    st.error(f"{i+1}. {step} (Cost: {cost})")
                else:
                    st.success(f"{i+1}. {step.replace('_', ' ').title()} (Cost: {cost})")
    
    elif app_mode == "Heuristic Test":
        test_risk_aware_heuristic()
    
    elif app_mode == "All Heuristics Comparison":
        st.header("📊 All Heuristics Comparison")
        
        # Test setup
        test_symptoms = ['fever', 'cough', 'headache']
        test_patient = {'age': 65, 'comorbidities': ['diabetes'], 'current_medications': ['warfarin']}
        
        heuristics = ["Manhattan", "Euclidean", "Symptom Frequency", "Severity Weighted", "Risk-Aware"]
        
        comparison_data = []
        
        for heuristic in heuristics:
            results = assistant.astar_checker.a_star_search(
                test_symptoms, 
                heuristic, 
                test_patient if heuristic == "Risk-Aware" else None
            )
            
            if results:
                top_disease, confidence, g, h, f = results[0]
                comparison_data.append({
                    'Heuristic': heuristic,
                    'Top Disease': top_disease,
                    'Confidence': f"{confidence:.1f}%",
                    'f-score': f"{f:.2f}",
                    'g-score': g,
                    'h-score': f"{h:.2f}"
                })
        
        # Display comparison table
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.table(df)
            
            # Highlight differences
            st.subheader("Key Observations")
            risk_aware_result = next((item for item in comparison_data if item['Heuristic'] == "Risk-Aware"), None)
            if risk_aware_result:
                st.info(f"**Risk-Aware Heuristic**: Prioritizes safety with patient-specific risk factors. Top result: {risk_aware_result['Top Disease']} with {risk_aware_result['Confidence']} confidence.")

if __name__ == "__main__":
    main()
