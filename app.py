# AI_Doctor_Personal_Health_Assistant_Enhanced.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import heapq
from typing import List, Tuple, Dict, Set
import math

# Try to import matplotlib with error handling
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib is not installed. Some visualizations will be disabled.")

# Try to import seaborn with error handling
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

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
        self.current_risk_level = "low"
    
    def detect_hazards(self, current_condition: Dict, planned_treatment: str) -> Tuple[bool, str]:
        """Detect hazardous conditions like rover detecting cliffs"""
        hazards_detected = []
        
        # Check critical vitals (like rover checking battery)
        if current_condition.get('heart_rate', 0) > 150 or current_condition.get('heart_rate', 0) < 40:
            hazards_detected.append("critical_vitals")
        
        # Check drug interactions
        current_meds = current_condition.get('current_medications', [])
        if planned_treatment in ['aspirin', 'ibuprofen'] and 'warfarin' in current_meds:
            hazards_detected.append("drug_interactions")
        
        # Check allergies
        allergies = current_condition.get('allergies', [])
        if planned_treatment in ['penicillin', 'amoxicillin'] and 'penicillin_allergy' in allergies:
            hazards_detected.append("allergy_risks")
        
        return len(hazards_detected) > 0, hazards_detected
    
    def emergency_stop(self, current_state: Dict) -> Dict:
        """Execute emergency stop and backtrack to last safe state"""
        if self.safe_state_history:
            safe_state = self.safe_state_history[-1]
            st.error("🚨 HAZARDOUS CONDITION DETECTED! Backtracking to last safe state.")
            return safe_state
        else:
            st.error("🚨 CRITICAL EMERGENCY! No safe state found. Initiating emergency protocol.")
            return self._get_emergency_protocol()
    
    def update_safe_state(self, patient_state: Dict):
        """Update safe state history (like rover's known safe cells)"""
        self.safe_state_history.append(patient_state.copy())
        # Keep only last 10 safe states to prevent memory issues
        if len(self.safe_state_history) > 10:
            self.safe_state_history.pop(0)

class RiskAssessmentModule:
    """Risk assessment similar to rover's terrain cost assessment"""
    
    def __init__(self):
        self.risk_costs = {
            'low_risk': 1.0,
            'moderate_risk': 5.0,
            'high_risk': 15.0,
            'critical_risk': 1000.0  # Like rocky terrain for rover
        }
        
        self.treatment_risks = {
            'routine_checkup': 'low_risk',
            'basic_medication': 'low_risk',
            'antibiotics': 'moderate_risk',
            'surgery_prep': 'high_risk',
            'emergency_intervention': 'critical_risk'
        }
    
    def calculate_treatment_risk(self, treatment_plan: List[str], patient_condition: Dict) -> float:
        """Calculate total risk cost of treatment plan"""
        total_risk = 0
        
        for treatment in treatment_plan:
            risk_level = self.treatment_risks.get(treatment, 'moderate_risk')
            base_cost = self.risk_costs[risk_level]
            
            # Adjust based on patient condition (like rover's battery level)
            condition_modifier = self._get_condition_modifier(patient_condition)
            adjusted_cost = base_cost * condition_modifier
            
            total_risk += adjusted_cost
        
        return total_risk
    
    def _get_condition_modifier(self, condition: Dict) -> float:
        """Modify risk based on patient condition (similar to rover's battery affecting movement cost)"""
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
        
        # Vital stability
        if condition.get('vitals_stable', True) == False:
            modifier *= 2.0
        
        return modifier

class AStarSymptomChecker:
    """A* Search implementation for symptom checking with multiple heuristics"""
    
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
        matched_symptoms = current_symptoms.intersection(target_symptoms)
        unmatched_symptoms = target_symptoms - current_symptoms
        
        return len(unmatched_symptoms)  # Lower is better
    
    def heuristic_euclidean(self, current_symptoms: Set[str], target_disease: str) -> float:
        """Heuristic 2: Euclidean distance in symptom space"""
        if target_disease not in self.diseases_db:
            return float('inf')
        
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        matched = len(current_symptoms.intersection(target_symptoms))
        total_target = len(target_symptoms)
        
        # Euclidean distance: sqrt((total_target - matched)^2)
        return math.sqrt((total_target - matched) ** 2)
    
    def heuristic_symptom_frequency(self, current_symptoms: Set[str], target_disease: str) -> float:
        """Heuristic 3: Inverse of symptom frequency (rare symptoms are more significant)"""
        if target_disease not in self.diseases_db:
            return float('inf')
        
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        common_symptoms = current_symptoms.intersection(target_symptoms)
        
        # Calculate frequency score (lower for rare symptoms)
        frequency_score = 0
        for symptom in common_symptoms:
            # More diseases having this symptom = more common = less significant
            disease_count = len(self.symptoms_db.get(symptom, []))
            frequency_score += 1 / (disease_count + 1)  # +1 to avoid division by zero
        
        # We want to maximize rare symptom matches, so return negative
        return -frequency_score if frequency_score > 0 else float('inf')
    
    def heuristic_severity_weighted(self, current_symptoms: Set[str], target_disease: str) -> float:
        """Heuristic 4: Weighted by disease severity"""
        if target_disease not in self.diseases_db:
            return float('inf')
        
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        unmatched_symptoms = target_symptoms - current_symptoms
        
        # Severity weights
        severity_weights = {
            'emergency': 3.0,
            'high': 2.0,
            'moderate': 1.5,
            'low': 1.0
        }
        
        severity = self.diseases_db[target_disease].get('severity', 'moderate')
        weight = severity_weights.get(severity, 1.0)
        
        return len(unmatched_symptoms) / weight
    
    def heuristic_risk_aware(self, current_symptoms: Set[str], target_disease: str, patient_condition: Dict = None) -> float:
        """Risk-aware heuristic considering patient safety"""
        # Base heuristic from severity
        base_score = self.heuristic_severity_weighted(current_symptoms, target_disease)
        
        # Risk adjustment based on treatment complexity
        treatment_plan = self.diseases_db[target_disease].get('treatment_steps', [])
        risk_cost = self.risk_assessor.calculate_treatment_risk(treatment_plan, patient_condition)
        
        # Hazard detection penalty
        is_hazardous, hazards = self.hazard_detector.detect_hazards(patient_condition, target_disease)
        if is_hazardous:
            risk_cost *= 10  # Heavy penalty for hazards
        
        return base_score + risk_cost
    
    def a_star_search(self, selected_symptoms: List[str], heuristic_func: callable, patient_condition: Dict = None) -> List[Tuple[str, float]]:
        """Enhanced A* search with hazard detection"""
        current_symptoms_set = set(selected_symptoms)
        possible_diseases = set()
        
        # Get all possible diseases from selected symptoms
        for symptom in selected_symptoms:
            possible_diseases.update(self.symptoms_db.get(symptom, []))
        
        # Priority queue for A* search
        priority_queue = []
        
        for disease in possible_diseases:
            # Hazard detection check
            if patient_condition:
                is_hazardous, hazards = self.hazard_detector.detect_hazards(patient_condition, disease)
                if is_hazardous:
                    continue  # Skip hazardous paths entirely
            
            # g(n) = number of unmatched symptoms from selected ones
            target_symptoms = set(self.diseases_db.get(disease, {}).get('symptoms', []))
            g_score = len(current_symptoms_set - target_symptoms)
            
            # h(n) = heuristic estimate
            if patient_condition and heuristic_func.__name__ == 'heuristic_risk_aware':
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
            confidence = max(0, 100 - f_score * 10)  # Convert to percentage
            results.append((disease, confidence, g_score, h_score, f_score))
        
        return results

class EnhancedReflexMedicationAgent:
    """Enhanced reflex agent with emergency stop and backtracking"""
    
    def __init__(self):
        self.medication_schedule = {}
        self.adherence_history = {}
        self.hazard_detector = HazardDetectionModule()
        self.emergency_mode = False
        
    def add_medication(self, name: str, dosage: str, times: List[str], patient_condition: Dict):
        """Add medication with safety check"""
        # Check for hazards before adding
        is_hazardous, hazards = self.hazard_detector.detect_hazards(patient_condition, name)
        
        if is_hazardous:
            st.error(f"🚨 Cannot add {name}: Hazard detected - {hazards}")
            return False
        else:
            self.medication_schedule[name] = {
                'dosage': dosage,
                'times': times,
                'last_taken': None,
                'adherence': []
            }
            # Update safe state
            self.hazard_detector.update_safe_state(patient_condition)
            return True
    
    def check_medication_time(self, current_patient_state: Dict) -> List[str]:
        """Enhanced reflex agent with hazard monitoring"""
        if self.emergency_mode:
            return ["EMERGENCY MODE: Medication check suspended"]
            
        current_time = datetime.now()
        reminders = []
        
        # Monitor for hazards continuously
        is_hazardous, hazards = self.hazard_detector.detect_hazards(current_patient_state, "medication_check")
        if is_hazardous:
            self.emergency_mode = True
            safe_state = self.hazard_detector.emergency_stop(current_patient_state)
            return [f"EMERGENCY STOP: {hazards} detected. Reverted to safe state."]
        
        for med, info in self.medication_schedule.items():
            for time_str in info['times']:
                med_time = datetime.strptime(time_str, "%H:%M").time()
                current_time_only = current_time.time()
                
                # Check if it's medication time (within 30 minutes)
                time_diff = abs((current_time_only.hour - med_time.hour) * 60 + 
                              (current_time_only.minute - med_time.minute))
                
                if time_diff <= 30:
                    # Condition: medication due and not taken
                    if info['last_taken'] != current_time.date():
                        # Action: send reminder
                        reminders.append(f"Time to take {med} - {info['dosage']}")
                        
                        # Record adherence
                        if med not in self.adherence_history:
                            self.adherence_history[med] = []
                        self.adherence_history[med].append({
                            'timestamp': current_time,
                            'taken': False,
                            'reminded': True
                        })
        
        return reminders
    
    def mark_taken(self, medication: str):
        """Mark medication as taken"""
        if medication in self.medication_schedule:
            self.medication_schedule[medication]['last_taken'] = datetime.now().date()
            
            # Update adherence history
            if medication in self.adherence_history:
                for record in reversed(self.adherence_history[medication]):
                    if not record['taken']:
                        record['taken'] = True
                        record['actual_time'] = datetime.now()
                        break

class EnhancedPathPlanningModule:
    """Enhanced path planning with hazard avoidance"""
    
    def __init__(self, diseases_db):
        self.diseases_db = diseases_db
        self.treatment_graph = self._build_treatment_graph()
        self.hazard_detector = HazardDetectionModule()
        self.risk_assessor = RiskAssessmentModule()
    
    def _build_treatment_graph(self) -> Dict[str, List[Tuple[str, float]]]:
        """Build graph of treatment steps with costs and hazards"""
        graph = {}
        
        # Define treatment steps for different conditions with risk costs
        treatment_steps = {
            'flu': [('rest', 5), ('hydrate', 5), ('medication', 10), ('monitor', 5)],
            'covid': [('isolate', 5), ('test', 15), ('consult_doctor', 10), ('monitor_symptoms', 5)],
            'pneumonia': [('emergency_care', 15), ('antibiotics', 15), ('hospitalization', 1000), ('followup', 10)],
            'heart_attack': [('call_emergency', 1000), ('chew_aspirin', 1000), ('hospitalization', 1000)],
            'migraine': [('dark_room', 5), ('hydration', 5), ('pain_relief', 10), ('rest', 5)],
            'food_poisoning': [('hydrate', 5), ('rest', 5), ('avoid_solids', 5), ('monitor', 5)]
        }
        
        # Add hazardous paths
        hazardous_treatments = {
            'pneumonia': [('self_treat', 1000)],  # Like rover's cliff
            'heart_attack': [('delay_treatment', 1000)],
            'covid': [('ignore_symptoms', 1000)]
        }
        
        for disease, steps in treatment_steps.items():
            graph[disease] = steps
            if disease in hazardous_treatments:
                graph[disease].extend(hazardous_treatments[disease])
        
        return graph
    
    def plan_treatment_path(self, disease: str, patient_condition: Dict, current_step: str = None) -> List[Tuple[str, float]]:
        """Plan optimal treatment path avoiding hazards"""
        if disease not in self.treatment_graph:
            return []
        
        treatment_steps = self.treatment_graph[disease]
        safe_path = []
        
        # Filter out hazardous steps based on patient condition
        for step, cost in treatment_steps:
            is_hazardous, hazards = self.hazard_detector.detect_hazards(patient_condition, step)
            if not is_hazardous:
                safe_path.append((step, cost))
            else:
                st.warning(f"⚠️ Hazard detected for step '{step}': {hazards}")
        
        if current_step:
            # Find current step and return remaining safe path
            current_index = next((i for i, (step, _) in enumerate(safe_path) 
                               if step == current_step), 0)
            return safe_path[current_index:]
        
        return safe_path
    
    def emergency_reroute(self, disease: str, hazard_type: str, patient_condition: Dict) -> List[Tuple[str, float]]:
        """Emergency rerouting when hazards are detected"""
        st.error(f"🚨 Emergency reroute initiated due to: {hazard_type}")
        
        # Default emergency protocol
        emergency_protocol = [
            ('stop_current_treatment', 1000),
            ('assess_patient', 1000),
            ('implement_emergency_care', 1000),
            ('consult_specialist', 1000)
        ]
        
        return emergency_protocol

class AIHealthcareAssistant:
    def __init__(self):
        self.symptoms_db = self._initialize_symptoms_database()
        self.diseases_db = self._initialize_diseases_database()
        self.patients_history = {}
        self.astar_checker = AStarSymptomChecker(self.symptoms_db, self.diseases_db)
        self.reflex_agent = EnhancedReflexMedicationAgent()
        self.path_planner = EnhancedPathPlanningModule(self.diseases_db)
        self.hazard_detector = HazardDetectionModule()
        self.risk_assessor = RiskAssessmentModule()
        
    def _initialize_symptoms_database(self):
        """Enhanced symptom database with hazard indicators"""
        return {
            'fever': ['flu', 'covid', 'pneumonia', 'malaria'],
            'cough': ['flu', 'covid', 'pneumonia', 'bronchitis'],
            'headache': ['flu', 'migraine', 'covid', 'hypertension'],
            'fatigue': ['flu', 'covid', 'anemia', 'depression'],
            'chest_pain': ['heart_attack', 'pneumonia', 'angina'],
            'shortness_of_breath': ['covid', 'pneumonia', 'asthma', 'heart_attack'],
            'nausea': ['food_poisoning', 'migraine', 'pregnancy'],
            'vomiting': ['food_poisoning', 'migraine', 'gastroenteritis'],
            'muscle_pain': ['flu', 'covid', 'fibromyalgia'],
            'sore_throat': ['flu', 'covid', 'strep_throat'],
            'sudden_numbness': ['stroke', 'neurological_emergency'],  # Hazard indicator
            'severe_bleeding': ['trauma', 'hemorrhage'],  # Hazard indicator
            'loss_of_consciousness': ['stroke', 'heart_attack']  # Hazard indicator
        }
    
    def _initialize_diseases_database(self):
        """Enhanced knowledge base with hazard information"""
        return {
            'flu': {
                'symptoms': ['fever', 'cough', 'headache', 'fatigue', 'muscle_pain', 'sore_throat'],
                'severity': 'moderate',
                'recommendation': 'Rest, hydrate, take antiviral medication if prescribed',
                'treatment_steps': ['rest', 'hydrate', 'medication', 'monitor'],
                'hazards': ['dehydration', 'secondary_infection']
            },
            'covid': {
                'symptoms': ['fever', 'cough', 'fatigue', 'shortness_of_breath', 'muscle_pain'],
                'severity': 'high',
                'recommendation': 'Isolate, get tested, consult doctor immediately',
                'treatment_steps': ['isolate', 'test', 'consult_doctor', 'monitor_symptoms'],
                'hazards': ['pneumonia', 'long_covid', 'thrombosis']
            },
            'pneumonia': {
                'symptoms': ['fever', 'cough', 'chest_pain', 'shortness_of_breath'],
                'severity': 'high',
                'recommendation': 'Emergency care required, antibiotics may be needed',
                'treatment_steps': ['emergency_care', 'antibiotics', 'hospitalization', 'followup'],
                'hazards': ['respiratory_failure', 'sepsis']
            },
            'heart_attack': {
                'symptoms': ['chest_pain', 'shortness_of_breath', 'sudden_numbness'],
                'severity': 'emergency',
                'recommendation': 'CALL EMERGENCY SERVICES IMMEDIATELY',
                'treatment_steps': ['call_emergency', 'chew_aspirin', 'hospitalization'],
                'hazards': ['cardiac_arrest', 'death']
            },
            'migraine': {
                'symptoms': ['headache', 'nausea'],
                'severity': 'moderate',
                'recommendation': 'Rest in dark room, stay hydrated, consider pain relief',
                'treatment_steps': ['dark_room', 'hydration', 'pain_relief', 'rest'],
                'hazards': ['status_migrainosus', 'medication_overuse']
            },
            'food_poisoning': {
                'symptoms': ['nausea', 'vomiting'],
                'severity': 'moderate',
                'recommendation': 'Stay hydrated, rest, avoid solid foods initially',
                'treatment_steps': ['hydrate', 'rest', 'avoid_solids', 'monitor'],
                'hazards': ['dehydration', 'electrolyte_imbalance']
            },
            'stroke': {
                'symptoms': ['sudden_numbness', 'loss_of_consciousness', 'headache'],
                'severity': 'emergency',
                'recommendation': 'CALL EMERGENCY SERVICES IMMEDIATELY - Time is brain!',
                'treatment_steps': ['call_emergency', 'hospitalization', 'rehabilitation'],
                'hazards': ['permanent_disability', 'death']
            }
        }

class EnhancedHealthMonitor:
    def __init__(self):
        self.vital_signs_history = []
        self.hazard_detector = HazardDetectionModule()
        self.alert_thresholds = {
            'heart_rate': (60, 100),
            'bp_systolic': (90, 140),
            'bp_diastolic': (60, 90),
            'temperature': (36.1, 37.8),
            'spo2': (95, 100)
        }
    
    def add_vital_signs(self, heart_rate, bp_systolic, bp_diastolic, temperature, spo2):
        """Enhanced monitoring with hazard detection"""
        timestamp = datetime.now()
        vital_data = {
            'timestamp': timestamp,
            'heart_rate': heart_rate,
            'bp_systolic': bp_systolic,
            'bp_diastolic': bp_diastolic,
            'temperature': temperature,
            'spo2': spo2
        }
        
        self.vital_signs_history.append(vital_data)
        
        # Check for critical values (like rover's low battery)
        alerts = self._check_critical_values(vital_data)
        return alerts
    
    def _check_critical_values(self, vital_data: Dict) -> List[str]:
        """Check for critical values that require emergency stop"""
        alerts = []
        
        if vital_data['heart_rate'] < 40 or vital_data['heart_rate'] > 150:
            alerts.append(f"CRITICAL: Heart rate {vital_data['heart_rate']} bpm")
        
        if vital_data['spo2'] < 90:
            alerts.append(f"CRITICAL: Oxygen saturation {vital_data['spo2']}%")
        
        if vital_data['temperature'] > 39.0:
            alerts.append(f"CRITICAL: High fever {vital_data['temperature']}°C")
        
        return alerts
    
    def analyze_trends(self):
        """Enhanced trend analysis with hazard prediction"""
        if len(self.vital_signs_history) < 2:
            return ["Insufficient data for trend analysis"]
        
        df = pd.DataFrame(self.vital_signs_history)
        trends = []
        warnings = []
        
        # Simple trend analysis
        hr_trend = "stable"
        if len(df) > 1:
            hr_change = df['heart_rate'].iloc[-1] - df['heart_rate'].iloc[-2]
            if abs(hr_change) > 10:
                hr_trend = "increasing" if hr_change > 0 else "decreasing"
                if abs(hr_change) > 20:
                    warnings.append("Rapid heart rate change detected")
        
        trends.append(f"Heart rate trend: {hr_trend}")
        
        # Check for deteriorating trends (like rover detecting approaching cliff)
        if len(df) >= 3:
            recent_spo2 = df['spo2'].tail(3)
            if all(recent_spo2.diff().dropna() < 0):  # Continuously decreasing
                warnings.append("Oxygen saturation continuously decreasing - Potential hazard")
        
        return trends + (["⚠️ " + warning for warning in warnings] if warnings else [])

def demonstrate_hazard_detection():
    """Demo function to show hazard detection and backtracking"""
    st.header("🚨 Hazard Detection & Emergency Response")
    
    # Simulate patient scenario
    st.subheader("Simulated Patient Scenario")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Initial Safe State:**")
        safe_state = {
            'symptoms': ['headache', 'fever'],
            'heart_rate': 85,
            'spo2': 98,
            'current_medications': [],
            'allergies': []
        }
        st.json(safe_state)
    
    with col2:
        st.write("**Current State with Hazard:**")
        hazardous_state = {
            'symptoms': ['chest_pain', 'shortness_of_breath'],
            'heart_rate': 45,  # Critical value
            'spo2': 85,       # Critical value
            'current_medications': ['warfarin'],
            'allergies': ['penicillin']
        }
        st.json(hazardous_state)
    
    # Initialize hazard detector
    hazard_detector = HazardDetectionModule()
    hazard_detector.update_safe_state(safe_state)
    
    # Detect hazards
    is_hazardous, hazards = hazard_detector.detect_hazards(hazardous_state, "aspirin")
    
    if is_hazardous:
        st.error(f"Hazards Detected: {', '.join(hazards)}")
        
        if st.button("Execute Emergency Stop & Backtrack"):
            safe_state = hazard_detector.emergency_stop(hazardous_state)
            st.success("✅ Backtracked to safe state!")
            st.write("**Restored Safe State:**")
            st.json(safe_state)

def compare_enhanced_heuristics():
    """Compare original vs enhanced heuristics"""
    st.header("Enhanced vs Original Heuristics Comparison")
    
    # Sample data
    symptoms_db = {
        'fever': ['flu', 'covid', 'pneumonia'],
        'cough': ['flu', 'covid', 'pneumonia'],
        'chest_pain': ['heart_attack', 'pneumonia'],
        'shortness_of_breath': ['covid', 'pneumonia', 'heart_attack']
    }
    
    diseases_db = {
        'flu': {'symptoms': ['fever', 'cough'], 'severity': 'moderate', 'treatment_steps': ['rest', 'hydrate']},
        'pneumonia': {'symptoms': ['fever', 'cough', 'chest_pain'], 'severity': 'high', 'treatment_steps': ['emergency_care', 'antibiotics']},
        'heart_attack': {'symptoms': ['chest_pain', 'shortness_of_breath'], 'severity': 'emergency', 'treatment_steps': ['call_emergency']}
    }
    
    astar = AStarSymptomChecker(symptoms_db, diseases_db)
    test_symptoms = ['fever', 'cough', 'chest_pain']
    patient_condition = {'age': 70, 'comorbidities': ['hypertension'], 'vitals_stable': False}
    
    # Compare results
    original_results = astar.a_star_search(test_symptoms, astar.heuristic_severity_weighted)
    enhanced_results = astar.a_star_search(test_symptoms, astar.heuristic_risk_aware, patient_condition)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Heuristic")
        for disease, confidence, g, h, f in original_results[:3]:
            st.write(f"**{disease}**: {confidence:.1f}% (f-score: {f:.1f})")
    
    with col2:
        st.subheader("Risk-Aware Heuristic")
        for disease, confidence, g, h, f in enhanced_results[:3]:
            st.write(f"**{disease}**: {confidence:.1f}% (f-score: {f:.1f})")
    
    st.info("Note: Risk-aware heuristic penalizes hazardous conditions and considers patient-specific risks")

def main():
    st.title("🏥 AI Healthcare Assistant - Part 2 Enhanced")
    st.markdown("""
    **Enhanced with Planetary Rover-inspired AI Functionalities:**
    
    🚨 **Hazard Detection & Emergency Stop** (Like Rover's Cliff Detection)
    - Critical condition monitoring
    - Automatic backtracking to safe states
    - Emergency protocol activation
    
    🛡️ **Risk-Aware Path Planning** (Like Rover's Terrain Cost Mapping)
    - Treatment risk assessment
    - Hazardous path avoidance
    - Emergency rerouting
    
    🔄 **Enhanced Reflex Agent** (Like Rover's Battery Management)
    - Continuous safety monitoring
    - Condition-action rules with hazard checks
    - Safe state maintenance
    """)
    
    # Initialize AI assistant
    assistant = AIHealthcareAssistant()
    monitor = EnhancedHealthMonitor()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a feature",
        ["Symptom Checker (A*)", "Reflex Medication Agent", "Path Planning", 
         "Heuristics Comparison", "Health Analytics", "Hazard Detection Demo", "Enhanced Features"]
    )
    
    if app_mode == "Symptom Checker (A*)":
        st.header("🔍 Enhanced Symptom Checker with Risk-Aware A*")
        
        symptoms = list(assistant.symptoms_db.keys())
        selected_symptoms = st.multiselect("Select your symptoms:", symptoms)
        
        # Patient condition input
        st.subheader("Patient Condition (for risk assessment)")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=40)
            comorbidities = st.multiselect("Comorbidities", ["hypertension", "diabetes", "asthma", "heart_disease"])
        with col2:
            current_meds = st.multiselect("Current Medications", ["warfarin", "insulin", "beta_blockers", "none"])
            allergies = st.multiselect("Allergies", ["penicillin", "aspirin", "contrast_dye", "none"])
        
        patient_condition = {
            'age': age,
            'comorbidities': comorbidities,
            'current_medications': current_meds,
            'allergies': allergies,
            'vitals_stable': True
        }
        
        heuristic_choice = st.selectbox(
            "Choose A* Heuristic:",
            ["Manhattan", "Euclidean", "Symptom Frequency", "Severity Weighted", "Risk-Aware (NEW)"]
        )
        
        if selected_symptoms:
            if st.button("Run Enhanced A* Diagnosis"):
                # Map heuristic choice to function
                heuristic_map = {
                    "Manhattan": assistant.astar_checker.heuristic_manhattan,
                    "Euclidean": assistant.astar_checker.heuristic_euclidean,
                    "Symptom Frequency": assistant.astar_checker.heuristic_symptom_frequency,
                    "Severity Weighted": assistant.astar_checker.heuristic_severity_weighted
                }
                
                if heuristic_choice == "Risk-Aware (NEW)":
                    results = assistant.astar_checker.a_star_search(
                        selected_symptoms, 
                        heuristic_map[heuristic_choice],
                        patient_condition
                    )
                else:
                    results = assistant.astar_checker.a_star_search(
                        selected_symptoms, 
                        heuristic_map[heuristic_choice]
                    )
                
                st.subheader(f"A* Results using {heuristic_choice} Heuristic")
                for disease, confidence, g_score, h_score, f_score in results[:5]:
                    if confidence > 0:
                        disease_info = assistant.diseases_db.get(disease, {})
                        st.write(f"**{disease}** ({confidence:.1f}%)")
                        st.write(f"  g(n)={g_score}, h(n)={h_score:.2f}, f(n)={f_score:.2f}")
                        
                        # Hazard check for this disease
                        is_hazardous, hazards = assistant.hazard_detector.detect_hazards(patient_condition, disease)
                        if is_hazardous:
                            st.error(f"  🚨 Hazards: {', '.join(hazards)}")
                        
                        st.write(f"  Recommendation: {disease_info.get('recommendation', 'Consult doctor')}")
                        st.write("---")
    
    elif app_mode == "Reflex Medication Agent":
        st.header("💊 Enhanced Reflex Medication Agent")
        
        # Patient condition for hazard checking
        st.subheader("Patient Information for Safety Checks")
        allergies = st.multiselect("Patient Allergies", ["penicillin", "aspirin", "sulfa", "none"])
        current_conditions = st.multiselect("Current Conditions", ["pregnancy", "renal_disease", "liver_disease", "none"])
        
        patient_condition = {
            'allergies': allergies,
            'current_conditions': current_conditions,
            'current_medications': []
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Add Medication")
            med_name = st.text_input("Medication Name")
            dosage = st.text_input("Dosage")
            schedule_times = st.multiselect("Schedule", ["08:00", "12:00", "18:00", "20:00"])
            
            if st.button("Add Medication with Safety Check") and med_name and dosage:
                success = assistant.reflex_agent.add_medication(med_name, dosage, schedule_times, patient_condition)
                if success:
                    st.success(f"✅ Safely added {med_name}")
                    patient_condition['current_medications'].append(med_name)
        
        with col2:
            st.subheader("Current Reminders")
            reminders = assistant.reflex_agent.check_medication_time(patient_condition)
            if reminders:
                for reminder in reminders:
                    if "EMERGENCY" in reminder:
                        st.error(reminder)
                    else:
                        st.warning(reminder)
                        if st.button(f"Mark as Taken", key=reminder):
                            assistant.reflex_agent.mark_taken(reminder.split(" - ")[0].replace("Time to take ", ""))
                            st.rerun()
            else:
                st.info("No medications due")
    
    elif app_mode == "Path Planning":
        st.header("🛣️ Enhanced Treatment Path Planning")
        
        disease = st.selectbox("Select Condition:", list(assistant.diseases_db.keys()))
        current_step = st.selectbox("Current Step:", [""] + assistant.diseases_db.get(disease, {}).get('treatment_steps', []))
        
        # Patient condition for risk assessment
        st.subheader("Patient Risk Profile")
        age = st.slider("Age", 0, 100, 45)
        risk_factors = st.multiselect("Risk Factors", ["smoker", "obese", "sedentary", "family_history", "none"])
        
        patient_condition = {
            'age': age,
            'risk_factors': risk_factors,
            'vitals_stable': True
        }
        
        if disease:
            treatment_path = assistant.path_planner.plan_treatment_path(disease, patient_condition, current_step if current_step else None)
            
            st.subheader("Safe Treatment Path (Hazards Avoided)")
            total_risk = assistant.risk_assessor.calculate_treatment_risk([step for step, cost in treatment_path], patient_condition)
            
            st.write(f"**Total Risk Score**: {total_risk:.1f}")
            
            for i, (step, cost) in enumerate(treatment_path):
                risk_level = "🟢 Low" if cost <= 10 else "🟡 Medium" if cost <= 15 else "🔴 High"
                status = "✅ Current" if i == 0 and current_step else "➡️ Next" if i == 0 else "📋 Future"
                st.write(f"{status} Step {i+1}: {step.replace('_', ' ').title()} (cost: {cost}, risk: {risk_level})")
    
    elif app_mode == "Heuristics Comparison":
        compare_enhanced_heuristics()
    
    elif app_mode == "Hazard Detection Demo":
        demonstrate_hazard_detection()
    
    elif app_mode == "Enhanced Features":
        st.header("🎯 Part 2 Enhanced Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚨 Hazard Detection System")
            st.markdown("""
            **Inspired by Rover's Cliff Detection:**
            - Continuous patient state monitoring
            - Critical value thresholds
            - Automatic emergency stops
            - Backtracking to safe states
            - Multi-hazard detection:
              - Critical vitals
              - Drug interactions  
              - Allergy risks
              - Treatment contraindications
            """)
            
            # Demo hazard detection
            st.subheader("Quick Hazard Check")
            check_med = st.selectbox("Medication to check:", ["aspirin", "penicillin", "warfarin", "insulin"])
            check_allergy = st.selectbox("Patient has allergy:", ["none", "penicillin", "aspirin", "sulfa"])
            
            test_condition = {
                'allergies': [check_allergy] if check_allergy != "none" else [],
                'current_medications': ['warfarin'] if check_med == 'aspirin' else []
            }
            
            is_hazardous, hazards = assistant.hazard_detector.detect_hazards(test_condition, check_med)
            if is_hazardous:
                st.error(f"🚨 Hazard detected: {hazards}")
            else:
                st.success("✅ No hazards detected")
        
        with col2:
            st.subheader("🛡️ Risk-Aware Planning")
            st.markdown("""
            **Inspired by Rover's Terrain Cost Mapping:**
            - Treatment risk quantification
            - Patient-specific risk adjustment
            - Hazardous path avoidance
            - Emergency rerouting capability
            - Cost-based decision making
            """)
            
            st.subheader("Risk Assessment Demo")
            demo_treatment = st.selectbox("Treatment plan:", 
                                        ["routine_checkup", "antibiotics", "surgery_prep", "emergency_intervention"])
            demo_age = st.slider("Patient age:", 0, 100, 65)
            
            demo_condition = {'age': demo_age, 'comorbidities': ['hypertension']}
            risk_cost = assistant.risk_assessor.calculate_treatment_risk([demo_treatment], demo_condition)
            
            st.write(f"**Risk Cost**: {risk_cost:.1f}")
            if risk_cost < 10:
                st.success("Low risk treatment")
            elif risk_cost < 50:
                st.warning("Moderate risk treatment")
            else:
                st.error("High risk treatment - Consider alternatives")
    
    elif app_mode == "Health Analytics":
        st.header("📈 Enhanced Health Analytics")
        
        # Vital signs input
        st.subheader("Vital Signs Monitoring")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 75)
            bp_systolic = st.number_input("Systolic BP", 60, 200, 120)
        
        with col2:
            bp_diastolic = st.number_input("Diastolic BP", 40, 120, 80)
            temperature = st.number_input("Temperature (°C)", 35.0, 42.0, 36.8)
        
        with col3:
            spo2 = st.number_input("Oxygen Saturation (%)", 70, 100, 98)
        
        if st.button("Add Vital Signs & Analyze"):
            alerts = monitor.add_vital_signs(heart_rate, bp_systolic, bp_diastolic, temperature, spo2)
            trends = monitor.analyze_trends()
            
            if alerts:
                st.error("Critical Alerts:")
                for alert in alerts:
                    st.write(f"🚨 {alert}")
            
            st.success("Trend Analysis:")
            for trend in trends:
                st.write(f"📊 {trend}")

if __name__ == "__main__":
    main()
