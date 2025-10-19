# AI_Doctor_Personal_Health_Assistant.py
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

class AStarSymptomChecker:
    """A* Search implementation for symptom checking with multiple heuristics"""
    
    def __init__(self, symptoms_db, diseases_db):
        self.symptoms_db = symptoms_db
        self.diseases_db = diseases_db
        
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
    
    def a_star_search(self, selected_symptoms: List[str], heuristic_func: callable) -> List[Tuple[str, float]]:
        """A* search implementation for disease diagnosis"""
        current_symptoms_set = set(selected_symptoms)
        possible_diseases = set()
        
        # Get all possible diseases from selected symptoms
        for symptom in selected_symptoms:
            possible_diseases.update(self.symptoms_db.get(symptom, []))
        
        # Priority queue for A* search
        priority_queue = []
        
        for disease in possible_diseases:
            # g(n) = number of unmatched symptoms from selected ones
            target_symptoms = set(self.diseases_db.get(disease, {}).get('symptoms', []))
            g_score = len(current_symptoms_set - target_symptoms)
            
            # h(n) = heuristic estimate
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

class ReflexMedicationAgent:
    """Simple reflex agent for medication management"""
    
    def __init__(self):
        self.medication_schedule = {}
        self.adherence_history = {}
    
    def add_medication(self, name: str, dosage: str, times: List[str]):
        """Add medication to schedule"""
        self.medication_schedule[name] = {
            'dosage': dosage,
            'times': times,
            'last_taken': None,
            'adherence': []
        }
    
    def check_medication_time(self) -> List[str]:
        """Reflex agent - condition-action rules"""
        current_time = datetime.now()
        reminders = []
        
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

class PathPlanningModule:
    """Path planning for treatment recommendation sequence"""
    
    def __init__(self, diseases_db):
        self.diseases_db = diseases_db
        self.treatment_graph = self._build_treatment_graph()
    
    def _build_treatment_graph(self) -> Dict[str, List[Tuple[str, float]]]:
        """Build graph of treatment steps with costs"""
        graph = {}
        
        # Define treatment steps for different conditions
        treatment_steps = {
            'flu': ['rest', 'hydrate', 'medication', 'monitor'],
            'covid': ['isolate', 'test', 'consult_doctor', 'monitor_symptoms'],
            'pneumonia': ['emergency_care', 'antibiotics', 'hospitalization', 'followup'],
            'migraine': ['dark_room', 'hydration', 'pain_relief', 'rest']
        }
        
        # Add edges with costs (simulated)
        for disease, steps in treatment_steps.items():
            graph[disease] = []
            for i, step in enumerate(steps):
                cost = (i + 1) * 10  # Increasing cost for later steps
                graph[disease].append((step, cost))
        
        return graph
    
    def plan_treatment_path(self, disease: str, current_step: str = None) -> List[Tuple[str, float]]:
        """Plan optimal treatment path using A*"""
        if disease not in self.treatment_graph:
            return []
        
        # Simple linear path planning for treatment sequence
        treatment_steps = self.treatment_graph[disease]
        
        if current_step:
            # Find current step and return remaining path
            current_index = next((i for i, (step, _) in enumerate(treatment_steps) 
                               if step == current_step), 0)
            return treatment_steps[current_index:]
        
        return treatment_steps

class AIHealthcareAssistant:
    def __init__(self):
        self.symptoms_db = self._initialize_symptoms_database()
        self.diseases_db = self._initialize_diseases_database()
        self.patients_history = {}
        self.astar_checker = AStarSymptomChecker(self.symptoms_db, self.diseases_db)
        self.reflex_agent = ReflexMedicationAgent()
        self.path_planner = PathPlanningModule(self.diseases_db)
        
    def _initialize_symptoms_database(self):
        """Constraint Satisfaction Problem: Symptom-Disease relationships"""
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
            'sore_throat': ['flu', 'covid', 'strep_throat']
        }
    
    def _initialize_diseases_database(self):
        """Knowledge Base using First-Order Logic concepts"""
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
                'treatment_steps': ['emergency_care', 'antibiotics', 'hospitalization', 'followup']
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

class HealthMonitor:
    def __init__(self):
        self.vital_signs_history = []
    
    def add_vital_signs(self, heart_rate, bp_systolic, bp_diastolic, temperature, spo2):
        """Markov Decision Process for vital signs monitoring"""
        timestamp = datetime.now()
        self.vital_signs_history.append({
            'timestamp': timestamp,
            'heart_rate': heart_rate,
            'bp_systolic': bp_systolic,
            'bp_diastolic': bp_diastolic,
            'temperature': temperature,
            'spo2': spo2
        })
    
    def analyze_trends(self):
        """Time series analysis using HMM concepts"""
        if len(self.vital_signs_history) < 2:
            return ["Insufficient data for trend analysis"]
        
        df = pd.DataFrame(self.vital_signs_history)
        trends = []
        
        # Simple trend analysis
        hr_trend = "stable"
        if len(df) > 1:
            hr_change = df['heart_rate'].iloc[-1] - df['heart_rate'].iloc[-2]
            if abs(hr_change) > 10:
                hr_trend = "increasing" if hr_change > 0 else "decreasing"
        
        trends.append(f"Heart rate trend: {hr_trend}")
        return trends

def compare_heuristics_demo():
    """Demo function to compare different heuristics"""
    st.header("A* Heuristics Comparison")
    
    # Create sample data for comparison
    symptoms_db = {
        'fever': ['flu', 'covid', 'pneumonia'],
        'cough': ['flu', 'covid', 'pneumonia'],
        'headache': ['flu', 'migraine'],
        'fatigue': ['flu', 'covid']
    }
    
    diseases_db = {
        'flu': {'symptoms': ['fever', 'cough', 'headache', 'fatigue'], 'severity': 'moderate'},
        'covid': {'symptoms': ['fever', 'cough', 'fatigue'], 'severity': 'high'},
        'pneumonia': {'symptoms': ['fever', 'cough'], 'severity': 'high'},
        'migraine': {'symptoms': ['headache'], 'severity': 'moderate'}
    }
    
    astar = AStarSymptomChecker(symptoms_db, diseases_db)
    test_symptoms = ['fever', 'cough', 'headache']
    
    # Test all heuristics
    heuristics = [
        ('Manhattan', astar.heuristic_manhattan),
        ('Euclidean', astar.heuristic_euclidean),
        ('Symptom Frequency', astar.heuristic_symptom_frequency),
        ('Severity Weighted', astar.heuristic_severity_weighted)
    ]
    
    results_comparison = []
    
    for heuristic_name, heuristic_func in heuristics:
        results = astar.a_star_search(test_symptoms, heuristic_func)
        results_comparison.append({
            'Heuristic': heuristic_name,
            'Results': results[:3],  # Top 3 results
            'Top Disease': results[0][0] if results else 'None',
            'Confidence': f"{results[0][1]:.1f}%" if results else 'N/A'
        })
    
    # Display comparison
    st.subheader("Heuristic Comparison Results")
    for comp in results_comparison:
        st.write(f"**{comp['Heuristic']}**: Top result - {comp['Top Disease']} ({comp['Confidence']})")
        
        # Show detailed scores
        if comp['Results']:
            for disease, confidence, g, h, f in comp['Results'][:2]:
                st.write(f"  - {disease}: g={g}, h={h:.2f}, f={f:.2f}, confidence={confidence:.1f}%")

def main():
    st.title("🏥 AI Healthcare Assistant - Enhanced")
    st.markdown("""
    Enhanced with complete AI curriculum implementations:
    - **Reflex Agent** - Medication management with condition-action rules
    - **A* Path Planning** - Symptom checking with 4 different heuristics
    - **Treatment Path Planning** - Optimal treatment sequence planning
    """)
    
    # Initialize AI assistant
    assistant = AIHealthcareAssistant()
    monitor = HealthMonitor()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a feature",
        ["Symptom Checker (A*)", "Reflex Medication Agent", "Path Planning", 
         "Heuristics Comparison", "Health Analytics"]
    )
    
    if app_mode == "Symptom Checker (A*)":
        st.header("🔍 Symptom Checker with A* Search")
        
        symptoms = list(assistant.symptoms_db.keys())
        selected_symptoms = st.multiselect("Select your symptoms:", symptoms)
        
        heuristic_choice = st.selectbox(
            "Choose A* Heuristic:",
            ["Manhattan", "Euclidean", "Symptom Frequency", "Severity Weighted"]
        )
        
        if selected_symptoms:
            if st.button("Run A* Diagnosis"):
                # Map heuristic choice to function
                heuristic_map = {
                    "Manhattan": assistant.astar_checker.heuristic_manhattan,
                    "Euclidean": assistant.astar_checker.heuristic_euclidean,
                    "Symptom Frequency": assistant.astar_checker.heuristic_symptom_frequency,
                    "Severity Weighted": assistant.astar_checker.heuristic_severity_weighted
                }
                
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
                        st.write(f"  Recommendation: {disease_info.get('recommendation', 'Consult doctor')}")
                        st.write("---")
    
    elif app_mode == "Reflex Medication Agent":
        st.header("💊 Reflex Medication Agent")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Add Medication")
            med_name = st.text_input("Medication Name")
            dosage = st.text_input("Dosage")
            schedule_times = st.multiselect("Schedule", ["08:00", "12:00", "18:00", "20:00"])
            
            if st.button("Add Medication") and med_name and dosage:
                assistant.reflex_agent.add_medication(med_name, dosage, schedule_times)
                st.success(f"Added {med_name}")
        
        with col2:
            st.subheader("Current Reminders")
            reminders = assistant.reflex_agent.check_medication_time()
            if reminders:
                for reminder in reminders:
                    st.warning(reminder)
                    if st.button(f"Mark as Taken", key=reminder):
                        assistant.reflex_agent.mark_taken(reminder.split(" - ")[0].replace("Time to take ", ""))
                        st.rerun()
            else:
                st.info("No medications due")
    
    elif app_mode == "Path Planning":
        st.header("🛣️ Treatment Path Planning")
        
        disease = st.selectbox("Select Condition:", list(assistant.diseases_db.keys()))
        current_step = st.selectbox("Current Step:", [""] + assistant.diseases_db.get(disease, {}).get('treatment_steps', []))
        
        if disease:
            treatment_path = assistant.path_planner.plan_treatment_path(disease, current_step if current_step else None)
            
            st.subheader("Optimal Treatment Path")
            for i, (step, cost) in enumerate(treatment_path):
                status = "✅ Current" if i == 0 and current_step else "➡️ Next" if i == 0 else "📋 Future"
                st.write(f"{status} Step {i+1}: {step.replace('_', ' ').title()} (cost: {cost})")
    
    elif app_mode == "Heuristics Comparison":
        compare_heuristics_demo()
    
    elif app_mode == "Health Analytics":
        st.header("📈 Health Analytics")
        # ... (rest of your existing health analytics code)

if __name__ == "__main__":
    main()
