# AI_Doctor_Personal_Health_Assistant_COMPLETE.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import heapq
from typing import List, Tuple, Dict, Set
import math
import itertools
from collections import defaultdict, deque

# Configure Streamlit page
st.set_page_config(
    page_title="AI Doctor Helper - Complete AI Curriculum",
    page_icon="🏥",
    layout="wide"
)

# =============================================================================
# CORE AI MODULES (Topics 1-5)
# =============================================================================

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

class AStarSymptomChecker:
    """A* Search implementation for symptom checking - Topic 5: Informed Search"""
    
    def __init__(self, symptoms_db, diseases_db):
        self.symptoms_db = symptoms_db
        self.diseases_db = diseases_db
        self.hazard_detector = HazardDetectionModule()
        self.risk_assessor = RiskAssessmentModule()
        
    def heuristic_manhattan(self, current_symptoms: Set[str], target_disease: str) -> float:
        """
        Topic 5: Manhattan Distance Heuristic
        Equation: h(n) = |unmatched symptoms|
        Measures absolute difference in symptom sets
        """
        if target_disease not in self.diseases_db:
            return float('inf')
        
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        unmatched_symptoms = target_symptoms - current_symptoms
        
        st.sidebar.info(f"Manhattan: Target symptoms: {len(target_symptoms)}, Unmatched: {len(unmatched_symptoms)}")
        return len(unmatched_symptoms)

    def heuristic_euclidean(self, current_symptoms: Set[str], target_disease: str) -> float:
        """
        Topic 5: Euclidean Distance Heuristic  
        Equation: h(n) = √(∑(symptom_difference)²)
        Measures straight-line distance in symptom space
        """
        if target_disease not in self.diseases_db:
            return float('inf')
        
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        matched = len(current_symptoms.intersection(target_symptoms))
        total_target = len(target_symptoms)
        
        distance = math.sqrt((total_target - matched) ** 2)
        st.sidebar.info(f"Euclidean: Matched {matched}/{total_target}, Distance: {distance:.2f}")
        return distance

    def heuristic_symptom_frequency(self, current_symptoms: Set[str], target_disease: str) -> float:
        """
        Topic 5: Symptom Frequency Heuristic
        Equation: h(n) = -∑(1/(frequency(symptom) + 1))
        Prefers rare symptoms that are more specific to diseases
        """
        if target_disease not in self.diseases_db:
            return float('inf')
        
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        common_symptoms = current_symptoms.intersection(target_symptoms)
        
        frequency_score = 0
        for symptom in common_symptoms:
            disease_count = len(self.symptoms_db.get(symptom, []))
            frequency_score += 1 / (disease_count + 1)
        
        score = -frequency_score if frequency_score > 0 else float('inf')
        st.sidebar.info(f"Symptom Frequency: Common symptoms: {len(common_symptoms)}, Score: {score:.2f}")
        return score

    def heuristic_severity_weighted(self, current_symptoms: Set[str], target_disease: str) -> float:
        """
        Topic 5: Severity Weighted Heuristic
        Equation: h(n) = |unmatched symptoms| / severity_weight
        Prioritizes more severe conditions
        """
        if target_disease not in self.diseases_db:
            return float('inf')
        
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        unmatched_symptoms = target_symptoms - current_symptoms
        
        severity_weights = {
            'emergency': 3.0, 'high': 2.0, 'moderate': 1.5, 'low': 1.0
        }
        
        severity = self.diseases_db[target_disease].get('severity', 'moderate')
        weight = severity_weights.get(severity, 1.0)
        
        score = len(unmatched_symptoms) / weight
        st.sidebar.info(f"Severity Weighted: Unmatched: {len(unmatched_symptoms)}, Severity: {severity}, Score: {score:.2f}")
        return score

    def heuristic_risk_aware(self, current_symptoms: Set[str], target_disease: str, patient_condition: Dict = None) -> float:
        """
        COMPULSORY: Risk-aware heuristic considering patient safety
        Equation: h(n) = base_score + risk_cost
        Combines symptom matching with patient-specific risk assessment
        """
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
        
        final_score = base_score + risk_cost
        st.sidebar.info(f"Risk-Aware: Base: {base_score:.2f}, Risk: {risk_cost:.2f}, Final: {final_score:.2f}")
        return final_score

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
    """Simple reflex agent for medication management - Topic 2: Intelligent Agents"""
    
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
    """Path planning for treatment recommendation sequence - Topic 3: Problem Solving"""
    
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

# =============================================================================
# ADVANCED AI MODULES (Topics 6-22)
# =============================================================================

class MedicalCSP:
    """Constraint Satisfaction Problem for treatment planning - Topic 6: CSP"""
    
    def __init__(self, variables: List[str], domains: Dict[str, List[str]]):
        self.variables = variables
        self.domains = domains
        self.constraints = []
    
    def add_constraint(self, constraint_func):
        """Add a constraint function"""
        self.constraints.append(constraint_func)
    
    def is_consistent(self, assignment: Dict[str, str]) -> bool:
        """Check if current assignment satisfies all constraints"""
        for constraint in self.constraints:
            if not constraint(assignment):
                return False
        return True
    
    def backtracking_search(self, assignment: Dict[str, str] = None) -> Dict[str, str]:
        """Backtracking search for CSP solution"""
        if assignment is None:
            assignment = {}
        
        if len(assignment) == len(self.variables):
            return assignment
        
        unassigned = [v for v in self.variables if v not in assignment]
        first = unassigned[0]
        
        for value in self.domains[first]:
            new_assignment = assignment.copy()
            new_assignment[first] = value
            
            if self.is_consistent(new_assignment):
                result = self.backtracking_search(new_assignment)
                if result is not None:
                    return result
        
        return None

class BayesianNetwork:
    """Bayesian Network for probabilistic medical diagnosis - Topic 11: Probability"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.initialize_medical_network()
    
    def initialize_medical_network(self):
        """Initialize medical Bayesian network with diseases, symptoms, and risk factors"""
        # Prior probabilities of diseases
        self.nodes = {
            'Flu': 0.05,
            'Covid': 0.02,
            'Pneumonia': 0.01,
            'Heart_Attack': 0.005,
            'Migraine': 0.08,
            'Food_Poisoning': 0.03
        }
        
        # Conditional probability tables
        self.edges = {
            'Fever': {
                'Flu': 0.9, 'Covid': 0.95, 'Pneumonia': 0.85, 'Heart_Attack': 0.1, 'Migraine': 0.2, 'Food_Poisoning': 0.4
            },
            'Cough': {
                'Flu': 0.8, 'Covid': 0.9, 'Pneumonia': 0.95, 'Heart_Attack': 0.3, 'Migraine': 0.1, 'Food_Poisoning': 0.1
            },
            'Chest_Pain': {
                'Flu': 0.1, 'Covid': 0.2, 'Pneumonia': 0.7, 'Heart_Attack': 0.95, 'Migraine': 0.05, 'Food_Poisoning': 0.1
            }
        }
    
    def infer(self, evidence: Dict[str, bool]) -> Dict[str, float]:
        """Naive Bayes inference given evidence (symptoms)"""
        posterior = {}
        
        for disease, prior in self.nodes.items():
            # P(Disease | Evidence) ∝ P(Disease) * ∏ P(Evidence_i | Disease)
            likelihood = 1.0
            
            for symptom, present in evidence.items():
                if symptom in self.edges:
                    prob = self.edges[symptom].get(disease, 0.01)
                    likelihood *= prob if present else (1 - prob)
            
            posterior[disease] = prior * likelihood
        
        # Normalize
        total = sum(posterior.values())
        if total > 0:
            posterior = {d: p/total for d, p in posterior.items()}
        
        return posterior

class MarkovDecisionProcess:
    """MDP for treatment planning over time - Topic 18: Sequential Decision Making"""
    
    def __init__(self):
        # States: health levels 0-4 (0=critical, 4=excellent)
        self.states = [0, 1, 2, 3, 4]
        
        # Actions: treatments
        self.actions = ['monitor', 'medicate', 'hospitalize', 'surgery']
        
        # Rewards: based on health state and treatment cost
        self.rewards = {
            state: state * 10 for state in self.states  # Higher health = higher reward
        }
        
        # Treatment costs (negative rewards)
        self.action_costs = {
            'monitor': -1, 'medicate': -5, 'hospitalize': -20, 'surgery': -50
        }
    
    def value_iteration(self, gamma: float = 0.9, theta: float = 1e-6) -> Tuple[Dict, Dict]:
        """Value iteration algorithm for MDP"""
        V = {state: 0 for state in self.states}
        policy = {state: 'monitor' for state in self.states}
        
        while True:
            delta = 0
            for state in self.states:
                v = V[state]
                max_value = float('-inf')
                
                for action in self.actions:
                    action_value = 0
                    # Simplified transition - in reality, would use transition probabilities
                    if action == 'medicate' and state < 4:
                        action_value = self.rewards[state + 1] + self.action_costs[action] + gamma * V[state + 1]
                    elif action == 'hospitalize' and state < 3:
                        action_value = self.rewards[state + 2] + self.action_costs[action] + gamma * V[state + 2]
                    else:
                        action_value = self.rewards[state] + self.action_costs[action] + gamma * V[state]
                    
                    if action_value > max_value:
                        max_value = action_value
                        policy[state] = action
                
                V[state] = max_value
                delta = max(delta, abs(v - V[state]))
            
            if delta < theta:
                break
        
        return V, policy

class ReinforcementLearningAgent:
    """Q-learning agent for treatment optimization - Topic 21: Reinforcement Learning"""
    
    def __init__(self, states: List[int], actions: List[str], alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.states = states
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def choose_action(self, state: int) -> str:
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            if not q_values:
                return random.choice(self.actions)
            return max(q_values.keys(), key=lambda a: q_values[a])
    
    def update_q_value(self, state: int, action: str, reward: float, next_state: int):
        """Q-learning update rule"""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

# =============================================================================
# MAIN HEALTHCARE ASSISTANT
# =============================================================================

class AIHealthcareAssistant:
    def __init__(self):
        self.symptoms_db = self._initialize_symptoms_database()
        self.diseases_db = self._initialize_diseases_database()
        self.astar_checker = AStarSymptomChecker(self.symptoms_db, self.diseases_db)
        self.reflex_agent = ReflexMedicationAgent()
        self.path_planner = PathPlanningModule(self.diseases_db)
        self.bayesian_network = BayesianNetwork()
        self.mdp = MarkovDecisionProcess()
        
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

# =============================================================================
# DEMONSTRATION FUNCTIONS FOR PRESENTATION
# =============================================================================

def demonstrate_a_star_heuristics():
    """Topic 5: Demonstrate different A* heuristics with equations"""
    st.header("🎯 A* Search Heuristics Comparison")
    
    assistant = AIHealthcareAssistant()
    test_symptoms = ['fever', 'cough', 'headache']
    
    st.subheader("Test Symptoms: fever, cough, headache")
    
    # Create columns for each heuristic
    heuristics = ["Manhattan", "Euclidean", "Symptom Frequency", "Severity Weighted", "Risk-Aware"]
    
    for heuristic in heuristics:
        with st.expander(f"{heuristic} Distance Heuristic", expanded=True):
            st.write(f"**Algorithm**: {heuristic}")
            
            if heuristic == "Risk-Aware":
                patient_condition = {'age': 65, 'comorbidities': ['diabetes'], 'current_medications': ['warfarin']}
                results = assistant.astar_checker.a_star_search(test_symptoms, heuristic, patient_condition)
            else:
                results = assistant.astar_checker.a_star_search(test_symptoms, heuristic)
            
            if results:
                for disease, confidence, g_score, h_score, f_score in results[:3]:
                    st.write(f"**{disease}**: {confidence:.1f}% confidence")
                    st.write(f"g(n)={g_score}, h(n)={h_score:.2f}, f(n)={f_score:.2f}")

def demonstrate_bayesian_network():
    """Topic 11: Demonstrate Bayesian probability in diagnosis"""
    st.header("🕸️ Bayesian Network Diagnosis")
    
    bn = BayesianNetwork()
    
    st.subheader("Medical Bayesian Network Structure")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Prior Probabilities (Diseases):**")
        for disease, prob in bn.nodes.items():
            st.write(f"P({disease}) = {prob:.3f}")
    
    with col2:
        st.write("**Conditional Probabilities (P(Symptom|Disease)):**")
        st.write("Fever probabilities:")
        for disease, prob in bn.edges['Fever'].items():
            st.write(f"P(Fever|{disease}) = {prob:.2f}")
    
    # Interactive diagnosis
    st.subheader("Interactive Diagnosis with Evidence")
    evidence = {}
    symptoms = list(bn.edges.keys())
    
    for symptom in symptoms:
        evidence[symptom] = st.checkbox(symptom)
    
    if st.button("Calculate Bayesian Probabilities"):
        posterior = bn.infer(evidence)
        
        st.write("**Posterior Probabilities (P(Disease|Evidence)):**")
        for disease, prob in sorted(posterior.items(), key=lambda x: x[1], reverse=True):
            st.write(f"P({disease}|Evidence) = {prob:.4f}")

def demonstrate_markov_decision_process():
    """Topic 18: Demonstrate MDP for treatment planning"""
    st.header("🎯 Markov Decision Process - Treatment Optimization")
    
    mdp = MarkovDecisionProcess()
    V, policy = mdp.value_iteration()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Optimal Value Function V*(s)")
        st.write("State (Health Level) → Expected Total Reward")
        for state, value in V.items():
            health_status = ["Critical", "Poor", "Fair", "Good", "Excellent"]
            st.write(f"Health {health_status[state]}: V* = {value:.2f}")
    
    with col2:
        st.subheader("Optimal Policy π*(s)")
        st.write("State → Best Action")
        for state, action in policy.items():
            health_status = ["Critical", "Poor", "Fair", "Good", "Excellent"]
            st.write(f"Health {health_status[state]}: {action}")

def demonstrate_constraint_satisfaction():
    """Topic 6: Demonstrate CSP for treatment planning"""
    st.header("🔗 Constraint Satisfaction Problem - Treatment Planning")
    
    # Define CSP for treatment planning
    variables = ['medication', 'therapy', 'diet', 'exercise']
    domains = {
        'medication': ['aspirin', 'antibiotic', 'antiviral', 'none'],
        'therapy': ['rest', 'physio', 'respiratory', 'none'],
        'diet': ['normal', 'liquid', 'soft', 'restricted'],
        'exercise': ['none', 'light', 'moderate', 'intensive']
    }
    
    csp = MedicalCSP(variables, domains)
    
    # Add medical constraints
    def constraint1(assignment):
        # If antibiotic, then soft diet
        if assignment.get('medication') == 'antibiotic' and assignment.get('diet') not in ['soft', 'liquid']:
            return False
        return True
    
    def constraint2(assignment):
        # If respiratory therapy, no intensive exercise
        if assignment.get('therapy') == 'respiratory' and assignment.get('exercise') == 'intensive':
            return False
        return True
    
    csp.add_constraint(constraint1)
    csp.add_constraint(constraint2)
    
    st.write("**Variables and Domains:**")
    st.write(domains)
    
    st.write("**Constraints:**")
    st.write("1. If antibiotic medication, then soft or liquid diet")
    st.write("2. If respiratory therapy, no intensive exercise")
    
    if st.button("Find Valid Treatment Plan"):
        solution = csp.backtracking_search()
        if solution:
            st.success("**Valid Treatment Plan Found:**")
            for var, value in solution.items():
                st.write(f"{var}: {value}")
        else:
            st.error("No valid treatment plan found!")

def demonstrate_reinforcement_learning():
    """Topic 21: Demonstrate RL for treatment learning"""
    st.header("🤖 Reinforcement Learning - Treatment Strategy Learning")
    
    # Initialize RL agent
    states = [0, 1, 2, 3, 4]  # Health states
    actions = ['monitor', 'medicate', 'hospitalize']
    rl_agent = ReinforcementLearningAgent(states, actions)
    
    st.subheader("Q-Learning Training Simulation")
    
    if st.button("Train RL Agent (10 episodes)"):
        progress_bar = st.progress(0)
        rewards_history = []
        
        for episode in range(10):
            state = 2  # Start from fair health
            total_reward = 0
            
            for step in range(5):  # 5 steps per episode
                action = rl_agent.choose_action(state)
                
                # Simulate environment
                if action == 'medicate' and state < 4:
                    next_state = state + 1
                    reward = 10
                elif action == 'hospitalize' and state < 3:
                    next_state = state + 2
                    reward = 15
                else:
                    next_state = max(state - 1, 0)
                    reward = -5
                
                rl_agent.update_q_value(state, action, reward, next_state)
                total_reward += reward
                state = next_state
            
            rewards_history.append(total_reward)
            progress_bar.progress((episode + 1) / 10)
        
        # Display results
        st.subheader("Learning Results")
        st.line_chart(rewards_history)
        st.write("**Learned Q-Table (partial):**")
        st.write(dict(rl_agent.q_table))

def demonstrate_reflex_agent():
    """Topic 2: Demonstrate reflex agent for medication management"""
    st.header("🤖 Reflex Agent - Medication Management")
    
    agent = ReflexMedicationAgent()
    
    st.subheader("Condition-Action Rules")
    st.write("""
    **Rules:**
    - IF current_time = medication_time AND medication_not_taken THEN remind_patient
    - IF hazard_detected THEN emergency_stop
    - IF safe_condition THEN update_safe_state
    """)
    
    # Simulate medication schedule
    patient_condition = {'allergies': [], 'current_medications': []}
    agent.add_medication("Aspirin", "100mg", ["08:00", "20:00"], patient_condition)
    agent.add_medication("Antibiotic", "500mg", ["12:00"], patient_condition)
    
    st.write("**Current Medication Schedule:**")
    st.write(agent.medication_schedule)
    
    if st.button("Check for Medication Reminders"):
        reminders = agent.check_medication_time(patient_condition)
        if reminders:
            for reminder in reminders:
                st.warning(reminder)
        else:
            st.info("No medications due at this time")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.title("🏥 AI Doctor Helper - Complete AI Curriculum")
    st.markdown("""
    ## AI Topics Demonstration for Presentation
    
    **Select a topic below to see live demonstrations of AI algorithms in healthcare:**
    """)
    
    # Topic selection
    topic = st.sidebar.selectbox(
        "🎓 Select AI Topic to Demonstrate",
        [
            "🏠 Overview",
            "🎯 A* Search Heuristics", 
            "🤖 Reflex Agents",
            "🔗 Constraint Satisfaction",
            "🕸️ Bayesian Networks", 
            "🎯 Markov Decision Processes",
            "🤖 Reinforcement Learning",
            "💊 Complete Symptom Checker"
        ]
    )
    
    if topic == "🏠 Overview":
        st.header("📚 AI Curriculum Coverage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Core AI (Topics 1-5)")
            st.write("""
            - **Topic 1**: Introduction to AI
            - **Topic 2**: Intelligent Agents ✓
            - **Topic 3**: Problem Solving Agents ✓
            - **Topic 4**: Uninformed Search
            - **Topic 5**: Informed Search (A*) ✓
            """)
            
            st.subheader("✅ Advanced AI (Topics 6-10)")
            st.write("""
            - **Topic 6**: Constraint Satisfaction ✓
            - **Topic 7**: CSP Algorithms ✓
            - **Topic 8**: Symbolic AI
            - **Topic 9**: Propositional Logic
            - **Topic 10**: First-Order Logic
            """)
        
        with col2:
            st.subheader("✅ Probabilistic AI (Topics 11-16)")
            st.write("""
            - **Topic 11**: Probability Review ✓
            - **Topic 12**: Bayes Theorem ✓
            - **Topic 13**: Bayesian Networks ✓
            - **Topic 14**: Probabilistic Reasoning
            - **Topic 15**: HMMs
            - **Topic 16**: Utility Theory
            """)
            
            st.subheader("✅ Sequential Decisions (Topics 17-22)")
            st.write("""
            - **Topic 17**: MDP Introduction
            - **Topic 18**: MDP Algorithms ✓
            - **Topic 19**: Value Iteration ✓
            - **Topic 20**: POMDPs
            - **Topic 21**: Reinforcement Learning ✓
            - **Topic 22**: RL Applications ✓
            """)
    
    elif topic == "🎯 A* Search Heuristics":
        demonstrate_a_star_heuristics()
        
    elif topic == "🤖 Reflex Agents":
        demonstrate_reflex_agent()
        
    elif topic == "🔗 Constraint Satisfaction":
        demonstrate_constraint_satisfaction()
        
    elif topic == "🕸️ Bayesian Networks":
        demonstrate_bayesian_network()
        
    elif topic == "🎯 Markov Decision Processes":
        demonstrate_markov_decision_process()
        
    elif topic == "🤖 Reinforcement Learning":
        demonstrate_reinforcement_learning()
        
    elif topic == "💊 Complete Symptom Checker":
        st.header("🔍 Complete Symptom Checker with All AI Techniques")
        
        assistant = AIHealthcareAssistant()
        
        # Symptoms selection
        symptoms = list(assistant.symptoms_db.keys())
        selected_symptoms = st.multiselect("Select your symptoms:", symptoms)
        
        if selected_symptoms:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Patient Information")
                age = st.number_input("Age", min_value=0, max_value=120, value=40)
                comorbidities = st.multiselect("Comorbidities", ["diabetes", "hypertension", "asthma", "heart_disease"])
                current_meds = st.multiselect("Current Medications", ["warfarin", "insulin", "aspirin", "none"])
                allergies = st.multiselect("Allergies", ["penicillin", "aspirin", "sulfa", "none"])
            
            patient_condition = {
                'age': age,
                'comorbidities': comorbidities,
                'current_medications': [m for m in current_meds if m != "none"],
                'allergies': [a for a in allergies if a != "none"]
            }
            
            heuristic_choice = st.selectbox(
                "Select A* Heuristic:",
                ["Manhattan", "Euclidean", "Symptom Frequency", "Severity Weighted", "Risk-Aware"]
            )
            
            if st.button("Run Comprehensive Diagnosis"):
                with col2:
                    st.subheader(f"Diagnosis Results ({heuristic_choice} Heuristic)")
                    
                    # Run A* search
                    results = assistant.astar_checker.a_star_search(
                        selected_symptoms, 
                        heuristic_choice, 
                        patient_condition if heuristic_choice == "Risk-Aware" else None
                    )
                    
                    if results:
                        for disease, confidence, g_score, h_score, f_score in results[:3]:
                            if confidence > 10:
                                disease_info = assistant.diseases_db.get(disease, {})
                                
                                st.write(f"### {disease} ({confidence:.1f}%)")
                                st.write(f"**Scores**: g(n)={g_score}, h(n)={h_score:.2f}, f(n)={f_score:.2f}")
                                st.write(f"**Recommendation**: {disease_info.get('recommendation', 'Consult healthcare provider')}")
                                
                                # Show treatment path
                                treatment_path = assistant.path_planner.plan_treatment_path(disease, patient_condition)
                                st.write("**Treatment Plan:**")
                                for step, cost in treatment_path:
                                    st.write(f"- {step.replace('_', ' ').title()} (cost: {cost})")
                                
                                st.write("---")
                    else:
                        st.warning("No matching conditions found.")

if __name__ == "__main__":
    main()
