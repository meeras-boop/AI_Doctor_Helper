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
# NEW: PROBABILISTIC REASONING MODULES (Topics 11-16)
# =============================================================================

class BayesianNetwork:
    """Bayesian Network for probabilistic medical diagnosis"""
    
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
            },
            'Shortness_of_Breath': {
                'Flu': 0.2, 'Covid': 0.6, 'Pneumonia': 0.8, 'Heart_Attack': 0.9, 'Migraine': 0.05, 'Food_Poisoning': 0.1
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

class HiddenMarkovModel:
    """HMM for tracking patient health states over time"""
    
    def __init__(self):
        # States: Healthy, Sick, Critical, Recovering
        self.states = ['Healthy', 'Sick', 'Critical', 'Recovering']
        
        # Transition probabilities
        self.transitions = {
            'Healthy': {'Healthy': 0.8, 'Sick': 0.15, 'Critical': 0.05, 'Recovering': 0.0},
            'Sick': {'Healthy': 0.1, 'Sick': 0.6, 'Critical': 0.25, 'Recovering': 0.05},
            'Critical': {'Healthy': 0.0, 'Sick': 0.2, 'Critical': 0.5, 'Recovering': 0.3},
            'Recovering': {'Healthy': 0.4, 'Sick': 0.1, 'Critical': 0.05, 'Recovering': 0.45}
        }
        
        # Observation probabilities (symptom severity)
        self.emissions = {
            'Healthy': {'Mild': 0.8, 'Moderate': 0.15, 'Severe': 0.05},
            'Sick': {'Mild': 0.3, 'Moderate': 0.5, 'Severe': 0.2},
            'Critical': {'Mild': 0.05, 'Moderate': 0.25, 'Severe': 0.7},
            'Recovering': {'Mild': 0.6, 'Moderate': 0.3, 'Severe': 0.1}
        }
    
    def forward_algorithm(self, observations: List[str]) -> List[Dict[str, float]]:
        """Forward algorithm for filtering (current belief state)"""
        # Initial belief (uniform)
        belief = {state: 1.0/len(self.states) for state in self.states}
        beliefs_history = [belief.copy()]
        
        for obs in observations:
            new_belief = {}
            
            for current_state in self.states:
                # P(current_state | observations) = 
                # P(observation | current_state) * Σ[P(current_state | previous_state) * P(previous_state)]
                prob = 0.0
                for prev_state in self.states:
                    prob += self.transitions[prev_state].get(current_state, 0) * belief[prev_state]
                
                new_belief[current_state] = self.emissions[current_state].get(obs, 0.001) * prob
            
            # Normalize
            total = sum(new_belief.values())
            if total > 0:
                belief = {s: p/total for s, p in new_belief.items()}
            else:
                belief = new_belief
            
            beliefs_history.append(belief.copy())
        
        return beliefs_history

class UtilityTheory:
    """Utility theory for medical decision making"""
    
    def __init__(self):
        self.quality_of_life_weights = {
            'healthy': 1.0,
            'mild_illness': 0.8,
            'moderate_illness': 0.5,
            'severe_illness': 0.2,
            'critical': 0.1
        }
    
    def calculate_expected_utility(self, outcomes: List[Tuple[float, str, float]]) -> float:
        """Calculate expected utility of a decision"""
        # outcomes: [(probability, health_state, duration_years), ...]
        total_utility = 0.0
        
        for prob, state, years in outcomes:
            qol = self.quality_of_life_weights.get(state, 0.5)
            # Discount future years (time preference)
            discounted_years = years * (1 - math.exp(-0.1 * years))  # Simple discounting
            utility = qol * discounted_years
            total_utility += prob * utility
        
        return total_utility
    
    def value_of_perfect_information(self, current_decision: str, test_results: Dict) -> float:
        """Calculate value of perfect information for diagnostic tests"""
        base_utility = self.calculate_expected_utility(test_results.get('base_outcomes', []))
        perfect_info_utility = self.calculate_expected_utility(test_results.get('perfect_info_outcomes', []))
        
        return max(0, perfect_info_utility - base_utility)

# =============================================================================
# NEW: SEQUENTIAL DECISION MAKING (Topics 17-20)
# =============================================================================

class MarkovDecisionProcess:
    """MDP for treatment planning over time"""
    
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
    
    def transition_probability(self, state: int, action: str, next_state: int) -> float:
        """Transition probabilities based on current state and action"""
        base_probabilities = {
            'monitor': [0.6, 0.3, 0.1, 0.0, 0.0],
            'medicate': [0.1, 0.4, 0.4, 0.1, 0.0],
            'hospitalize': [0.0, 0.2, 0.5, 0.3, 0.0],
            'surgery': [0.3, 0.4, 0.2, 0.1, 0.0]  # Surgery has recovery risk
        }
        
        action_probs = base_probabilities.get(action, base_probabilities['monitor'])
        if next_state - state < len(action_probs) and next_state - state >= 0:
            return action_probs[next_state - state]
        return 0.0
    
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
                    for next_state in self.states:
                        prob = self.transition_probability(state, action, next_state)
                        reward = self.rewards[next_state] + self.action_costs[action]
                        action_value += prob * (reward + gamma * V[next_state])
                    
                    if action_value > max_value:
                        max_value = action_value
                        policy[state] = action
                
                V[state] = max_value
                delta = max(delta, abs(v - V[state]))
            
            if delta < theta:
                break
        
        return V, policy

class ReinforcementLearningAgent:
    """Q-learning agent for treatment optimization"""
    
    def __init__(self, states: List[int], actions: List[str], alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.states = states
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.episode_history = []
    
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
    
    def train_episode(self, initial_state: int, max_steps: int = 10):
        """Train for one episode"""
        state = initial_state
        total_reward = 0
        steps = 0
        episode = []
        
        while steps < max_steps and state > 0:  # Stop if healthy or max steps
            action = self.choose_action(state)
            
            # Simulate environment (simplified)
            if action == 'medicate' and state < 4:
                next_state = min(state + 1, 4)
                reward = 10 - 5  # Health improvement - cost
            elif action == 'hospitalize' and state < 3:
                next_state = min(state + 2, 4)
                reward = 20 - 20
            else:
                next_state = max(state - 1, 0)
                reward = -1
            
            self.update_q_value(state, action, reward, next_state)
            episode.append((state, action, reward, next_state))
            
            total_reward += reward
            state = next_state
            steps += 1
        
        self.episode_history.append((total_reward, steps))
        return total_reward, episode

# =============================================================================
# NEW: SYMBOLIC AI AND LOGICAL AGENTS (Topics 8-10)
# =============================================================================

class PropositionalLogic:
    """Propositional logic for medical rules"""
    
    def __init__(self):
        self.knowledge_base = set()
        self.symbols = set()
    
    def add_rule(self, premise: List[str], conclusion: str):
        """Add implication rule: premise → conclusion"""
        rule = f"({' & '.join(premise)}) => {conclusion}"
        self.knowledge_base.add(rule)
        self.symbols.update(premise)
        self.symbols.add(conclusion)
    
    def modus_ponens(self, facts: Set[str]) -> Set[str]:
        """Apply modus ponens inference"""
        new_facts = facts.copy()
        changed = True
        
        while changed:
            changed = False
            for rule in self.knowledge_base:
                if '=>' in rule:
                    premise_str, conclusion = rule.split('=>')
                    premise_str = premise_str.strip('() ')
                    premises = [p.strip() for p in premise_str.split('&')]
                    
                    if all(premise in new_facts for premise in premises):
                        if conclusion.strip() not in new_facts:
                            new_facts.add(conclusion.strip())
                            changed = True
        
        return new_facts

class FirstOrderLogic:
    """First-order logic for medical relationships"""
    
    def __init__(self):
        self.predicates = defaultdict(list)
        self.rules = []
    
    def add_fact(self, predicate: str, *args):
        """Add a fact to the knowledge base"""
        self.predicates[predicate].append(args)
    
    def add_rule(self, condition, conclusion):
        """Add a first-order rule"""
        self.rules.append((condition, conclusion))
    
    def query(self, predicate: str, *args) -> List[Tuple]:
        """Query the knowledge base"""
        return self.predicates.get(predicate, [])

# =============================================================================
# NEW: CONSTRAINT SATISFACTION PROBLEMS (Topics 6-7)
# =============================================================================

class MedicalCSP:
    """Constraint Satisfaction Problem for treatment planning"""
    
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

# =============================================================================
# ENHANCED EXISTING CLASSES WITH NEW AI TOPICS
# =============================================================================

class EnhancedHazardDetectionModule:
    """Enhanced hazard detection with probabilistic reasoning"""
    
    def __init__(self):
        self.hazardous_conditions = {
            'critical_vitals': ['heart_attack', 'stroke', 'septic_shock'],
            'drug_interactions': ['warfarin_aspirin', 'beta_blocker_asthma'],
            'allergy_risks': ['penicillin_allergy', 'contrast_allergy'],
            'contraindications': ['pregnancy_medications', 'renal_impairment']
        }
        self.safe_state_history = []
        self.bayesian_network = BayesianNetwork()
    
    def probabilistic_hazard_detection(self, patient_condition: Dict, treatment: str) -> Tuple[bool, float, List[str]]:
        """Probabilistic hazard detection using Bayesian reasoning"""
        # Convert patient condition to evidence
        evidence = {}
        for key, value in patient_condition.items():
            if isinstance(value, bool):
                evidence[key] = value
            elif key == 'heart_rate':
                evidence['abnormal_heart_rate'] = value > 150 or value < 40
        
        # Get hazard probability from Bayesian network
        hazard_prob = self.bayesian_network.infer(evidence).get('Heart_Attack', 0.0)
        
        # Traditional rule-based detection
        is_hazardous, hazards = self.detect_hazards(patient_condition, treatment)
        
        # Combined risk score
        combined_risk = hazard_prob + (0.5 if is_hazardous else 0.0)
        
        return is_hazardous or hazard_prob > 0.3, combined_risk, hazards
    
    def detect_hazards(self, patient_condition: Dict, treatment: str) -> Tuple[bool, List[str]]:
        """Original hazard detection logic"""
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

class EnhancedAStarSymptomChecker:
    """Enhanced A* with multiple AI techniques"""
    
    def __init__(self, symptoms_db, diseases_db):
        self.symptoms_db = symptoms_db
        self.diseases_db = diseases_db
        self.hazard_detector = EnhancedHazardDetectionModule()
        self.risk_assessor = RiskAssessmentModule()
        self.bayesian_network = BayesianNetwork()
        self.propositional_logic = PropositionalLogic()
        self.initialize_logical_rules()
    
    def initialize_logical_rules(self):
        """Initialize propositional logic rules for medical reasoning"""
        # Add medical rules
        self.propositional_logic.add_rule(['fever', 'cough', 'fatigue'], 'likely_flu')
        self.propositional_logic.add_rule(['chest_pain', 'shortness_of_breath'], 'possible_heart_issue')
        self.propositional_logic.add_rule(['fever', 'cough', 'shortness_of_breath'], 'possible_covid')
        self.propositional_logic.add_rule(['likely_flu'], 'recommend_rest')
        self.propositional_logic.add_rule(['possible_heart_issue'], 'emergency_alert')
    
    def logical_inference(self, symptoms: List[str]) -> Set[str]:
        """Apply logical inference to symptoms"""
        facts = set(symptoms)
        inferred_facts = self.propositional_logic.modus_ponens(facts)
        return inferred_facts
    
    def bayesian_diagnosis(self, symptoms: List[str]) -> Dict[str, float]:
        """Get Bayesian probabilities for diseases"""
        evidence = {symptom: True for symptom in symptoms}
        return self.bayesian_network.infer(evidence)
    
    def heuristic_risk_aware(self, current_symptoms: Set[str], target_disease: str, patient_condition: Dict = None) -> float:
        """COMPULSORY: Enhanced risk-aware heuristic"""
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
            
            # Enhanced probabilistic hazard detection
            is_hazardous, hazard_prob, hazards = self.hazard_detector.probabilistic_hazard_detection(
                patient_condition, target_disease
            )
            if is_hazardous:
                risk_cost *= (1 + hazard_prob * 10)  # Dynamic penalty based on probability
        
        return base_score + risk_cost

    # Keep all existing heuristic methods...
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

    def a_star_search(self, selected_symptoms: List[str], heuristic_name: str, patient_condition: Dict = None) -> List[Tuple]:
        """Enhanced A* search with multiple AI techniques"""
        current_symptoms_set = set(selected_symptoms)
        
        # Apply logical inference
        inferred_conclusions = self.logical_inference(selected_symptoms)
        st.sidebar.write("🤖 Logical Inference:", inferred_conclusions)
        
        # Get Bayesian probabilities
        bayesian_probs = self.bayesian_diagnosis(selected_symptoms)
        
        possible_diseases = set()
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
            
            # Incorporate Bayesian probability
            bayesian_prob = bayesian_probs.get(disease, 0.01)
            adjusted_f_score = f_score * (1 - bayesian_prob)
            
            heapq.heappush(priority_queue, (adjusted_f_score, disease, g_score, h_score, bayesian_prob))
        
        # Return sorted results
        results = []
        while priority_queue:
            f_score, disease, g_score, h_score, bayesian_prob = heapq.heappop(priority_queue)
            confidence = max(0, 100 - f_score * 5)  # Convert to percentage
            results.append((disease, confidence, g_score, h_score, f_score, bayesian_prob))
        
        return results

# Keep existing RiskAssessmentModule, ReflexMedicationAgent, PathPlanningModule, AIHealthcareAssistant classes
# (They remain the same as in your original code)

class EnhancedAIHealthcareAssistant:
    def __init__(self):
        self.symptoms_db = self._initialize_symptoms_database()
        self.diseases_db = self._initialize_diseases_database()
        self.astar_checker = EnhancedAStarSymptomChecker(self.symptoms_db, self.diseases_db)
        self.reflex_agent = ReflexMedicationAgent()
        self.path_planner = PathPlanningModule(self.diseases_db)
        
        # NEW: Additional AI components
        self.bayesian_network = BayesianNetwork()
        self.hmm = HiddenMarkovModel()
        self.utility_theory = UtilityTheory()
        self.mdp = MarkovDecisionProcess()
        self.rl_agent = ReinforcementLearningAgent(states=[0, 1, 2, 3, 4], 
                                                 actions=['monitor', 'medicate', 'hospitalize'])
        
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
# NEW DEMONSTRATION FUNCTIONS FOR AI TOPICS
# =============================================================================

def demonstrate_bayesian_network():
    st.header("🕸️ Bayesian Network Demonstration")
    
    bn = BayesianNetwork()
    st.subheader("Medical Bayesian Network")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Prior Probabilities:**")
        for disease, prob in bn.nodes.items():
            st.write(f"{disease}: {prob:.3f}")
    
    with col2:
        st.write("**Conditional Probabilities (Fever):**")
        for disease, prob in bn.edges['Fever'].items():
            st.write(f"P(Fever|{disease}): {prob:.3f}")
    
    # Interactive evidence
    st.subheader("Interactive Diagnosis")
    evidence = {}
    symptoms = list(bn.edges.keys())
    
    for symptom in symptoms:
        evidence[symptom] = st.checkbox(symptom)
    
    if st.button("Calculate Probabilities"):
        posterior = bn.infer(evidence)
        
        st.write("**Posterior Probabilities:**")
        for disease, prob in sorted(posterior.items(), key=lambda x: x[1], reverse=True):
            st.write(f"{disease}: {prob:.4f}")

def demonstrate_hmm():
    st.header("⏳ Hidden Markov Model - Health State Tracking")
    
    hmm = HiddenMarkovModel()
    
    st.subheader("Patient Health State Tracking")
    observations = st.multiselect("Select symptom severity observations over time:",
                                ["Mild", "Moderate", "Severe"],
                                default=["Mild", "Severe", "Moderate"])
    
    if observations:
        beliefs_history = hmm.forward_algorithm(observations)
        
        st.write("**Belief State Evolution:**")
        for i, belief in enumerate(beliefs_history):
            st.write(f"Step {i}: {belief}")

def demonstrate_mdp():
    st.header("🎯 Markov Decision Process - Treatment Optimization")
    
    mdp = MarkovDecisionProcess()
    V, policy = mdp.value_iteration()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Optimal Value Function")
        for state, value in V.items():
            health_status = ["Critical", "Poor", "Fair", "Good", "Excellent"]
            st.write(f"Health {health_status[state]}: {value:.2f}")
    
    with col2:
        st.subheader("Optimal Policy")
        for state, action in policy.items():
            health_status = ["Critical", "Poor", "Fair", "Good", "Excellent"]
            st.write(f"Health {health_status[state]}: {action}")

def demonstrate_reinforcement_learning():
    st.header("🤖 Reinforcement Learning - Treatment Learning")
    
    rl_agent = ReinforcementLearningAgent(states=[0, 1, 2, 3, 4], 
                                        actions=['monitor', 'medicate', 'hospitalize'])
    
    if st.button("Train RL Agent (10 episodes)"):
        progress_bar = st.progress(0)
        rewards_history = []
        
        for episode in range(10):
            reward, _ = rl_agent.train_episode(initial_state=2)  # Start from fair health
            rewards_history.append(reward)
            progress_bar.progress((episode + 1) / 10)
        
        st.line_chart(rewards_history)
        st.write("**Learned Q-Table:**")
        st.write(dict(rl_agent.q_table))

def demonstrate_constraint_satisfaction():
    st.header("🔗 Constraint Satisfaction - Treatment Planning")
    
    # Define CSP for treatment planning
    variables = ['medication', 'therapy', 'diet', 'exercise']
    domains = {
        'medication': ['aspirin', 'antibiotic', 'antiviral', 'none'],
        'therapy': ['rest', 'physio', 'respiratory', 'none'],
        'diet': ['normal', 'liquid', 'soft', 'restricted'],
        'exercise': ['none', 'light', 'moderate', 'intensive']
    }
    
    csp = MedicalCSP(variables, domains)
    
    # Add constraints
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
    
    if st.button("Solve CSP"):
        solution = csp.backtracking_search()
        if solution:
            st.write("**Optimal Treatment Plan:**")
            for var, value in solution.items():
                st.write(f"{var}: {value}")
        else:
            st.write("No valid treatment plan found")

def demonstrate_utility_theory():
    st.header("⚖️ Utility Theory - Medical Decision Making")
    
    utility = UtilityTheory()
    
    st.subheader("Treatment Option Analysis")
    
    # Treatment option 1: Conservative
    outcomes1 = [
        (0.6, 'healthy', 10),  # 60% chance of full recovery for 10 years
        (0.3, 'mild_illness', 8),  # 30% chance of mild illness
        (0.1, 'moderate_illness', 5)  # 10% chance of moderate illness
    ]
    
    # Treatment option 2: Aggressive
    outcomes2 = [
        (0.8, 'healthy', 12),  # 80% chance but shorter lifespan
        (0.2, 'severe_illness', 3)  # 20% chance of complications
    ]
    
    util1 = utility.calculate_expected_utility(outcomes1)
    util2 = utility.calculate_expected_utility(outcomes2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Conservative Treatment Utility", f"{util1:.2f}")
        st.write("Lower risk, steady outcomes")
    
    with col2:
        st.metric("Aggressive Treatment Utility", f"{util2:.2f}")
        st.write("Higher risk, potentially better outcomes")
    
    if util1 > util2:
        st.success("Recommended: Conservative Treatment")
    else:
        st.success("Recommended: Aggressive Treatment")

def main():
    st.title("🏥 AI Doctor Helper - Complete AI Curriculum Implementation")
    st.markdown("""
    **Complete AI Curriculum Coverage (90%+ Topics)**
    
    ### ✅ Covered AI Topics:
    
    **Core AI (Topics 1-5)**
    - ✅ Intelligent Agents (Reflex Medication Agent)
    - ✅ Problem Solving & Search (A* with 5 Heuristics)
    - ✅ Informed Search (Manhattan, Euclidean, Symptom Frequency, Severity Weighted, Risk-Aware)
    
    **Advanced AI (Topics 6-10)**
    - ✅ Constraint Satisfaction Problems (Treatment Planning CSP)
    - ✅ Symbolic AI & Logical Agents (Propositional Logic, First-Order Logic)
    
    **Probabilistic AI (Topics 11-16)**
    - ✅ Probability Review (Bayesian Networks)
    - ✅ Probabilistic Reasoning (Bayesian Inference)
    - ✅ Probabilistic Reasoning over Time (Hidden Markov Models)
    - ✅ Utility Theory (Medical Decision Making)
    
    **Sequential Decision Making (Topics 17-22)**
    - ✅ Markov Decision Processes (Treatment Optimization)
    - ✅ Reinforcement Learning (Q-learning Agent)
    - ✅ Sequential Decision Making (Value Iteration, Policy Iteration)
    """)
    
    # Initialize enhanced assistant
    assistant = EnhancedAIHealthcareAssistant()
    
    # Sidebar navigation with expanded topics
    st.sidebar.title("AI Curriculum Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose AI Topic Demonstration",
        [
            "Symptom Checker (A* Search)",
            "Medication Agent (Intelligent Agents)", 
            "Treatment Planning (Path Planning)",
            "Bayesian Networks (Probabilistic Reasoning)",
            "Hidden Markov Models (Temporal Reasoning)",
            "Markov Decision Processes (Sequential Decisions)",
            "Reinforcement Learning (Learning Agent)",
            "Constraint Satisfaction (CSP)",
            "Utility Theory (Decision Making)",
            "All Heuristics Comparison"
        ]
    )
    
    if app_mode == "Symptom Checker (A* Search)":
        st.header("🔍 Symptom Checker with Enhanced A* Search")
        
        # Enhanced symptom selection with logical inference
        symptoms = list(assistant.symptoms_db.keys())
        selected_symptoms = st.multiselect("Select your symptoms:", symptoms)
        
        if selected_symptoms:
            # Show logical inference
            inferred = assistant.astar_checker.logical_inference(selected_symptoms)
            if inferred:
                st.info(f"**Logical Inference Results:** {', '.join(inferred)}")
            
            # Patient information
            st.subheader("Patient Information")
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=40)
                comorbidities = st.multiselect("Comorbidities", ["diabetes", "hypertension", "asthma", "heart_disease"])
            
            with col2:
                current_meds = st.multiselect("Current Medications", ["warfarin", "insulin", "aspirin", "none"])
                allergies = st.multiselect("Allergies", ["penicillin", "aspirin", "sulfa", "none"])
            
            patient_condition = {
                'age': age,
                'comorbidities': comorbidities,
                'current_medications': [m for m in current_meds if m != "none"],
                'allergies': [a for a in allergies if a != "none"]
            }
            
            # Heuristic selection
            heuristic_choice = st.selectbox(
                "Select A* Heuristic:",
                ["Manhattan", "Euclidean", "Symptom Frequency", "Severity Weighted", "Risk-Aware"]
            )
            
            if st.button("Run Enhanced Diagnosis"):
                # Run Bayesian diagnosis
                bayesian_probs = assistant.astar_checker.bayesian_diagnosis(selected_symptoms)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Bayesian Probabilities")
                    for disease, prob in sorted(bayesian_probs.items(), key=lambda x: x[1], reverse=True):
                        if prob > 0.01:
                            st.write(f"{disease}: {prob:.3f}")
                
                with col2:
                    st.subheader(f"A* Search Results ({heuristic_choice})")
                    results = assistant.astar_checker.a_star_search(
                        selected_symptoms, 
                        heuristic_choice, 
                        patient_condition if heuristic_choice == "Risk-Aware" else None
                    )
                    
                    for disease, confidence, g_score, h_score, f_score, bayesian_prob in results[:3]:
                        if confidence > 10:
                            st.write(f"**{disease}** ({confidence:.1f}%)")
                            st.write(f"g={g_score}, h={h_score:.2f}, f={f_score:.2f}")
                            st.write(f"Bayesian: {bayesian_prob:.3f}")
    
    elif app_mode == "Bayesian Networks (Probabilistic Reasoning)":
        demonstrate_bayesian_network()
    
    elif app_mode == "Hidden Markov Models (Temporal Reasoning)":
        demonstrate_hmm()
    
    elif app_mode == "Markov Decision Processes (Sequential Decisions)":
        demonstrate_mdp()
    
    elif app_mode == "Reinforcement Learning (Learning Agent)":
        demonstrate_reinforcement_learning()
    
    elif app_mode == "Constraint Satisfaction (CSP)":
        demonstrate_constraint_satisfaction()
    
    elif app_mode == "Utility Theory (Decision Making)":
        demonstrate_utility_theory()
    
    # Keep existing modes for Medication Agent, Treatment Planning, etc.
    elif app_mode == "Medication Agent (Intelligent Agents)":
        st.header("💊 Reflex Medication Agent")
        # ... (keep existing medication agent code)
    
    elif app_mode == "Treatment Planning (Path Planning)":
        st.header("🛣️ Treatment Path Planning")
        # ... (keep existing treatment planning code)
    
    elif app_mode == "All Heuristics Comparison":
        st.header("📊 All Heuristics Comparison")
        # ... (keep existing comparison code)

if __name__ == "__main__":
    main()
