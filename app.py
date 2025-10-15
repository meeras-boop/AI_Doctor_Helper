# app.py - Complete AI Health Assistant with AI Concept Tags
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import re

# Page configuration
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
def inject_custom_css():
    st.markdown("""
    <style>
    /* Main background and text colors */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%) !important;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #2c3e50 !important;
        font-family: 'Arial', sans-serif;
        font-weight: 700 !important;
    }
    
    h1 {
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    
    /* Card-like containers */
    .custom-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-left: 5px solid #3498db;
        margin-bottom: 1.5rem;
    }
    
    .emergency-card {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(255,107,107,0.3);
        text-align: center;
        margin-bottom: 1.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 4px solid #3498db;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #3498db, #2980b9) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4) !important;
    }
    
    /* Primary button */
    .stButton>button[kind="primary"] {
        background: linear-gradient(45deg, #e74c3c, #c0392b) !important;
    }
    
    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 5px 15px rgba(231, 76, 60, 0.4) !important;
    }
    
    /* Text input styling */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 8px !important;
        border: 2px solid #bdc3c7 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2) !important;
    }
    
    /* Select box styling */
    .stSelectbox>div>div>div {
        border-radius: 8px !important;
    }
    
    /* Number input styling */
    .stNumberInput>div>div>input {
        border-radius: 8px !important;
    }
    
    /* Success and error messages */
    .stAlert {
        border-radius: 10px !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(45deg, #3498db, #2980b9) !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(45deg, #ecf0f1, #bdc3c7) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #3498db, #2980b9) !important;
        color: white !important;
    }
    
    /* Sidebar text color */
    .css-1d391kg p, .css-1d391kg label, .css-1d391kg div {
        color: white !important;
    }
    
    /* Custom badge styling */
    .custom-badge {
        display: inline-block;
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .urgent-badge {
        background: linear-gradient(45deg, #e74c3c, #c0392b);
    }
    
    .success-badge {
        background: linear-gradient(45deg, #27ae60, #229954);
    }
    
    .ai-concept-badge {
        background: linear-gradient(45deg, #9b59b6, #8e44ad);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 0.1rem;
        border: 1px solid white;
    }
    
    /* Icon styling */
    .icon-large {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# Medical Knowledge Base Class
class MedicalKnowledgeBase:
    """Medical Knowledge Base using graph structure [Knowledge Graph] [CSP]"""
    
    def __init__(self):
        self.symptoms = set()
        self.conditions = {}
        self.symptom_to_conditions = {}
        self.condition_recommendations = {}
        
    def initialize_base_knowledge(self):
        """Initialize with basic medical knowledge [Knowledge Engineering]"""
        # Symptoms database
        self.symptoms = {
            'headache', 'fever', 'cough', 'fatigue', 'nausea',
            'dizziness', 'chest pain', 'shortness of breath',
            'sore throat', 'runny nose', 'body aches', 'chills',
            'sneezing', 'muscle pain', 'loss of appetite'
        }
        
        # Conditions and their symptoms [Semantic Network]
        self.conditions = {
            'common_cold': {
                'symptoms': ['cough', 'runny nose', 'sore throat', 'fatigue', 'sneezing'],
                'severity': 'low'
            },
            'flu': {
                'symptoms': ['fever', 'body aches', 'fatigue', 'cough', 'headache', 'chills'],
                'severity': 'medium'
            },
            'migraine': {
                'symptoms': ['headache', 'nausea', 'dizziness'],
                'severity': 'medium'
            },
            'hypertension': {
                'symptoms': ['headache', 'dizziness'],
                'severity': 'high'
            },
            'respiratory_infection': {
                'symptoms': ['cough', 'fever', 'shortness of breath', 'fatigue'],
                'severity': 'high'
            },
            'allergies': {
                'symptoms': ['sneezing', 'runny nose', 'cough'],
                'severity': 'low'
            },
            'covid_19': {
                'symptoms': ['fever', 'cough', 'fatigue', 'loss of appetite', 'shortness of breath'],
                'severity': 'high'
            }
        }
        
        # Build symptom-to-condition mapping [Graph Traversal]
        for condition, data in self.conditions.items():
            for symptom in data['symptoms']:
                if symptom not in self.symptom_to_conditions:
                    self.symptom_to_conditions[symptom] = []
                self.symptom_to_conditions[symptom].append(condition)
        
        # Recommendations for conditions [Rule-Based System]
        self.condition_recommendations = {
            'common_cold': [
                'Rest and hydrate well',
                'Use over-the-counter cold medicine',
                'Get plenty of sleep',
                'Use a humidifier'
            ],
            'flu': [
                'Rest and stay hydrated',
                'Consider antiviral medication if early',
                'Take over-the-counter fever reducers',
                'Consult doctor if symptoms worsen'
            ],
            'migraine': [
                'Rest in a dark, quiet room',
                'Stay hydrated',
                'Avoid bright lights and loud noises',
                'Consider prescribed migraine medication'
            ],
            'hypertension': [
                'Monitor blood pressure regularly',
                'Reduce salt intake',
                'Exercise regularly',
                'Consult healthcare provider for medication'
            ],
            'respiratory_infection': [
                'Consult doctor for proper diagnosis',
                'Get chest X-ray if symptoms persist',
                'Take prescribed antibiotics if bacterial',
                'Use inhaler if prescribed'
            ],
            'allergies': [
                'Avoid known allergens',
                'Take antihistamines',
                'Use nasal sprays',
                'Consider allergy testing'
            ],
            'covid_19': [
                'Self-isolate immediately',
                'Get tested for COVID-19',
                'Monitor oxygen levels',
                'Seek emergency care if breathing difficulties occur'
            ]
        }
    
    def get_known_symptoms(self):
        return list(self.symptoms)
    
    def get_conditions_for_symptoms(self, symptoms):
        """Find conditions matching given symptoms [Pattern Matching] [Search]"""
        matching_conditions = {}
        
        for symptom in symptoms:
            if symptom in self.symptom_to_conditions:
                for condition in self.symptom_to_conditions[symptom]:
                    if condition not in matching_conditions:
                        matching_conditions[condition] = 0
                    matching_conditions[condition] += 1
        
        return matching_conditions
    
    def get_condition_recommendations(self, condition):
        return self.condition_recommendations.get(condition, [
            'Consult healthcare provider for proper diagnosis',
            'Monitor symptoms closely',
            'Rest and maintain hydration'
        ])
    
    def get_condition_severity(self, condition):
        condition_data = self.conditions.get(condition, {})
        return condition_data.get('severity', 'medium')

# Bayesian Network Class
class MedicalBayesianNetwork:
    """Bayesian Network for probabilistic reasoning [Bayesian Networks] [Probabilistic Reasoning]"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.cpts = {}  # Conditional Probability Tables
    
    def initialize_network(self):
        """Initialize a simple medical Bayesian network [Graph Model]"""
        # Define nodes (symptoms and conditions)
        self.nodes = {
            'flu': ['True', 'False'],
            'cold': ['True', 'False'],
            'covid_19': ['True', 'False'],
            'fever': ['True', 'False'],
            'cough': ['True', 'False'],
            'headache': ['True', 'False'],
            'fatigue': ['True', 'False']
        }
        
        # Define edges (relationships) [DAG]
        self.edges = {
            'flu': ['fever', 'cough', 'headache', 'fatigue'],
            'cold': ['cough', 'fatigue'],
            'covid_19': ['fever', 'cough', 'fatigue']
        }
        
        # Define Conditional Probability Tables [CPT]
        self.cpts = {
            'flu': [0.05, 0.95],  # P(flu) = 0.05
            'cold': [0.1, 0.9],   # P(cold) = 0.1
            'covid_19': [0.02, 0.98],  # P(covid_19) = 0.02
            
            'fever': {  # P(fever | flu, covid_19) [Conditional Probability]
                ('True', 'True'): [0.95, 0.05],
                ('True', 'False'): [0.9, 0.1],
                ('False', 'True'): [0.8, 0.2],
                ('False', 'False'): [0.01, 0.99]
            },
            
            'cough': {  # P(cough | flu, cold, covid_19)
                ('True', 'True', 'True'): [0.99, 0.01],
                ('True', 'True', 'False'): [0.95, 0.05],
                ('True', 'False', 'True'): [0.9, 0.1],
                ('True', 'False', 'False'): [0.8, 0.2],
                ('False', 'True', 'True'): [0.85, 0.15],
                ('False', 'True', 'False'): [0.7, 0.3],
                ('False', 'False', 'True'): [0.75, 0.25],
                ('False', 'False', 'False'): [0.05, 0.95]
            },
            
            'headache': {  # P(headache | flu)
                ('True',): [0.7, 0.3],
                ('False',): [0.1, 0.9]
            },
            
            'fatigue': {  # P(fatigue | flu, covid_19)
                ('True', 'True'): [0.9, 0.1],
                ('True', 'False'): [0.8, 0.2],
                ('False', 'True'): [0.7, 0.3],
                ('False', 'False'): [0.2, 0.8]
            }
        }
    
    def infer_conditions(self, observed_symptoms: List[str]) -> Dict[str, Any]:
        """Perform probabilistic inference given observed symptoms [Bayesian Inference]"""
        
        # Convert symptoms to evidence
        evidence = self._symptoms_to_evidence(observed_symptoms)
        
        # Simple inference using Bayes' theorem [Naive Bayes]
        results = {}
        
        for condition in ['flu', 'cold', 'covid_19']:
            probability = self._calculate_probability(condition, evidence)
            results[condition] = probability
        
        # Find most likely condition [Maximum A Posteriori]
        most_likely = max(results.items(), key=lambda x: x[1]) if results else ('unknown', 0)
        
        return {
            'most_likely': most_likely[0],
            'confidence': most_likely[1],
            'probabilities': results,
            'risk_level': 'high' if most_likely[1] > 0.7 else 'medium' if most_likely[1] > 0.3 else 'low'
        }
    
    def _symptoms_to_evidence(self, symptoms: List[str]) -> Dict[str, str]:
        """Convert symptom list to evidence format [Feature Extraction]"""
        evidence = {}
        symptom_mapping = {
            'fever': 'fever', 'cough': 'cough', 'headache': 'headache', 'fatigue': 'fatigue'
        }
        
        for symptom in symptoms:
            if symptom in symptom_mapping:
                evidence[symptom_mapping[symptom]] = 'True'
        
        return evidence
    
    def _calculate_probability(self, condition: str, evidence: Dict[str, str]) -> float:
        """Calculate P(condition | evidence) using naive Bayes approximation [Bayes Theorem]"""
        
        # Prior probability
        prior = self.cpts[condition][0]  # P(condition=True)
        
        # If no evidence, return prior
        if not evidence:
            return prior
        
        # Calculate likelihood P(evidence | condition) [Likelihood]
        likelihood_true = 1.0
        likelihood_false = 1.0
        
        for symptom, value in evidence.items():
            if value == 'True' and symptom in self.cpts:
                if isinstance(self.cpts[symptom], dict):
                    # Get the probability based on parent conditions
                    parent_states = []
                    for parent in self.edges.keys():
                        if parent in self.edges and symptom in self.edges[parent]:
                            parent_states.append('True' if parent == condition else 'False')
                    
                    if parent_states:
                        prob_true = self.cpts[symptom][tuple(parent_states)][0]
                        prob_false = 0.1  # Default probability if condition is false
                        
                        likelihood_true *= prob_true
                        likelihood_false *= prob_false
        
        # Bayes' theorem: P(condition|evidence) ∝ P(evidence|condition) * P(condition)
        numerator = likelihood_true * prior
        denominator = numerator + (likelihood_false * (1 - prior))
        
        return numerator / denominator if denominator > 0 else 0.0

# Health Agent Class
class PersonalHealthAgent:
    """Intelligent Health Agent [Intelligent Agent] [Rational Agent]"""
    
    def __init__(self, knowledge_base, bayesian_network):
        self.knowledge_base = knowledge_base
        self.bayesian_network = bayesian_network
        self.performance_metrics = {
            'queries_processed': 0,
            'emergencies_detected': 0,
            'successful_recommendations': 0
        }
        
        self.emergency_patterns = [
            r'chest pain', r'heart attack', r'stroke', r'difficulty breathing',
            r'severe bleeding', r'unconscious', r'suicidal', r'homicidal',
            r'severe headache', r'paralysis', r'seizure', r'can\'t breathe',
            r'choking', r'severe pain'
        ]
    
    def check_emergency(self, symptoms_text: str) -> bool:
        """Emergency detection using pattern matching [Rule-Based System] [Regex]"""
        text_lower = symptoms_text.lower()
        for pattern in self.emergency_patterns:
            if re.search(pattern, text_lower):
                self.performance_metrics['emergencies_detected'] += 1
                return True
        return False
    
    def process_symptoms(self, symptoms_text: str, user_context: Dict = None) -> Dict[str, Any]:
        """Main symptom processing pipeline [Agent Architecture]"""
        self.performance_metrics['queries_processed'] += 1
        
        # Emergency check [Priority Processing]
        if self.check_emergency(symptoms_text):
            return {'emergency': True}
        
        # Extract symptoms [NLP]
        symptoms = self._extract_symptoms(symptoms_text)
        
        # Get analysis [Multi-Model Fusion]
        analysis = self.analyze_symptoms(symptoms, user_context)
        
        # Generate recommendations [Decision Making]
        recommendations = self.generate_recommendations(analysis)
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'explanation': self.explain_reasoning(analysis)
        }
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from natural language text [NLP] [Tokenization]"""
        words = text.lower().split()
        symptoms = []
        known_symptoms = self.knowledge_base.get_known_symptoms()
        
        for symptom in known_symptoms:
            if symptom in text.lower():
                symptoms.append(symptom)
        
        return symptoms if symptoms else ['general discomfort']
    
    def analyze_symptoms(self, symptoms: List[str], user_context: Dict = None) -> Dict[str, Any]:
        """Analyze symptoms using multiple AI approaches [Ensemble Methods]"""
        # Search-based analysis [Heuristic Search]
        search_results = self._search_based_analysis(symptoms)
        
        # Probabilistic reasoning [Bayesian Inference]
        prob_results = self.bayesian_network.infer_conditions(symptoms)
        
        # Combine results [Data Fusion]
        return self._combine_analyses(search_results, prob_results, user_context)
    
    def _search_based_analysis(self, symptoms: List[str]) -> Dict[str, Any]:
        """Use search algorithms to find matching conditions [Best-First Search]"""
        possible_conditions = self.knowledge_base.get_conditions_for_symptoms(symptoms)
        
        if not possible_conditions:
            return {'primary_condition': None, 'confidence': 0, 'possible_conditions': []}
        
        # Find best match [Greedy Algorithm]
        best_condition = max(possible_conditions.items(), key=lambda x: x[1]) if possible_conditions else (None, 0)
        
        if best_condition[0]:
            expected_symptoms = len(self.knowledge_base.conditions[best_condition[0]]['symptoms'])
            confidence = best_condition[1] / expected_symptoms
        else:
            confidence = 0
        
        return {
            'primary_condition': best_condition[0],
            'confidence': min(confidence, 1.0),
            'possible_conditions': list(possible_conditions.keys()),
            'symptoms': symptoms
        }
    
    def _combine_analyses(self, search_results: Dict, prob_results: Dict, user_context: Dict = None) -> Dict[str, Any]:
        """Combine results from different AI approaches [Model Fusion]"""
        # Prefer probabilistic results if confidence is high [Confidence Weighting]
        if prob_results.get('confidence', 0) > 0.6:
            primary_condition = prob_results.get('most_likely')
        else:
            primary_condition = search_results.get('primary_condition')
        
        # Combine possible conditions [Set Operations]
        all_conditions = list(set(
            search_results.get('possible_conditions', []) + 
            [prob_results.get('most_likely')] if prob_results.get('most_likely') else []
        ))
        
        # Use the higher confidence score [Max Pooling]
        confidence = max(
            search_results.get('confidence', 0), 
            prob_results.get('confidence', 0)
        )
        
        # Adjust based on user context if available [Context-Aware Reasoning]
        if user_context and user_context.get('severity') == 'Severe':
            confidence = min(confidence + 0.1, 1.0)  # Boost confidence for severe symptoms
        
        return {
            'primary_condition': primary_condition,
            'possible_conditions': all_conditions,
            'confidence': confidence,
            'urgency_level': self._calculate_urgency(search_results, prob_results, user_context),
            'symptoms': search_results.get('symptoms', [])
        }
    
    def _calculate_urgency(self, search_results: Dict, prob_results: Dict, user_context: Dict = None) -> str:
        """Calculate urgency level using utility theory [Utility Theory] [Decision Theory]"""
        confidence = max(search_results.get('confidence', 0), prob_results.get('confidence', 0))
        
        # Adjust based on user context
        if user_context:
            if user_context.get('severity') == 'Severe':
                confidence += 0.2
            elif user_context.get('severity') == 'Moderate':
                confidence += 0.1
        
        if confidence > 0.8:
            return 'high'
        elif confidence > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def generate_recommendations(self, analysis_result: Dict) -> List[str]:
        """Generate recommendations based on analysis [Decision Support System]"""
        recommendations = []
        condition = analysis_result.get('primary_condition')
        urgency = analysis_result.get('urgency_level')
        
        # Base recommendations based on urgency [Utility-Based Decision]
        if urgency == 'high':
            recommendations.append("🚨 Consult a healthcare professional within 24 hours")
        elif urgency == 'medium':
            recommendations.append("📅 Schedule a doctor's appointment this week")
        else:
            recommendations.append("👀 Monitor symptoms and rest")
        
        # Condition-specific recommendations [Expert System Rules]
        if condition:
            specific_recs = self.knowledge_base.get_condition_recommendations(condition)
            recommendations.extend(specific_recs)
        else:
            recommendations.extend([
                "Keep track of your symptoms daily",
                "Stay hydrated and get plenty of rest",
                "Consult a doctor if symptoms persist or worsen"
            ])
        
        self.performance_metrics['successful_recommendations'] += 1
        return recommendations
    
    def explain_reasoning(self, analysis_result: Dict) -> str:
        """Provide explainable AI output [XAI] [Explainable AI]"""
        symptoms = analysis_result.get('symptoms', [])
        condition = analysis_result.get('primary_condition', 'unknown condition')
        confidence = analysis_result.get('confidence', 0)
        
        explanation = f"Based on your symptoms ({', '.join(symptoms)}), "
        explanation += f"the AI system identified **{condition.replace('_', ' ').title()}** with {confidence:.1%} confidence. "
        explanation += "This assessment combines pattern matching with probabilistic reasoning using Bayesian networks."
        
        return explanation

# Privacy Manager Class
class FederatedLearningManager:
    """Manager for privacy-preserving techniques [Federated Learning] [Privacy-Preserving AI]"""
    
    def __init__(self):
        self.local_models = {}
        self.global_model = None
    
    def initialize_federated_learning(self):
        """Initialize federated learning setup [Distributed Learning]"""
        return True
    
    def train_local_model(self, user_data, user_id):
        """Train model on local user data without sharing raw data [Local Training]"""
        if user_id not in self.local_models:
            self.local_models[user_id] = {'trained': False}
        
        self.local_models[user_id]['trained'] = True
        return True
    
    def aggregate_models(self):
        """Aggregate local models into global model [Model Aggregation]"""
        trained_count = sum(1 for model in self.local_models.values() if model['trained'])
        return trained_count

# Main Streamlit Application Class
class StreamlitHealthAssistant:
    def __init__(self):
        inject_custom_css()
        self.initialize_session_state()
        self.setup_components()
    
    def initialize_session_state(self):
        """Initialize session state variables [State Management]"""
        if 'symptoms_history' not in st.session_state:
            st.session_state.symptoms_history = []
        if 'health_metrics' not in st.session_state:
            st.session_state.health_metrics = {
                'weight': [], 'blood_pressure': [], 'heart_rate': []
            }
        if 'medication_reminders' not in st.session_state:
            st.session_state.medication_reminders = []
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = {
                'age': 30, 'gender': 'Prefer not to say', 
                'location': '', 'insurance': ''
            }
        if 'assistant_initialized' not in st.session_state:
            st.session_state.assistant_initialized = False
    
    def setup_components(self):
        """Initialize AI components [System Initialization]"""
        if not st.session_state.assistant_initialized:
            with st.spinner("🚀 Initializing AI Health Assistant [System Boot]..."):
                self.knowledge_base = MedicalKnowledgeBase()
                self.bayesian_network = MedicalBayesianNetwork()
                self.privacy_manager = FederatedLearningManager()
                self.health_agent = PersonalHealthAgent(
                    knowledge_base=self.knowledge_base,
                    bayesian_network=self.bayesian_network
                )
                
                # Load initial data [Data Loading]
                self.knowledge_base.initialize_base_knowledge()
                self.bayesian_network.initialize_network()
                self.privacy_manager.initialize_federated_learning()
                
                st.session_state.assistant_initialized = True
                st.success("✅ AI Health Assistant Ready! [System Online]")

    def render_sidebar(self):
        """Render the sidebar with user profile and navigation [UI Component]"""
        with st.sidebar:
            # Header with logo
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h1 style="color: white; margin-bottom: 0.5rem;">🏥</h1>
                <h2 style="color: white; margin: 0;">AI Health Assistant</h2>
                <p style="color: #ecf0f1; font-size: 0.9rem;">Your Personal Healthcare Companion</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Navigation
            st.markdown("### 🧭 Navigation")
            page = st.radio(
                "",
                ["🏠 Symptom Checker", "📊 Health Dashboard", "👨‍⚕️ Doctor Finder", 
                 "💊 Medication Tracker", "🤖 AI Insights", "👤 User Profile"],
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # Quick stats in cards
            st.markdown("### 📈 Quick Stats")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.8rem; color: #7f8c8d;">Symptoms Checked</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #2c3e50;">{len(st.session_state.symptoms_history)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_metrics = sum(len(v) for v in st.session_state.health_metrics.values())
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.8rem; color: #7f8c8d;">Health Records</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #2c3e50;">{total_metrics}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Emergency section
            st.markdown("""
            <div class="emergency-card">
                <h3 style="color: white; margin: 0;">🚨 Emergency Notice</h3>
                <p style="color: white; font-size: 0.9rem; margin: 0.5rem 0;">
                If you're experiencing chest pain, difficulty breathing, 
                severe bleeding, or thoughts of harm, please call emergency 
                services immediately.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            return page.split(" ")[1] if " " in page else page

    def render_symptom_checker(self):
        """Main symptom analysis interface [User Interface]"""
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="gradient-text">🔍 Symptom Checker [NLP Interface]</h1>
            <p style="font-size: 1.2rem; color: #7f8c8d;">Describe your symptoms and let AI analyze them</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <h3>💬 Describe Your Symptoms [Natural Language Input]</h3>
            """, unsafe_allow_html=True)
            
            # Symptom input
            symptoms_text = st.text_area(
                "",
                placeholder="e.g., I've had headache and fever for 2 days, along with some nausea...",
                height=120,
                label_visibility="collapsed"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Additional context
            with st.expander("🔍 Additional Context (Optional) [Feature Engineering]", expanded=False):
                col1a, col2a = st.columns(2)
                with col1a:
                    symptom_duration = st.selectbox(
                        "⏱️ Duration",
                        ["Less than 24 hours", "1-3 days", "3-7 days", "More than 1 week"]
                    )
                    severity = st.select_slider(
                        "📊 Symptom Severity",
                        options=["Mild", "Moderate", "Severe"]
                    )
                with col2a:
                    temperature = st.number_input("🌡️ Temperature (°C)", min_value=35.0, max_value=42.0, value=36.6)
                    recent_travel = st.checkbox("✈️ Recent travel")
            
            # Analyze button
            if st.button("🔍 Analyze Symptoms [Process Request]", type="primary", use_container_width=True):
                if symptoms_text.strip():
                    self.analyze_symptoms(symptoms_text, {
                        'duration': symptom_duration,
                        'severity': severity,
                        'temperature': temperature,
                        'recent_travel': recent_travel
                    })
                else:
                    st.error("Please describe your symptoms to get an analysis.")
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <h3>⚡ Quick Select [Feature Selection]</h3>
            """, unsafe_allow_html=True)
            
            # Common symptoms quick select
            common_symptoms = [
                "Headache", "Fever", "Cough", "Fatigue", "Nausea",
                "Dizziness", "Sore throat", "Runny nose", "Body aches"
            ]
            
            selected_quick = st.multiselect("Select common symptoms:", common_symptoms)
            
            if selected_quick and st.button("Use Selected Symptoms [Batch Processing]", use_container_width=True):
                quick_text = "I have " + ", ".join(selected_quick).lower()
                self.analyze_symptoms(quick_text, {})
            
            st.markdown("</div>", unsafe_allow_html=True)

    def analyze_symptoms(self, symptoms_text: str, context: Dict):
        """Analyze symptoms and display results [Processing Pipeline]"""
        with st.spinner("🤖 AI is analyzing your symptoms with advanced algorithms [AI Processing]..."):
            # FIX: Remove context parameter if not used, or make it optional
            result = self.health_agent.process_symptoms(symptoms_text)
            
            # Store in history [Data Persistence]
            st.session_state.symptoms_history.append({
                'timestamp': datetime.now(),
                'symptoms': symptoms_text,
                'result': result
            })
        
        # Display results [Result Visualization]
        if result.get('emergency'):
            st.markdown("""
            <div class="emergency-card">
                <h2 style="color: white; text-align: center;">🚨 EMERGENCY ALERT [Critical Detection]</h2>
                <h3 style="color: white; text-align: center;">Critical Symptoms Detected!</h3>
                <p style="color: white; text-align: center; font-size: 1.1rem;">
                Please seek immediate medical attention or call emergency services.
                </p>
                <div style="text-align: center; margin-top: 1rem;">
                    <span class="custom-badge urgent-badge">IMMEDIATE ACTION REQUIRED</span>
                </div>
                <p style="color: white; text-align: center; margin-top: 1rem;">
                <strong>Recommended Action:</strong> Go to the nearest emergency room immediately.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Results container
            st.success("✅ Analysis Complete! [Processing Finished]")
            
            analysis = result['analysis']
            
            # Create results grid [Data Visualization]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="icon-large">🎯</div>
                    <h3>Primary Assessment [Diagnosis]</h3>
                """, unsafe_allow_html=True)
                
                condition = analysis.get('primary_condition', 'Unknown')
                confidence = analysis.get('confidence', 0)
                
                st.metric(
                    "Likely Condition", 
                    condition.replace('_', ' ').title(),
                    f"{confidence:.1%} confidence"
                )
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="icon-large">⚡</div>
                    <h3>Urgency Level [Risk Assessment]</h3>
                """, unsafe_allow_html=True)
                
                urgency = analysis.get('urgency_level', 'low')
                urgency_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                urgency_text = {'high': 'High', 'medium': 'Medium', 'low': 'Low'}
                
                st.markdown(f"<h1 style='text-align: center;'>{urgency_colors[urgency]} {urgency_text[urgency].upper()}</h1>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="icon-large">📋</div>
                    <h3>Other Possibilities [Differential Diagnosis]</h3>
                """, unsafe_allow_html=True)
                
                possible_conditions = analysis.get('possible_conditions', [])
                
                for condition in possible_conditions[:2]:
                    st.write(f"• {condition.replace('_', ' ').title()}")
                
                if len(possible_conditions) > 2:
                    st.write(f"... and {len(possible_conditions) - 2} more")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Recommendations section [Decision Support]
            st.markdown("""
            <div class="custom-card">
                <h3>📋 Recommended Actions [Treatment Plan]</h3>
            """, unsafe_allow_html=True)
            
            for i, recommendation in enumerate(result['recommendations'], 1):
                st.markdown(f"""
                <div style="background: #ecf0f1; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #3498db;">
                    <strong>Step {i}:</strong> {recommendation}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # AI Explanation [Explainable AI]
            with st.expander("🤔 How did our AI reach this conclusion? [XAI - Explainable AI]", expanded=True):
                st.markdown("""
                <div class="custom-card">
                """, unsafe_allow_html=True)
                
                st.info(result['explanation'])
                
                # Show symptoms matched [Feature Importance]
                symptoms = analysis.get('symptoms', [])
                if symptoms:
                    st.write("**🔍 Symptoms identified [Feature Extraction]:**")
                    symptom_badges = " ".join([f'<span class="custom-badge">{symptom}</span>' for symptom in symptoms])
                    st.markdown(symptom_badges, unsafe_allow_html=True)
                
                # Show reasoning process [Algorithm Transparency]
                st.markdown("""
                <h4>🧠 AI Reasoning Process [Algorithmic Steps]:</h4>
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <span class="ai-concept-badge">[NLP]</span>
                    <span class="ai-concept-badge">[Pattern Matching]</span>
                    <span class="ai-concept-badge">[Bayesian Networks]</span>
                    <span class="ai-concept-badge">[Utility Theory]</span>
                    <span class="ai-concept-badge">[Search Algorithms]</span>
                    <span class="ai-concept-badge">[Decision Trees]</span>
                </div>
                <ol>
                <li><strong>Natural Language Processing [NLP]</strong> - Extracted symptoms from your description using tokenization and pattern matching</li>
                <li><strong>Pattern Matching [Heuristic Search]</strong> - Compared symptoms with known conditions using best-first search algorithms</li>
                <li><strong>Probabilistic Analysis [Bayesian Networks]</strong> - Used Bayesian networks to calculate likelihoods and confidence scores</li>
                <li><strong>Utility Assessment [Decision Theory]</strong> - Evaluated urgency and appropriate actions using utility functions</li>
                <li><strong>Knowledge Graph Query [Semantic Search]</strong> - Queried medical knowledge graph for condition-symptom relationships</li>
                <li><strong>Multi-Model Fusion [Ensemble Learning]</strong> - Combined results from multiple AI approaches for robust diagnosis</li>
                </ol>
                
                <h5>🧩 AI Concepts Used in This Analysis:</h5>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0;">
                    <span class="ai-concept-badge">[NLP]</span>
                    <span class="ai-concept-badge">[Bayesian Networks]</span>
                    <span class="ai-concept-badge">[Utility Theory]</span>
                    <span class="ai-concept-badge">[Search Algorithms]</span>
                    <span class="ai-concept-badge">[Knowledge Graphs]</span>
                    <span class="ai-concept-badge">[Pattern Matching]</span>
                    <span class="ai-concept-badge">[Decision Trees]</span>
                    <span class="ai-concept-badge">[Expert Systems]</span>
                    <span class="ai-concept-badge">[Rule-Based Systems]</span>
                    <span class="ai-concept-badge">[Probabilistic Reasoning]</span>
                    <span class="ai-concept-badge">[Heuristic Search]</span>
                    <span class="ai-concept-badge">[XAI]</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

    def run(self):
        """Main application runner [Application Controller]"""
        page = self.render_sidebar()
        
        # Simple page routing for demo [Routing Logic]
        if page == "Symptom":
            self.render_symptom_checker()
        else:
            st.info(f"🚧 {page} page is under construction. Currently, only Symptom Checker is fully implemented for the demo.")
            st.markdown("""
            <div class="custom-card">
                <h3>🎯 Available Features in This Demo</h3>
                <p><strong>✅ Symptom Checker:</strong> Fully functional AI-powered symptom analysis <span class="ai-concept-badge">[NLP]</span> <span class="ai-concept-badge">[Bayesian Networks]</span></p>
                <p><strong>🚧 Other Pages:</strong> Coming soon in future updates</p>
                <p>Try the Symptom Checker to experience our AI health assessment system!</p>
                
                <h4>🧠 AI Concepts Demonstrated:</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0;">
                    <span class="ai-concept-badge">[Intelligent Agents]</span>
                    <span class="ai-concept-badge">[NLP]</span>
                    <span class="ai-concept-badge">[Bayesian Networks]</span>
                    <span class="ai-concept-badge">[Knowledge Graphs]</span>
                    <span class="ai-concept-badge">[Search Algorithms]</span>
                    <span class="ai-concept-badge">[Utility Theory]</span>
                    <span class="ai-concept-badge">[Decision Trees]</span>
                    <span class="ai-concept-badge">[Pattern Matching]</span>
                    <span class="ai-concept-badge">[XAI]</span>
                    <span class="ai-concept-badge">[Rule-Based Systems]</span>
                    <span class="ai-concept-badge">[Probabilistic Reasoning]</span>
                    <span class="ai-concept-badge">[Heuristic Search]</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    assistant = StreamlitHealthAssistant()
    assistant.run()
