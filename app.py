# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import re

# Import our modules
from knowledge.medical_kb import MedicalKnowledgeBase
from reasoning.bayesian_network import MedicalBayesianNetwork
from agents.health_agent import PersonalHealthAgent
from utils.privacy import FederatedLearningManager

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

class StreamlitHealthAssistant:
    def __init__(self):
        inject_custom_css()
        self.initialize_session_state()
        self.setup_components()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
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
        """Initialize AI components"""
        if not st.session_state.assistant_initialized:
            with st.spinner("🚀 Initializing AI Health Assistant..."):
                self.knowledge_base = MedicalKnowledgeBase()
                self.bayesian_network = MedicalBayesianNetwork()
                self.privacy_manager = FederatedLearningManager()
                self.health_agent = PersonalHealthAgent(
                    knowledge_base=self.knowledge_base,
                    bayesian_network=self.bayesian_network
                )
                
                # Load initial data
                self.knowledge_base.initialize_base_knowledge()
                self.bayesian_network.initialize_network()
                
                st.session_state.assistant_initialized = True
                st.success("✅ AI Health Assistant Ready!")

    def create_metric_card(self, title, value, delta=None, icon="📊"):
        """Create a beautiful metric card"""
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{title}**")
            st.markdown(f"# {value}")
            if delta:
                st.markdown(f"**{delta}**")
        with col2:
            st.markdown(f'<div style="font-size: 2.5rem; text-align: center;">{icon}</div>', unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with user profile and navigation"""
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
        """Main symptom analysis interface"""
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="gradient-text">🔍 Symptom Checker</h1>
            <p style="font-size: 1.2rem; color: #7f8c8d;">Describe your symptoms and let AI analyze them</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <h3>💬 Describe Your Symptoms</h3>
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
            with st.expander("🔍 Additional Context (Optional)", expanded=False):
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
            if st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True):
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
                <h3>⚡ Quick Select</h3>
            """, unsafe_allow_html=True)
            
            # Common symptoms quick select
            common_symptoms = [
                "Headache", "Fever", "Cough", "Fatigue", "Nausea",
                "Dizziness", "Sore throat", "Runny nose", "Body aches"
            ]
            
            selected_quick = st.multiselect("Select common symptoms:", common_symptoms)
            
            if selected_quick and st.button("Use Selected Symptoms", use_container_width=True):
                quick_text = "I have " + ", ".join(selected_quick).lower()
                self.analyze_symptoms(quick_text, {})
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def analyze_symptoms(self, symptoms_text: str, context: Dict):
        """Analyze symptoms and display results"""
        with st.spinner("🤖 AI is analyzing your symptoms with advanced algorithms..."):
            result = self.health_agent.process_symptoms(symptoms_text, context)
            
            # Store in history
            st.session_state.symptoms_history.append({
                'timestamp': datetime.now(),
                'symptoms': symptoms_text,
                'result': result
            })
        
        # Display results
        if result.get('emergency'):
            st.markdown("""
            <div class="emergency-card">
                <h2 style="color: white; text-align: center;">🚨 EMERGENCY ALERT</h2>
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
            st.success("✅ Analysis Complete!")
            
            analysis = result['analysis']
            
            # Create results grid
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="icon-large">🎯</div>
                    <h3>Primary Assessment</h3>
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
                    <h3>Urgency Level</h3>
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
                    <h3>Other Possibilities</h3>
                """, unsafe_allow_html=True)
                
                possible_conditions = analysis.get('possible_conditions', [])
                
                for condition in possible_conditions[:2]:  # Show top 2
                    st.write(f"• {condition.replace('_', ' ').title()}")
                
                if len(possible_conditions) > 2:
                    st.write(f"... and {len(possible_conditions) - 2} more")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Recommendations section
            st.markdown("""
            <div class="custom-card">
                <h3>📋 Recommended Actions</h3>
            """, unsafe_allow_html=True)
            
            for i, recommendation in enumerate(result['recommendations'], 1):
                st.markdown(f"""
                <div style="background: #ecf0f1; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #3498db;">
                    <strong>Step {i}:</strong> {recommendation}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # AI Explanation
            with st.expander("🤔 How did our AI reach this conclusion?", expanded=True):
                st.markdown("""
                <div class="custom-card">
                """, unsafe_allow_html=True)
                
                st.info(result['explanation'])
                
                # Show symptoms matched
                symptoms = analysis.get('symptoms', [])
                if symptoms:
                    st.write("**🔍 Symptoms identified:**")
                    symptom_badges = " ".join([f'<span class="custom-badge">{symptom}</span>' for symptom in symptoms])
                    st.markdown(symptom_badges, unsafe_allow_html=True)
                
                # Show reasoning process
                st.markdown("""
                <h4>🧠 AI Reasoning Process:</h4>
                <ol>
                <li><strong>Natural Language Processing</strong> - Extracted symptoms from your description</li>
                <li><strong>Pattern Matching</strong> - Compared symptoms with known conditions</li>
                <li><strong>Probabilistic Analysis</strong> - Used Bayesian networks to calculate likelihoods</li>
                <li><strong>Utility Assessment</strong> - Evaluated urgency and appropriate actions</li>
                </ol>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    def render_health_dashboard(self):
        """Health metrics tracking dashboard"""
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="gradient-text">📊 Health Dashboard</h1>
            <p style="font-size: 1.2rem; color: #7f8c8d;">Track and monitor your health metrics</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📥 Add Metrics", "📈 View Trends", "📋 Health History"])
        
        with tab1:
            self.render_metrics_input()
        
        with tab2:
            self.render_health_trends()
        
        with tab3:
            self.render_symptoms_history()
    
    def render_metrics_input(self):
        """Input form for health metrics"""
        st.markdown("""
        <div class="custom-card">
            <h3>📥 Add New Health Metrics</h3>
        """, unsafe_allow_html=True)
        
        with st.form("health_metrics_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🏋️ Weight")
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
                
                st.subheader("💓 Blood Pressure")
                bp_systolic = st.number_input("Systolic", min_value=80, max_value=200, value=120)
            
            with col2:
                st.subheader("❤️ Heart Rate")
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
                
                bp_diastolic = st.number_input("Diastolic", min_value=50, max_value=150, value=80)
            
            with col3:
                st.subheader("🌬️ Other Metrics")
                blood_oxygen = st.number_input("Blood Oxygen (%)", min_value=80, max_value=100, value=98)
                stress_level = st.slider("Stress Level", 1, 10, 5)
            
            st.subheader("📝 Additional Notes")
            notes = st.text_area("", placeholder="Any additional observations...")
            
            if st.form_submit_button("💾 Save Metrics", use_container_width=True):
                timestamp = datetime.now()
                
                # Store metrics
                if weight > 0:
                    st.session_state.health_metrics['weight'].append({
                        'timestamp': timestamp, 'value': weight
                    })
                
                if bp_systolic > 0 and bp_diastolic > 0:
                    st.session_state.health_metrics['blood_pressure'].append({
                        'timestamp': timestamp, 
                        'systolic': bp_systolic, 
                        'diastolic': bp_diastolic
                    })
                
                if heart_rate > 0:
                    st.session_state.health_metrics['heart_rate'].append({
                        'timestamp': timestamp, 'value': heart_rate
                    })
                
                st.success("✅ Health metrics saved successfully!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_health_trends(self):
        """Display health metrics trends"""
        st.markdown("""
        <div class="custom-card">
            <h3>📈 Health Trends & Analytics</h3>
        """, unsafe_allow_html=True)
        
        # Weight trend
        if st.session_state.health_metrics['weight']:
            weight_df = pd.DataFrame(st.session_state.health_metrics['weight'])
            fig = px.line(weight_df, x='timestamp', y='value', 
                         title="📊 Weight Trend Over Time", markers=True)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No weight data available. Add some metrics to see trends!")
        
        # Blood pressure trend
        if st.session_state.health_metrics['blood_pressure']:
            bp_df = pd.DataFrame(st.session_state.health_metrics['blood_pressure'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bp_df['timestamp'], y=bp_df['systolic'], 
                                   name='Systolic', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=bp_df['timestamp'], y=bp_df['diastolic'], 
                                   name='Diastolic', line=dict(color='blue')))
            fig.update_layout(
                title="🩸 Blood Pressure Trend",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Heart rate trend
        if st.session_state.health_metrics['heart_rate']:
            hr_df = pd.DataFrame(st.session_state.health_metrics['heart_rate'])
            fig = px.line(hr_df, x='timestamp', y='value', 
                         title="💓 Heart Rate Trend", markers=True)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_symptoms_history(self):
        """Display symptoms history"""
        st.markdown("""
        <div class="custom-card">
            <h3>📋 Symptoms Analysis History</h3>
        """, unsafe_allow_html=True)
        
        if not st.session_state.symptoms_history:
            st.info("No symptoms analysis history yet. Use the Symptom Checker to get started!")
            return
        
        for i, entry in enumerate(reversed(st.session_state.symptoms_history[-10:])):  # Last 10 entries
            with st.expander(f"🕒 {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}: {entry['symptoms'][:50]}...", expanded=False):
                result = entry['result']
                
                if result.get('emergency'):
                    st.error("🚨 Emergency case - Immediate attention required")
                else:
                    analysis = result['analysis']
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**🎯 Condition:** {analysis.get('primary_condition', 'Unknown').replace('_', ' ').title()}")
                        st.write(f"**📊 Confidence:** {analysis.get('confidence', 0):.1%}")
                    
                    with col2:
                        urgency = analysis.get('urgency_level', 'low')
                        urgency_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[urgency]
                        st.write(f"**⚡ Urgency:** {urgency_icon} {urgency.upper()}")
                        st.write(f"**🔍 Symptoms:** {', '.join(analysis.get('symptoms', []))}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_doctor_finder(self):
        """Doctor recommendation system"""
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="gradient-text">👨‍⚕️ Doctor Finder</h1>
            <p style="font-size: 1.2rem; color: #7f8c8d;">Find the right healthcare professional for your needs</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <h3>🔍 Search Filters</h3>
            """, unsafe_allow_html=True)
            
            specialty = st.selectbox(
                "🎓 Medical Specialty",
                ["General Practice", "Cardiology", "Dermatology", "Neurology", 
                 "Pediatrics", "Orthopedics", "Emergency Medicine"]
            )
            
            location = st.text_input("📍 Location", placeholder="City or ZIP code")
            
            max_distance = st.slider("📏 Maximum Distance (km)", 5, 100, 25)
            
            insurance = st.selectbox(
                "🏥 Insurance",
                ["Any", "Medicare", "Medicaid", "Private Insurance", "Self-pay"]
            )
            
            availability = st.selectbox(
                "⏰ Availability",
                ["Any", "Next 24 hours", "Next 3 days", "Next week"]
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <h3>👨‍⚕️ Recommended Doctors</h3>
            """, unsafe_allow_html=True)
            
            # Mock doctor data
            doctors = [
                {
                    'name': 'Dr. Sarah Chen', 
                    'specialty': 'General Practice',
                    'distance': '2.3 km',
                    'rating': 4.8,
                    'availability': 'Tomorrow',
                    'insurance': ['Private Insurance', 'Medicare'],
                    'experience': '15 years',
                    'image': '👩‍⚕️'
                },
                {
                    'name': 'Dr. Michael Rodriguez',
                    'specialty': 'Cardiology', 
                    'distance': '5.1 km',
                    'rating': 4.9,
                    'availability': 'Today',
                    'insurance': ['Private Insurance'],
                    'experience': '20 years',
                    'image': '👨‍⚕️'
                },
                {
                    'name': 'Dr. Emily Watson',
                    'specialty': 'General Practice',
                    'distance': '1.2 km', 
                    'rating': 4.6,
                    'availability': 'Next 3 days',
                    'insurance': ['Medicare', 'Medicaid', 'Private Insurance'],
                    'experience': '10 years',
                    'image': '👩‍⚕️'
                }
            ]
            
            # Filter doctors based on selections
            filtered_doctors = [
                doc for doc in doctors 
                if (specialty == "General Practice" or doc['specialty'] == specialty)
            ]
            
            for i, doctor in enumerate(filtered_doctors):
                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; margin: 1rem 0; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 4px solid #3498db;">
                    <div style="display: flex; justify-content: between; align-items: center;">
                        <div style="flex: 1;">
                            <h3 style="margin: 0; color: #2c3e50;">{doctor['image']} {doctor['name']}</h3>
                            <p style="margin: 0.5rem 0; color: #7f8c8d;">
                                <strong>{doctor['specialty']}</strong> · {doctor['experience']} experience
                            </p>
                            <p style="margin: 0.5rem 0;">
                                📍 {doctor['distance']} away · ⭐ {doctor['rating']}/5.0
                            </p>
                            <div style="margin: 0.5rem 0;">
                                {" ".join([f'<span class="custom-badge success-badge">{ins}</span>' for ins in doctor['insurance']])}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <p style="margin: 0; color: #27ae60;"><strong>Available: {doctor['availability']}</strong></p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col2:
                    if st.button("📅 Book", key=f"book_{i}", use_container_width=True):
                        st.success(f"📅 Booking request sent to {doctor['name']}!")
                with col3:
                    if st.button("ℹ️ Profile", key=f"profile_{i}", use_container_width=True):
                        st.info(f"Showing profile for {doctor['name']}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def render_medication_tracker(self):
        """Medication reminder system"""
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="gradient-text">💊 Medication Tracker</h1>
            <p style="font-size: 1.2rem; color: #7f8c8d;">Manage your medications and reminders</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📝 Add Medication", "✅ Current Reminders"])
        
        with tab1:
            self.render_medication_input()
        
        with tab2:
            self.render_current_medications()
    
    def render_medication_input(self):
        """Input form for medications"""
        st.markdown("""
        <div class="custom-card">
            <h3>📝 Add New Medication</h3>
        """, unsafe_allow_html=True)
        
        with st.form("medication_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💊 Medication Details")
                med_name = st.text_input("Medication Name", placeholder="e.g., Aspirin")
                dosage = st.text_input("Dosage", placeholder="e.g., 500mg")
                frequency = st.selectbox(
                    "Frequency",
                    ["Once daily", "Twice daily", "Three times daily", 
                     "Every 6 hours", "As needed"]
                )
            
            with col2:
                st.subheader("📅 Schedule")
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date (optional)", value=None)
                notes = st.text_area("Special Instructions", placeholder="Any special instructions...")
            
            if st.form_submit_button("💾 Add Medication", use_container_width=True):
                if med_name and dosage:
                    reminder = {
                        'name': med_name,
                        'dosage': dosage,
                        'frequency': frequency,
                        'start_date': start_date,
                        'end_date': end_date,
                        'notes': notes,
                        'added_date': datetime.now()
                    }
                    
                    st.session_state.medication_reminders.append(reminder)
                    st.success(f"✅ Added {med_name} to your medication list!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_current_medications(self):
        """Display current medications"""
        st.markdown("""
        <div class="custom-card">
            <h3>✅ Current Medications</h3>
        """, unsafe_allow_html=True)
        
        if not st.session_state.medication_reminders:
            st.info("💊 No medications added yet. Add your first medication above!")
            return
        
        for i, med in enumerate(st.session_state.medication_reminders):
            with st.expander(f"💊 {med['name']} - {med['dosage']}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**📅 Frequency:** {med['frequency']}")
                    st.write(f"**🟢 Status:** Active")
                    st.write(f"**📝 Instructions:** {med['notes'] if med['notes'] else 'No special instructions'}")
                
                with col2:
                    st.write(f"**⏰ Start Date:** {med['start_date']}")
                    if med['end_date']:
                        st.write(f"**⏹️ End Date:** {med['end_date']}")
                    else:
                        st.write("**⏹️ End Date:** Ongoing")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ Taken Today", key=f"taken_{i}", use_container_width=True):
                        st.success(f"✅ Marked {med['name']} as taken today!")
                with col2:
                    if st.button("⏰ Remind Me", key=f"remind_{i}", use_container_width=True):
                        st.info(f"🔔 Reminder set for {med['name']}")
                with col3:
                    if st.button("🗑️ Remove", key=f"remove_{i}", use_container_width=True):
                        st.session_state.medication_reminders.pop(i)
                        st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_ai_insights(self):
        """AI and system insights"""
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="gradient-text">🤖 AI Insights</h1>
            <p style="font-size: 1.2rem; color: #7f8c8d;">Discover patterns and system performance</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="custom-card">
                <h3>📊 System Performance</h3>
            """, unsafe_allow_html=True)
            
            # Agent performance metrics
            metrics = self.health_agent.performance_metrics
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Queries Processed", metrics['queries_processed'], "AI Ready")
            with col2:
                st.metric("Emergencies Detected", metrics['emergencies_detected'], "Critical")
            with col3:
                st.metric("Successful Recommendations", metrics['successful_recommendations'], "Helpful")
            
            # AI model information
            with st.expander("🔧 AI Model Details", expanded=True):
                st.markdown("""
                **🤖 Current AI Models:**
                - Symptom Analysis: Bayesian Networks + Pattern Matching
                - Emergency Detection: Rule-based Logic Engine  
                - Recommendations: Utility Theory + Constraint Satisfaction
                - Natural Language Processing: TF-IDF + Keyword Extraction
                
                **🔒 Privacy Features:**
                - Federated Learning Ready
                - Local Data Processing
                - Anonymous Analytics
                - Secure Data Storage
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="custom-card">
                <h3>📈 Health Insights</h3>
            """, unsafe_allow_html=True)
            
            # Common conditions from history
            if st.session_state.symptoms_history:
                conditions = []
                for entry in st.session_state.symptoms_history:
                    if not entry['result'].get('emergency'):
                        condition = entry['result']['analysis'].get('primary_condition')
                        if condition:
                            conditions.append(condition)
                
                if conditions:
                    from collections import Counter
                    condition_counts = Counter(conditions)
                    
                    fig = px.pie(
                        values=list(condition_counts.values()),
                        names=[c.replace('_', ' ').title() for c in condition_counts.keys()],
                        title="🩺 Common Conditions Analysis"
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No condition data available yet. Use the Symptom Checker to generate insights!")
            else:
                st.info("No health data available yet. Start by checking your symptoms!")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def render_user_profile(self):
        """User profile management"""
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 class="gradient-text">👤 User Profile</h1>
            <p style="font-size: 1.2rem; color: #7f8c8d;">Manage your personal health information</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="custom-card">
            <h3>👤 Personal Information</h3>
        """, unsafe_allow_html=True)
        
        with st.form("user_profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📝 Basic Info")
                age = st.number_input("Age", min_value=1, max_value=120, 
                                    value=st.session_state.user_profile['age'])
                gender = st.selectbox(
                    "Gender",
                    ["Prefer not to say", "Male", "Female", "Other"],
                    index=["Prefer not to say", "Male", "Female", "Other"].index(
                        st.session_state.user_profile['gender']
                    )
                )
            
            with col2:
                st.subheader("📍 Contact & Insurance")
                location = st.text_input(
                    "Location", 
                    value=st.session_state.user_profile['location'],
                    placeholder="City, State"
                )
                insurance = st.text_input(
                    "Primary Insurance",
                    value=st.session_state.user_profile['insurance'],
                    placeholder="Insurance provider"
                )
            
            # Health conditions
            st.subheader("🩺 Health Information (Optional)")
            existing_conditions = st.multiselect(
                "Existing Health Conditions",
                ["Hypertension", "Diabetes", "Asthma", "Heart Disease", 
                 "Arthritis", "None", "Other"]
            )
            
            allergies = st.text_area("Allergies", placeholder="List any allergies...")
            current_medications = st.text_area("Current Medications", 
                                             placeholder="List current medications...")
            
            if st.form_submit_button("💾 Save Profile", use_container_width=True):
                st.session_state.user_profile.update({
                    'age': age,
                    'gender': gender,
                    'location': location,
                    'insurance': insurance,
                    'existing_conditions': existing_conditions,
                    'allergies': allergies,
                    'current_medications': current_medications
                })
                st.success("✅ Profile updated successfully!")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data management
        st.markdown("""
        <div class="custom-card">
            <h3>🔒 Data Management</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Health Data", use_container_width=True):
                st.info("📊 Export functionality would generate a comprehensive PDF health report here.")
        
        with col2:
            if st.button("🗑️ Clear All Data", use_container_width=True):
                st.warning("⚠️ This will permanently delete all your data!")
                if st.checkbox("I understand this action cannot be undone"):
                    st.session_state.symptoms_history = []
                    st.session_state.health_metrics = {'weight': [], 'blood_pressure': [], 'heart_rate': []}
                    st.session_state.medication_reminders = []
                    st.success("✅ All data cleared successfully!")
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def run(self):
        """Main application runner"""
        page = self.render_sidebar()
        
        # Page routing
        if page == "Symptom":
            self.render_symptom_checker()
        elif page == "Health":
            self.render_health_dashboard()
        elif page == "Doctor":
            self.render_doctor_finder()
        elif page == "Medication":
            self.render_medication_tracker()
        elif page == "AI":
            self.render_ai_insights()
        elif page == "User":
            self.render_user_profile()

# Run the application
if __name__ == "__main__":
    assistant = StreamlitHealthAssistant()
    assistant.run()
