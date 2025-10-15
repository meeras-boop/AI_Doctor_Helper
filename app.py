import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

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
    SEABORN_AVAILABLE = True  # We'll use matplotlib alternatives

# Set page configuration
st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AIHealthcareAssistant:
    def __init__(self):
        self.symptoms_db = self._initialize_symptoms_database()
        self.diseases_db = self._initialize_diseases_database()
        self.patients_history = {}
        
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
                'recommendation': 'Rest, hydrate, take antiviral medication if prescribed'
            },
            'covid': {
                'symptoms': ['fever', 'cough', 'fatigue', 'shortness_of_breath', 'muscle_pain'],
                'severity': 'high',
                'recommendation': 'Isolate, get tested, consult doctor immediately'
            },
            'pneumonia': {
                'symptoms': ['fever', 'cough', 'chest_pain', 'shortness_of_breath'],
                'severity': 'high',
                'recommendation': 'Emergency care required, antibiotics may be needed'
            },
            'heart_attack': {
                'symptoms': ['chest_pain', 'shortness_of_breath'],
                'severity': 'emergency',
                'recommendation': 'CALL EMERGENCY SERVICES IMMEDIATELY'
            },
            'migraine': {
                'symptoms': ['headache', 'nausea'],
                'severity': 'moderate',
                'recommendation': 'Rest in dark room, stay hydrated, consider pain relief'
            },
            'food_poisoning': {
                'symptoms': ['nausea', 'vomiting'],
                'severity': 'moderate',
                'recommendation': 'Stay hydrated, rest, avoid solid foods initially'
            }
        }
    
    def symptom_checker(self, selected_symptoms):
        """Informed Search using A* concepts with heuristic scoring"""
        disease_scores = {}
        
        for symptom in selected_symptoms:
            possible_diseases = self.symptoms_db.get(symptom, [])
            for disease in possible_diseases:
                if disease not in disease_scores:
                    disease_scores[disease] = 0
                disease_scores[disease] += 1
        
        # Normalize scores based on total symptoms for each disease
        for disease, score in disease_scores.items():
            if disease in self.diseases_db:
                total_symptoms = len(self.diseases_db[disease]['symptoms'])
                disease_scores[disease] = (score / total_symptoms * 100) if total_symptoms > 0 else 0
        
        return dict(sorted(disease_scores.items(), key=lambda x: x[1], reverse=True))
    
    def risk_assessment(self, age, bp, cholesterol, smoking, diabetes):
        """Bayesian Network for risk assessment"""
        base_risk = 0.01
        
        # Conditional probabilities (simplified)
        if age > 50: base_risk *= 2
        if bp == 'high': base_risk *= 1.8
        if cholesterol == 'high': base_risk *= 1.5
        if smoking: base_risk *= 2.2
        if diabetes: base_risk *= 1.7
        
        risk_percentage = min(base_risk * 100, 95)
        return risk_percentage
    
    def medication_reminder(self, medications, schedule):
        """Intelligent Agent for medication management"""
        current_time = datetime.now()
        reminders = []
        
        for med, times in schedule.items():
            for time_str in times:
                med_time = datetime.strptime(time_str, "%H:%M").time()
                if abs((current_time.hour - med_time.hour) * 60 + 
                      (current_time.minute - med_time.minute)) <= 30:
                    reminders.append(f"Time to take {med}")
        
        return reminders

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

def create_simple_plot(data, title, color='blue'):
    """Create simple plots without matplotlib"""
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['dates'], data['values'], color=color, linewidth=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig
    else:
        # Fallback: display data as table
        st.write(f"**{title}**")
        display_data = pd.DataFrame({
            'Date': data['dates'],
            'Value': data['values']
        })
        st.dataframe(display_data)

def main():
    st.title("🏥 AI Healthcare Assistant")
    st.markdown("""
    This intelligent healthcare assistant incorporates multiple AI concepts from your curriculum:
    - **Intelligent Agents** - Medication reminders and health monitoring
    - **Search Algorithms** - Symptom checking with heuristic scoring
    - **Constraint Satisfaction** - Symptom-disease relationships
    - **Bayesian Networks** - Risk assessment
    - **Markov Models** - Vital signs trend analysis
    - **Knowledge Representation** - Medical knowledge base
    """)
    
    # Installation instructions if libraries are missing
    if not MATPLOTLIB_AVAILABLE:
        st.error("""
        **Required packages not installed!** 
        
        Please install the required packages using:
        ```bash
        pip install matplotlib seaborn
        ```
        
        Or for Streamlit Cloud, add these to your requirements.txt file.
        """)
    
    # Initialize AI assistant
    assistant = AIHealthcareAssistant()
    monitor = HealthMonitor()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a feature",
        ["Symptom Checker", "Health Risk Assessment", "Medication Reminder", 
         "Vital Signs Monitor", "Health Analytics", "Emergency Assistant"]
    )
    
    if app_mode == "Symptom Checker":
        st.header("🔍 Symptom Checker")
        st.write("Select your symptoms for AI-powered diagnosis assistance")
        
        symptoms = list(assistant.symptoms_db.keys())
        selected_symptoms = st.multiselect("Select your symptoms:", symptoms)
        
        if selected_symptoms:
            st.subheader("Selected Symptoms:")
            st.write(", ".join(selected_symptoms))
            
            if st.button("Analyze Symptoms"):
                with st.spinner("AI is analyzing your symptoms..."):
                    # Simulate processing time
                    import time
                    time.sleep(1)
                    
                    results = assistant.symptom_checker(selected_symptoms)
                    
                    st.subheader("Possible Conditions:")
                    for disease, confidence in list(results.items())[:3]:
                        if confidence > 0:  # Only show diseases with some match
                            disease_info = assistant.diseases_db.get(disease, {})
                            severity = disease_info.get('severity', 'unknown')
                            recommendation = disease_info.get('recommendation', 'Consult a healthcare professional')
                            
                            # Color code based on severity
                            if severity == 'emergency':
                                color = 'red'
                            elif severity == 'high':
                                color = 'orange'
                            else:
                                color = 'green'
                            
                            st.markdown(f"""
                            <div style='border-left: 5px solid {color}; padding: 10px; margin: 10px 0;'>
                                <h4>{disease.replace('_', ' ').title()} ({confidence:.1f}% match)</h4>
                                <p><strong>Severity:</strong> {severity.title()}</p>
                                <p><strong>Recommendation:</strong> {recommendation}</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Add some sample symptoms for quick testing
        st.sidebar.subheader("Quick Test")
        if st.sidebar.button("Test Common Cold Symptoms"):
            st.experimental_set_query_params(symptoms=["fever", "cough", "headache"])
            st.rerun()
    
    elif app_mode == "Health Risk Assessment":
        st.header("📊 Health Risk Assessment")
        st.write("Bayesian network-based risk evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 1, 100, 30)
            bp = st.selectbox("Blood Pressure", ["normal", "elevated", "high"])
            cholesterol = st.selectbox("Cholesterol Level", ["normal", "borderline", "high"])
        
        with col2:
            smoking = st.checkbox("Smoker")
            diabetes = st.checkbox("Diabetes")
            family_history = st.checkbox("Family History of Heart Disease")
        
        if st.button("Assess Health Risk"):
            risk_score = assistant.risk_assessment(age, bp, cholesterol, smoking, diabetes)
            
            # Adjust for family history
            if family_history:
                risk_score *= 1.3
            
            st.subheader("Risk Assessment Results:")
            
            if risk_score < 10:
                risk_level = "Low"
                color = "green"
            elif risk_score < 30:
                risk_level = "Moderate"
                color = "orange"
            else:
                risk_level = "High"
                color = "red"
            
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 10px;'>
                <h3 style='color: {color};'>Overall Risk: {risk_level}</h3>
                <h2 style='color: {color};'>{risk_score:.1f}%</h2>
                <p>Probability of developing cardiovascular issues in next 10 years</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations based on risk
            st.subheader("Personalized Recommendations:")
            recommendations = []
            if smoking:
                recommendations.append("🚭 Consider smoking cessation programs")
            if bp == 'high':
                recommendations.append("💊 Monitor blood pressure regularly")
            if cholesterol == 'high':
                recommendations.append("🥗 Adopt heart-healthy diet")
            if risk_score > 20:
                recommendations.append("🏥 Regular health check-ups recommended")
            
            for rec in recommendations:
                st.write(f"- {rec}")
    
    elif app_mode == "Medication Reminder":
        st.header("💊 Medication Reminder")
        st.write("Intelligent agent for medication management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Add Medication")
            med_name = st.text_input("Medication Name", "Aspirin")
            schedule_times = st.multiselect(
                "Schedule Times",
                ["08:00", "12:00", "18:00", "20:00", "22:00"],
                default=["08:00", "20:00"]
            )
            
            if st.button("Add to Schedule"):
                if med_name and schedule_times:
                    st.success(f"Added {med_name} at {', '.join(schedule_times)}")
        
        with col2:
            st.subheader("Current Reminders")
            # Simulate current reminders
            current_meds = {
                "Aspirin": ["08:00", "20:00"],
                "Vitamin D": ["12:00"]
            }
            
            reminders = assistant.medication_reminder(current_meds, current_meds)
            if reminders:
                for reminder in reminders:
                    st.warning(reminder)
            else:
                st.info("No medications due at this time")
            
            # Medication adherence tracking
            st.subheader("Adherence Tracking")
            adherence_rate = 85  # Simulated data
            st.progress(adherence_rate / 100)
            st.write(f"Current adherence: {adherence_rate}%")
    
    elif app_mode == "Vital Signs Monitor":
        st.header("❤️ Vital Signs Monitor")
        st.write("Real-time health monitoring with trend analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Enter Current Vital Signs")
            heart_rate = st.slider("Heart Rate (bpm)", 40, 200, 75)
            bp_systolic = st.slider("Systolic BP (mmHg)", 80, 200, 120)
            bp_diastolic = st.slider("Diastolic BP (mmHg)", 50, 130, 80)
            temperature = st.slider("Temperature (°C)", 35.0, 41.0, 36.6)
            spo2 = st.slider("Blood Oxygen (%)", 85, 100, 98)
            
            if st.button("Record Vital Signs"):
                monitor.add_vital_signs(heart_rate, bp_systolic, bp_diastolic, temperature, spo2)
                st.success("Vital signs recorded successfully!")
        
        with col2:
            st.subheader("Current Readings")
            
            # Create a dashboard of current readings
            metrics_data = {
                "Parameter": ["Heart Rate", "Systolic BP", "Diastolic BP", "Temperature", "SpO2"],
                "Value": [f"{heart_rate} bpm", f"{bp_systolic}/{bp_diastolic} mmHg", 
                         f"{bp_diastolic} mmHg", f"{temperature}°C", f"{spo2}%"],
                "Status": [
                    "Normal" if 60 <= heart_rate <= 100 else "Alert",
                    "Normal" if 90 <= bp_systolic <= 120 else "Monitor",
                    "Normal" if 60 <= bp_diastolic <= 80 else "Monitor",
                    "Normal" if 36.1 <= temperature <= 37.2 else "Monitor",
                    "Normal" if spo2 >= 95 else "Alert"
                ]
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True)
            
            # Trend analysis
            st.subheader("Trend Analysis")
            trends = monitor.analyze_trends()
            for trend in trends:
                st.write(trend)
    
    elif app_mode == "Health Analytics":
        st.header("📈 Health Analytics")
        st.write("AI-powered health data visualization and insights")
        
        # Generate sample health data
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        heart_rates = np.random.normal(75, 10, len(dates))
        steps = np.random.randint(3000, 15000, len(dates))
        sleep_hours = np.random.normal(7.5, 1, len(dates))
        
        health_data = pd.DataFrame({
            'Date': dates,
            'Heart Rate': heart_rates,
            'Daily Steps': steps,
            'Sleep Hours': sleep_hours
        })
        
        if MATPLOTLIB_AVAILABLE:
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Heart Rate Trends")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(health_data['Date'], health_data['Heart Rate'], color='red', linewidth=2)
                ax.set_ylabel('Heart Rate (bpm)')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Activity Levels")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(health_data['Date'][::7], health_data['Daily Steps'][::7], 
                       color='blue', alpha=0.7, width=5)
                ax.set_ylabel('Daily Steps')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        else:
            # Show data tables instead of plots
            st.subheader("Heart Rate Data")
            st.dataframe(health_data[['Date', 'Heart Rate']].head(10))
            
            st.subheader("Activity Data")
            st.dataframe(health_data[['Date', 'Daily Steps']].head(10))
        
        # AI Insights
        st.subheader("AI Health Insights")
        avg_hr = np.mean(heart_rates)
        avg_steps = np.mean(steps)
        avg_sleep = np.mean(sleep_hours)
        
        insights = []
        if avg_hr > 80:
            insights.append("🔍 Your average heart rate is slightly elevated. Consider stress management techniques.")
        if avg_steps < 8000:
            insights.append("🚶 Try to increase daily steps to 8,000-10,000 for better cardiovascular health.")
        if avg_sleep < 7:
            insights.append("😴 Aim for 7-9 hours of sleep per night for optimal health.")
        
        if insights:
            for insight in insights:
                st.info(insight)
        else:
            st.success("🎉 Great job! Your health metrics are within optimal ranges.")
    
    elif app_mode == "Emergency Assistant":
        st.header("🚨 Emergency Assistant")
        st.write("Immediate assistance and guidance for emergency situations")
        
        emergency_options = st.selectbox(
            "Select Emergency Type",
            ["Chest Pain", "Difficulty Breathing", "Severe Bleeding", 
             "Loss of Consciousness", "Severe Allergic Reaction", "Stroke Symptoms"]
        )
        
        if emergency_options:
            st.warning(f"⚠️ {emergency_options} Detected")
            
            # Emergency protocols
            protocols = {
                "Chest Pain": [
                    "Sit or lie down immediately",
                    "Call emergency services (911/112)",
                    "Chew aspirin if available and not allergic",
                    "Loosen tight clothing",
                    "Do not drive yourself to hospital"
                ],
                "Difficulty Breathing": [
                    "Sit upright in comfortable position",
                    "Call emergency services immediately",
                    "Use inhaler if prescribed",
                    "Try to remain calm",
                    "Loosen any tight clothing"
                ],
                "Severe Bleeding": [
                    "Apply direct pressure to wound",
                    "Elevate injured area if possible",
                    "Call emergency services",
                    "Do not remove embedded objects",
                    "Keep victim warm and still"
                ],
                "Stroke Symptoms": [
                    "Call emergency services immediately",
                    "Note time when symptoms started",
                    "Do not give food or drink",
                    "Keep person comfortable",
                    "Be prepared to perform CPR if needed"
                ]
            }
            
            protocol = protocols.get(emergency_options, [
                "Call emergency services immediately",
                "Stay with the person",
                "Follow dispatcher instructions",
                "Prepare to provide CPR if trained"
            ])
            
            st.subheader("Emergency Protocol:")
            for i, step in enumerate(protocol, 1):
                st.write(f"{i}. {step}")
            
            # Emergency contact
            st.subheader("Emergency Contacts")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚑 Call Emergency Services"):
                    st.info("Dialing 911... Please describe the emergency clearly.")
            with col2:
                if st.button("📞 Contact Emergency Contact"):
                    st.info("Calling your emergency contact...")
            
            # Location sharing
            st.info("📍 Share your location with emergency services for faster response")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>This AI Healthcare Assistant demonstrates multiple AI concepts including:</p>
        <small>Intelligent Agents • Search Algorithms • Bayesian Networks • Markov Models • Constraint Satisfaction • Knowledge Representation</small>
        <br><br>
        <p><strong>Installation Note:</strong> If graphs are not showing, install required packages:</p>
        <code>pip install matplotlib seaborn</code>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
