# heuristic_comparison.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import heapq
import math

class AStarComparison:
    def __init__(self):
        self.symptoms_db, self.diseases_db = self._initialize_databases()
        
    def _initialize_databases(self):
        symptoms_db = {
            'fever': ['flu', 'covid', 'pneumonia', 'malaria', 'dengue'],
            'cough': ['flu', 'covid', 'pneumonia', 'bronchitis', 'tuberculosis'],
            'headache': ['flu', 'migraine', 'covid', 'hypertension', 'meningitis'],
            'fatigue': ['flu', 'covid', 'anemia', 'depression', 'mononucleosis'],
            'chest_pain': ['heart_attack', 'pneumonia', 'angina', 'pulmonary_embolism'],
            'shortness_of_breath': ['covid', 'pneumonia', 'asthma', 'heart_attack', 'copd'],
            'nausea': ['food_poisoning', 'migraine', 'pregnancy', 'appendicitis'],
            'vomiting': ['food_poisoning', 'migraine', 'gastroenteritis', 'appendicitis'],
            'muscle_pain': ['flu', 'covid', 'fibromyalgia', 'lyme_disease'],
            'sore_throat': ['flu', 'covid', 'strep_throat', 'mononucleosis']
        }
        
        diseases_db = {
            'flu': {
                'symptoms': ['fever', 'cough', 'headache', 'fatigue', 'muscle_pain', 'sore_throat'],
                'severity': 'moderate',
                'prevalence': 0.15  # Higher prevalence
            },
            'covid': {
                'symptoms': ['fever', 'cough', 'fatigue', 'shortness_of_breath', 'muscle_pain'],
                'severity': 'high',
                'prevalence': 0.08
            },
            'pneumonia': {
                'symptoms': ['fever', 'cough', 'chest_pain', 'shortness_of_breath'],
                'severity': 'high',
                'prevalence': 0.05
            },
            'migraine': {
                'symptoms': ['headache', 'nausea'],
                'severity': 'moderate',
                'prevalence': 0.12
            },
            'heart_attack': {
                'symptoms': ['chest_pain', 'shortness_of_breath'],
                'severity': 'emergency',
                'prevalence': 0.02
            },
            'food_poisoning': {
                'symptoms': ['nausea', 'vomiting'],
                'severity': 'moderate',
                'prevalence': 0.06
            }
        }
        
        return symptoms_db, diseases_db
    
    # Heuristic implementations (same as in main code)
    def heuristic_manhattan(self, current_symptoms, target_disease):
        if target_disease not in self.diseases_db:
            return float('inf')
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        unmatched_symptoms = target_symptoms - current_symptoms
        return len(unmatched_symptoms)
    
    def heuristic_euclidean(self, current_symptoms, target_disease):
        if target_disease not in self.diseases_db:
            return float('inf')
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        matched = len(current_symptoms.intersection(target_symptoms))
        total_target = len(target_symptoms)
        return math.sqrt((total_target - matched) ** 2)
    
    def heuristic_symptom_frequency(self, current_symptoms, target_disease):
        if target_disease not in self.diseases_db:
            return float('inf')
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        common_symptoms = current_symptoms.intersection(target_symptoms)
        frequency_score = 0
        for symptom in common_symptoms:
            disease_count = len(self.symptoms_db.get(symptom, []))
            frequency_score += 1 / (disease_count + 1)
        return -frequency_score if frequency_score > 0 else float('inf')
    
    def heuristic_severity_weighted(self, current_symptoms, target_disease):
        if target_disease not in self.diseases_db:
            return float('inf')
        target_symptoms = set(self.diseases_db[target_disease]['symptoms'])
        unmatched_symptoms = target_symptoms - current_symptoms
        severity_weights = {'emergency': 3.0, 'high': 2.0, 'moderate': 1.5, 'low': 1.0}
        severity = self.diseases_db[target_disease].get('severity', 'moderate')
        weight = severity_weights.get(severity, 1.0)
        return len(unmatched_symptoms) / weight
    
    def run_comparison(self, test_cases):
        results = []
        heuristics = [
            ('Manhattan', self.heuristic_manhattan),
            ('Euclidean', self.heuristic_euclidean),
            ('Symptom Frequency', self.heuristic_symptom_frequency),
            ('Severity Weighted', self.heuristic_severity_weighted)
        ]
        
        for case_name, symptoms in test_cases:
            for heuristic_name, heuristic_func in heuristics:
                # Run A* with this heuristic
                performance = self.evaluate_heuristic(symptoms, heuristic_func)
                results.append({
                    'Test Case': case_name,
                    'Heuristic': heuristic_name,
                    'Top Disease': performance['top_disease'],
                    'Confidence': performance['confidence'],
                    'Computation Time': performance['computation_time'],
                    'Accuracy': performance['accuracy'],
                    'Symptoms Count': len(symptoms)
                })
        
        return pd.DataFrame(results)
    
    def evaluate_heuristic(self, symptoms, heuristic_func):
        start_time = datetime.now()
        
        # Simplified A* implementation for evaluation
        current_symptoms_set = set(symptoms)
        possible_diseases = set()
        
        for symptom in symptoms:
            possible_diseases.update(self.symptoms_db.get(symptom, []))
        
        best_disease = None
        best_confidence = 0
        
        for disease in possible_diseases:
            target_symptoms = set(self.diseases_db.get(disease, {}).get('symptoms', []))
            g_score = len(current_symptoms_set - target_symptoms)
            h_score = heuristic_func(current_symptoms_set, disease)
            f_score = g_score + h_score
            confidence = max(0, 100 - f_score * 15)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_disease = disease
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
        
        # Simple accuracy estimation (in real scenario, you'd have ground truth)
        accuracy = self.estimate_accuracy(best_disease, symptoms)
        
        return {
            'top_disease': best_disease,
            'confidence': best_confidence,
            'computation_time': computation_time,
            'accuracy': accuracy
        }
    
    def estimate_accuracy(self, predicted_disease, symptoms):
        """Simple accuracy estimation based on symptom matching"""
        if predicted_disease is None:
            return 0
        
        actual_symptoms = set(self.diseases_db.get(predicted_disease, {}).get('symptoms', []))
        input_symptoms = set(symptoms)
        
        precision = len(actual_symptoms.intersection(input_symptoms)) / len(input_symptoms) if input_symptoms else 0
        recall = len(actual_symptoms.intersection(input_symptoms)) / len(actual_symptoms) if actual_symptoms else 0
        
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

# Create test cases
test_cases = [
    ('Common Cold', ['fever', 'cough', 'headache']),
    ('Respiratory Issue', ['cough', 'shortness_of_breath', 'chest_pain']),
    ('Gastrointestinal', ['nausea', 'vomiting']),
    ('Severe Case', ['chest_pain', 'shortness_of_breath', 'fever']),
    ('Mixed Symptoms', ['headache', 'fatigue', 'muscle_pain'])
]

# Run comparison
comparator = AStarComparison()
results_df = comparator.run_comparison(test_cases)

# Display results
print("Heuristic Comparison Results:")
print("=" * 50)
print(results_df.to_string(index=False))

# Visualization
plt.figure(figsize=(15, 10))

# Plot 1: Confidence by Heuristic
plt.subplot(2, 2, 1)
sns.boxplot(data=results_df, x='Heuristic', y='Confidence')
plt.title('Confidence Distribution by Heuristic')
plt.xticks(rotation=45)

# Plot 2: Computation Time
plt.subplot(2, 2, 2)
sns.boxplot(data=results_df, x='Heuristic', y='Computation Time')
plt.title('Computation Time by Heuristic')
plt.xticks(rotation=45)

# Plot 3: Accuracy
plt.subplot(2, 2, 3)
sns.boxplot(data=results_df, x='Heuristic', y='Accuracy')
plt.title('Accuracy by Heuristic')
plt.xticks(rotation=45)

# Plot 4: Performance by Symptom Count
plt.subplot(2, 2, 4)
sns.scatterplot(data=results_df, x='Symptoms Count', y='Computation Time', hue='Heuristic')
plt.title('Computation Time vs Symptoms Count')

plt.tight_layout()
plt.show()

# Summary Statistics
print("\nSummary Statistics:")
print("=" * 50)
summary = results_df.groupby('Heuristic').agg({
    'Confidence': ['mean', 'std'],
    'Computation Time': ['mean', 'std'],
    'Accuracy': ['mean', 'std']
}).round(3)
print(summary)
