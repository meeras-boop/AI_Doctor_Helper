import re
from typing import Dict, List, Any
from datetime import datetime

class PersonalHealthAgent:
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
            r'severe headache', r'paralysis', r'seizure'
        ]
    
    def check_emergency(self, symptoms_text: str) -> bool:
        text_lower = symptoms_text.lower()
        for pattern in self.emergency_patterns:
            if re.search(pattern, text_lower):
                self.performance_metrics['emergencies_detected'] += 1
                return True
        return False
    
    def process_symptoms(self, symptoms_text: str, user_context: Dict = None) -> Dict[str, Any]:
        self.performance_metrics['queries_processed'] += 1
        
        # Emergency check
        if self.check_emergency(symptoms_text):
            return {'emergency': True}
        
        # Extract symptoms
        symptoms = self._extract_symptoms(symptoms_text)
        
        # Get analysis
        analysis = self.analyze_symptoms(symptoms)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(analysis)
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'explanation': self.explain_reasoning(analysis)
        }
    
    def _extract_symptoms(self, text: str) -> List[str]:
        words = text.lower().split()
        symptoms = []
        known_symptoms = self.knowledge_base.get_known_symptoms()
        
        for symptom in known_symptoms:
            if symptom in text.lower():
                symptoms.append(symptom)
        
        return symptoms if symptoms else ['general discomfort']
    
    def analyze_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        # Search-based analysis
        search_results = self._search_based_analysis(symptoms)
        
        # Probabilistic reasoning
        prob_results = self.bayesian_network.infer_conditions(symptoms)
        
        # Combine results
        return self._combine_analyses(search_results, prob_results)
    
    def _search_based_analysis(self, symptoms: List[str]) -> Dict[str, Any]:
        possible_conditions = self.knowledge_base.get_conditions_for_symptoms(symptoms)
        
        if not possible_conditions:
            return {'primary_condition': None, 'confidence': 0, 'possible_conditions': []}
        
        # Find best match
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
    
    def _combine_analyses(self, search_results: Dict, prob_results: Dict) -> Dict[str, Any]:
        primary_condition = search_results.get('primary_condition') or prob_results.get('most_likely')
        
        return {
            'primary_condition': primary_condition,
            'possible_conditions': list(set(
                search_results.get('possible_conditions', []) + 
                prob_results.get('possible_conditions', [])
            )),
            'confidence': max(
                search_results.get('confidence', 0), 
                prob_results.get('confidence', 0)
            ),
            'urgency_level': self._calculate_urgency(search_results, prob_results),
            'symptoms': search_results.get('symptoms', [])
        }
    
    def _calculate_urgency(self, search_results: Dict, prob_results: Dict) -> str:
        if search_results.get('confidence', 0) > 0.8 or prob_results.get('risk_level') == 'high':
            return 'high'
        elif search_results.get('confidence', 0) > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def generate_recommendations(self, analysis_result: Dict) -> List[str]:
        recommendations = []
        condition = analysis_result.get('primary_condition')
        urgency = analysis_result.get('urgency_level')
        
        if urgency == 'high':
            recommendations.append("Consult a healthcare professional within 24 hours")
        elif urgency == 'medium':
            recommendations.append("Schedule a doctor's appointment this week")
        else:
            recommendations.append("Monitor symptoms and rest")
        
        if condition:
            specific_recs = self.knowledge_base.get_condition_recommendations(condition)
            recommendations.extend(specific_recs)
        
        self.performance_metrics['successful_recommendations'] += 1
        return recommendations
    
    def explain_reasoning(self, analysis_result: Dict) -> str:
        symptoms = analysis_result.get('symptoms', [])
        condition = analysis_result.get('primary_condition', 'unknown condition')
        confidence = analysis_result.get('confidence', 0)
        
        explanation = f"Based on your symptoms ({', '.join(symptoms)}), "
        explanation += f"the system identified '{condition.replace('_', ' ').title()}' with {confidence:.1%} confidence. "
        explanation += "This assessment combines pattern matching with probabilistic reasoning using Bayesian networks."
        
        return explanation
