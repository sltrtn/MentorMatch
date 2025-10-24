# ========== FLASK WEB APP FOR MENTOR MATCHING ==========
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ==================== LOAD RESOURCES ====================

print("Loading mentor matching system...")

# Load model and data
model_package = joblib.load('trained_matching_model.pkl')
mentors_df = pd.read_csv('mentors_dataset.csv')

trained_model = model_package['model']
learned_weights = model_package['weights']
feature_cols = model_package['feature_cols']

print(f"✓ Loaded model with R² score: {model_package['metrics']['r2']:.3f}")
print(f"✓ Loaded {len(mentors_df)} mentors")

# ==================== SIMILARITY CALCULATION ====================

def calculate_similarity(student, mentor):
    """Calculate all similarity factors"""
    
    similarities = {}
    
    # University (tier-based)
    tier1 = ['IIT Delhi', 'IIT Bombay', 'IIT Madras', 'BITS Pilani', 'IIIT Hyderabad']
    tier2 = ['NIT Trichy', 'NIT Surathkal', 'DTU Delhi', 'PESIT Bangalore']
    
    s_uni = student.get('university', '')
    m_uni = mentor.get('university', '')
    
    if s_uni == m_uni:
        similarities['university_sim'] = 1.0
    elif (s_uni in tier1 and m_uni in tier1) or (s_uni in tier2 and m_uni in tier2):
        similarities['university_sim'] = 0.7
    else:
        similarities['university_sim'] = 0.3
    
    # Academic stream
    similarities['academic_stream_sim'] = 1.0 if student.get('academic_stream') == mentor.get('academic_stream') else 0.0
    
    # CGPA similarity
    cgpa_diff = abs(student.get('cgpa', 7.5) - mentor.get('cgpa', 7.5))
    similarities['cgpa_sim'] = np.exp(-(cgpa_diff ** 2) / 2)
    
    # Language
    similarities['language_sim'] = 1.0 if student.get('language') == mentor.get('language') else 0.0
    
    # Company
    similarities['company_sim'] = 1.0 if student.get('current_company') == mentor.get('company') else 0.0
    
    # Job role
    similarities['job_role_sim'] = 1.0 if student.get('aspiring_role') == mentor.get('job_role') else 0.0
    
    # Region
    similarities['region_sim'] = 1.0 if student.get('region') == mentor.get('region') else 0.0
    
    # Career interest (most important)
    similarities['career_interest_sim'] = 1.0 if student.get('career_interest') == mentor.get('career_interest') else 0.0
    
    # Mood preference
    similarities['mood_preference_sim'] = 1.0 if student.get('mood_preference') == mentor.get('mood_preference') else 0.0
    
    # Experience gap
    exp_gap = mentor.get('experience_years', 10) - student.get('experience_years', 0)
    if 5 <= exp_gap <= 15:
        similarities['experience_gap_sim'] = 1.0
    elif 3 <= exp_gap < 5 or 15 < exp_gap <= 20:
        similarities['experience_gap_sim'] = 0.7
    else:
        similarities['experience_gap_sim'] = 0.3
    
    # Age difference
    age_diff = abs(mentor.get('age', 30) - student.get('age', 21))
    if 5 <= age_diff <= 20:
        similarities['age_diff_sim'] = 1.0 - ((age_diff - 5) / 15) * 0.3
    else:
        similarities['age_diff_sim'] = 0.4
    
    # Mentoring record
    similarities['mentoring_record_sim'] = mentor.get('mentoring_rating', 4.0) / 5.0
    
    # Availability
    avail_scores = {'Very High': 1.0, 'High': 0.8, 'Medium': 0.5, 'Low': 0.2}
    similarities['availability_sim'] = avail_scores.get(mentor.get('availability', 'Medium'), 0.5)
    
    return similarities

# ==================== ROUTES ====================

@app.route('/')
def home():
    """Home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Get mentor recommendations"""
    
    try:
        # Get form data
        student_data = {
            'name': request.form.get('name'),
            'university': request.form.get('university'),
            'academic_stream': request.form.get('stream'),
            'cgpa': float(request.form.get('cgpa')),
            'language': request.form.get('language'),
            'region': request.form.get('region'),
            'career_interest': request.form.get('career_interest'),
            'aspiring_role': request.form.get('role'),
            'current_company': request.form.get('company'),
            'mood_preference': request.form.get('mood'),
            'age': int(request.form.get('age')),
            'experience_years': int(request.form.get('experience'))
        }
        
        # Find matches
        matches = []
        
        for _, mentor in mentors_df.iterrows():
            similarities = calculate_similarity(student_data, mentor)
            
            # Create feature vector
            feature_vector = [similarities.get(col, 0.5) for col in feature_cols]
            
            # Predict score
            score = trained_model.predict([feature_vector])[0] * 100
            
            matches.append({
                'name': mentor['name'],
                'company': mentor['company'],
                'role': mentor['job_role'],
                'experience': int(mentor['experience_years']),
                'university': mentor['university'],
                'stream': mentor['academic_stream'],
                'rating': float(mentor['mentoring_rating']),
                'availability': mentor['availability'],
                'career_interest': mentor['career_interest'],
                'score': float(score)
            })
        
        # Get top 3
        top_matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:3]
        
        return render_template('results.html', 
                             student_name=student_data['name'],
                             matches=top_matches)
    
    except Exception as e:
        return f"Error: {str(e)}", 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON requests"""
    
    try:
        student_data = request.get_json()
        
        matches = []
        for _, mentor in mentors_df.iterrows():
            similarities = calculate_similarity(student_data, mentor)
            feature_vector = [similarities.get(col, 0.5) for col in feature_cols]
            score = trained_model.predict([feature_vector])[0] * 100
            
            matches.append({
                'mentor_id': mentor['user_id'],
                'name': mentor['name'],
                'company': mentor['company'],
                'score': float(score)
            })
        
        top_matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:3]
        
        return jsonify({'matches': top_matches})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n🚀 Starting Mentor Matching Website...")
    print("📍 Open: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
