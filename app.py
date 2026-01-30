import os
import re
import json
from io import BytesIO
from typing import List, Dict, Any
import requests
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Serverless LLM Config (Optional - Free Tier Hugging Face)
HF_API_KEY = os.getenv('HF_API_KEY', '')
HF_MODEL = os.getenv('HF_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

class CVProcessor:
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @staticmethod
    def extract_text(file):
        filename = file.filename.lower()
        
        if filename.endswith('.pdf'):
            try:
                pdf = PdfReader(file)
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text
            except Exception as e:
                return None
                
        elif filename.endswith('.docx'):
            try:
                doc = Document(file)
                return "\n".join([para.text for para in doc.paragraphs])
            except:
                return None
                
        elif filename.endswith('.txt'):
            try:
                return file.read().decode('utf-8')
            except:
                return None
        return None
    
    @staticmethod
    def extract_skills_dynamic(text):
        """Dynamically extract skills from CV text without hardcoded lists"""
        text_lower = text.lower()
        extracted_skills = set()
        
        # Pattern 1: Look for explicit skill sections and extract lists
        # Matches: "Skills: Python, JavaScript, React" or "Technologies • Python • JavaScript"
        section_patterns = [
            r'(?:skills|technical skills|technologies|tech stack|tools|proficiencies)[\s:]+([^\n]+(?:\n[^\n]+){0,5})',
            r'(?:programming languages|frameworks|databases|platforms)[\s:]+([^\n]+(?:\n[^\n]+){0,3})',
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Split by common delimiters: commas, bullets, pipes, semicolons
                parts = re.split(r'[,•|;/\n]', match)
                for part in parts:
                    skill = part.strip().lower()
                    # Clean up: remove years, "years", "experience", etc.
                    skill = re.sub(r'\d+\+?\s*years?(?:\s+of)?(?:\s+experience)?', '', skill).strip()
                    skill = re.sub(r'experience(?:\s+with)?', '', skill).strip()
                    if len(skill) > 1 and len(skill) < 25 and ' ' not in skill or skill.count(' ') < 2:
                        if skill and skill not in ['and', 'or', 'with', 'using', 'etc', 'various']:
                            extracted_skills.add(skill)
        
        # Pattern 2: Extract specific technical patterns (languages with versions, etc)
        # Matches: Python 3, C++, C#, .NET, Node.js, etc.
        tech_patterns = [
            r'\b(python\s*\d?\.?\d*)\b',
            r'\b(javascript|typescript|js|ts)\b',
            r'\b(java\s*\d?)\b',
            r'\b(c\+\+|c#|\.net|node\.js|vue\.js|react\.js)\b',
            r'\b(aws|gcp|azure|docker|kubernetes|terraform)\b',
            r'\b(react|angular|vue|svelte|django|flask|rails)\b',
            r'\b(sql|nosql|mongodb|postgres|mysql|sqlite)\b',
            r'\b(tensorflow|pytorch|keras|scikit|pandas|numpy)\b',
            r'\b(git|github|gitlab|bitbucket)\b',
            r'\b(agile|scrum|kanban|jira|confluence)\b',
            r'\b(linux|ubuntu|centos|bash|shell)\b',
            r'\b(html5?|css3?|sass|less|tailwind|bootstrap)\b',
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                clean_match = match.strip().lower()
                if clean_match:
                    extracted_skills.add(clean_match)
        
        # Pattern 3: Extract words appearing near action verbs (contextual extraction)
        action_contexts = ['developed', 'built', 'created', 'implemented', 'using', 'programmed', 
                          'coded', 'architected', 'designed', 'maintained', 'integrated']
        words = re.findall(r'\b[a-z]+(?:\+\+|#)?\b', text_lower)
        
        for i, word in enumerate(words):
            if len(word) > 2:
                # Check surrounding context (3 words before and after)
                start = max(0, i-3)
                end = min(len(words), i+4)
                context = words[start:end]
                
                if any(action in context for action in action_contexts):
                    # Likely a skill/technology
                    if word not in ['the', 'and', 'for', 'with', 'using', 'built', 'developed']:
                        extracted_skills.add(word)
        
        # Pattern 4: Extract years of experience
        exp_patterns = [
            r'(\d+)\+?\s*years?(?:\s+of)?(?:\s+professional)?(?:\s+relevant)?\s+experience',
            r'experience\s*[:\s]\s*(\d+)\+?\s*years',
            r'(\d+)\+?\s*yrs?(?:\s+exp)?',
            r'over\s+(\d+)\s+years'
        ]
        years_exp = 0
        for pattern in exp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                years_exp = int(match.group(1))
                break
        
        # Build search query from top skills (prioritize multi-word and specific terms)
        prioritized_skills = sorted(extracted_skills, key=lambda x: len(x), reverse=True)[:8]
        search_query = ' '.join(prioritized_skills)
        
        return {
            'skills': list(extracted_skills),
            'experience_years': years_exp,
            'raw_text': text,
            'search_query': search_query,
            'top_skills': prioritized_skills[:5]
        }

class LLMClient:
    @staticmethod
    def is_available():
        return bool(HF_API_KEY)
    
    @staticmethod
    def extract_skills(text):
        """Use free serverless LLM to extract structured skills from CV"""
        if not LLMClient.is_available():
            return None
            
        prompt = f"""<s>[INST] Analyze this CV and extract ONLY the technical skills, programming languages, frameworks, and tools mentioned. Return ONLY a JSON object in this exact format:
{{
  "skills": ["skill1", "skill2", "skill3"],
  "experience_years": number,
  "search_query": "string of top 3 skills for job searching"
}}

CV text:
{text[:4000]} [/INST]"""
        
        try:
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 200,
                        "temperature": 0.1,
                        "return_full_text": False
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get('generated_text', '')
                    json_match = re.search(r'\{.*\}', generated, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
        except Exception as e:
            print(f"LLM Error: {e}")
        return None
    
    @staticmethod
    def calculate_relevance(cv_data, job_desc, job_title):
        if not LLMClient.is_available():
            return None
            
        prompt = f"""<s>[INST] Rate how well this CV matches the job (0-100). Consider skills match and experience level. Respond with ONLY the number.

CV Skills: {', '.join(cv_data.get('skills', []))}
Experience: {cv_data.get('experience_years', 0)} years

Job Title: {job_title}
Job Description: {job_desc[:1000]}

Score (0-100): [/INST]"""
        
        try:
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 3,
                        "temperature": 0.1,
                        "return_full_text": False
                    }
                },
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get('generated_text', '').strip()
                    numbers = re.findall(r'\d+', text)
                    if numbers:
                        score = int(numbers[0])
                        return min(100, max(0, score))
        except:
            pass
        return None

class JobScraper:
    @staticmethod
    def scrape_remoteok(query=""):
        """Scrape RemoteOK using dynamic query from CV"""
        try:
            url = "https://remoteok.com/api"
            if query:
                # RemoteOK uses tags, extract first keyword if multiple
                first_keyword = query.split()[0] if query else ""
                if first_keyword:
                    url += f"?tag={first_keyword.lower()}"
                
            headers = {
                'User-Agent': 'Mozilla/5.0 (JobScraper/1.0)',
                'Accept': 'application/json'
            }
            
            resp = requests.get(url, headers=headers, timeout=10)
            jobs = []
            
            if resp.status_code == 200:
                data = resp.json()
                for item in data[1:] if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'id' not in data[0] else data:
                    if isinstance(item, dict) and 'id' in item:
                        desc = BeautifulSoup(item.get('description', ''), 'html.parser').get_text()
                        jobs.append({
                            'id': f"rok_{item['id']}",
                            'title': item.get('position', 'Unknown'),
                            'company': item.get('company', 'Unknown'),
                            'location': item.get('location', 'Remote') if item.get('location') else 'Remote',
                            'description': desc[:500] + '...' if len(desc) > 500 else desc,
                            'url': item.get('url', item.get('apply_url', '')),
                            'remote': True,
                            'salary': item.get('salary', ''),
                            'tags': item.get('tags', []),
                            'source': 'RemoteOK'
                        })
            return jobs
        except Exception as e:
            print(f"RemoteOK error: {e}")
            return []
    
    @staticmethod
    def scrape_arbeitnow(remote=None, location="", query=""):
        """Scrape Arbeitnow with CV-based search"""
        try:
            url = "https://www.arbeitnow.com/api/job-board-api"
            headers = {
                'User-Agent': 'Mozilla/5.0 (JobScraper/1.0)',
                'Accept': 'application/json'
            }
            
            params = {}
            if remote is not None:
                params['remote'] = 'true' if remote else 'false'
            if query:
                params['search'] = query[:50]  # API limit
                
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            jobs = []
            
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get('data', []):
                    # Location filter
                    if location and location.lower() not in item.get('location', '').lower():
                        continue
                        
                    tags = item.get('tags', [])
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(',')]
                        
                    desc = BeautifulSoup(item.get('description', ''), 'html.parser').get_text()
                    
                    jobs.append({
                        'id': f"arb_{item.get('slug', '')}",
                        'title': item.get('title', 'Unknown'),
                        'company': item.get('company_name', 'Unknown'),
                        'location': item.get('location', 'Not specified'),
                        'description': desc[:500] + '...' if len(desc) > 500 else desc,
                        'url': item.get('url', ''),
                        'remote': item.get('remote', False),
                        'salary': f"{item.get('salary_min', '')}-{item.get('salary_max', '')} {item.get('salary_currency', '')}",
                        'tags': tags,
                        'source': 'Arbeitnow',
                        'visa_sponsorship': item.get('visa_sponsorship', False)
                    })
            return jobs
        except Exception as e:
            print(f"Arbeitnow error: {e}")
            return []

class Matcher:
    @staticmethod
    def calculate_similarity(cv_text, job_desc):
        try:
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([cv_text, job_desc])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(similarity[0][0]) * 100
        except:
            return 50.0

@app.route('/')
def index():
    return render_template('index.html', llm_available=LLMClient.is_available())

@app.route('/api/upload-cv', methods=['POST'])
def upload_cv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    if not CVProcessor.allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    text = CVProcessor.extract_text(file)
    if not text:
        return jsonify({'error': 'Could not extract text from file'}), 400
    
    # Try LLM first, fallback to dynamic extraction
    cv_data = None
    if LLMClient.is_available():
        cv_data = LLMClient.extract_skills(text)
    
    if not cv_data:
        cv_data = CVProcessor.extract_skills_dynamic(text)
    else:
        cv_data['raw_text'] = text
        if 'search_query' not in cv_data:
            cv_data['search_query'] = ' '.join(cv_data.get('skills', [])[:5])
    
    session['cv_data'] = cv_data
    
    return jsonify({
        'success': True,
        'skills': cv_data.get('skills', []),
        'experience_years': cv_data.get('experience_years', 0),
        'search_query': cv_data.get('search_query', ''),
        'top_skills': cv_data.get('top_skills', []),
        'llm_used': LLMClient.is_available()
    })

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    remote = request.args.get('remote', 'all')
    location = request.args.get('location', '')
    manual_query = request.args.get('query', '')
    
    cv_data = session.get('cv_data')
    
    # Use CV-extracted search query if no manual override provided
    search_query = manual_query
    if cv_data and not manual_query:
        search_query = cv_data.get('search_query', '')
        # If specific skills requested via frontend, append them
        additional = request.args.get('skills', '')
        if additional:
            search_query += ' ' + additional
    
    remote_only = remote == 'true'
    onsite_only = remote == 'false'
    
    jobs = []
    
    # Scrape with dynamic query from CV
    if not onsite_only:
        remote_jobs = JobScraper.scrape_remoteok(search_query)
        jobs.extend(remote_jobs)
    
    arb_remote = True if remote_only else (False if onsite_only else None)
    arb_jobs = JobScraper.scrape_arbeitnow(remote=arb_remote, location=location, query=search_query)
    jobs.extend(arb_jobs)
    
    # Remove duplicates
    seen = set()
    unique_jobs = []
    for job in jobs:
        key = f"{job['title']}-{job['company']}"
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)
    
    # Calculate match scores
    if cv_data:
        cv_text = cv_data.get('raw_text', '')
        for job in unique_jobs:
            llm_score = None
            if LLMClient.is_available():
                llm_score = LLMClient.calculate_relevance(cv_data, job['description'], job['title'])
            
            if llm_score is not None:
                job['match_score'] = llm_score
                job['match_method'] = 'llm'
            else:
                job_text = f"{job['title']} {job['description']} {' '.join(job['tags'])}"
                job['match_score'] = round(Matcher.calculate_similarity(cv_text, job_text), 1)
                job['match_method'] = 'dynamic'
    else:
        for job in unique_jobs:
            job['match_score'] = None
            job['match_method'] = 'none'
    
    # Sort by match score
    unique_jobs.sort(key=lambda x: (x['match_score'] or 0, x['title']), reverse=True)
    
    return jsonify({
        'count': len(unique_jobs),
        'jobs': unique_jobs,
        'search_query_used': search_query,
        'cv_loaded': cv_data is not None
    })

@app.route('/api/clear-cv', methods=['POST'])
def clear_cv():
    session.pop('cv_data', None)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)