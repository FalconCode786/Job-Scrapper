import os
import re
import json
from io import BytesIO
from typing import List, Dict, Any, Set
import requests
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import time
from bs4 import BeautifulSoup

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

job_cache = {}
CACHE_TIMEOUT = 300

class SkillExtractor:
    """Dynamic hierarchical skill extraction without hardcoded lists"""
    
    @staticmethod
    def extract_all_categories(text: str) -> Dict[str, Any]:
        """Extract skills hierarchically based on CV structure only"""
        text_raw = text
        text_lower = text.lower()
        
        results = {
            'technical': set(),
            'general': set(),
            'soft': set(),
            'experience_years': 0,
            'raw_text': text_raw
        }
        
        # 1. Extract from Technical Sections
        tech_sections = SkillExtractor._find_sections(text_lower, [
            r'technical skills?[:\s]+(.*?)(?=\n\n|\n[A-Z]|education|experience|$)',
            r'programming languages?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'technologies?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'tech stack[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'frameworks?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'development tools?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'software[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'hard skills?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
        ])
        
        for section in tech_sections:
            skills = SkillExtractor._parse_skill_list(section)
            for skill in skills:
                if SkillExtractor._is_valid_skill(skill):
                    results['technical'].add(skill)
        
        # 2. Extract from Soft Skills Sections
        soft_sections = SkillExtractor._find_sections(text_lower, [
            r'soft skills?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'interpersonal skills?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'personal skills?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'attributes?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'strengths?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
        ])
        
        for section in soft_sections:
            skills = SkillExtractor._parse_skill_list(section)
            for skill in skills:
                if SkillExtractor._is_valid_skill(skill):
                    results['soft'].add(skill)
        
        # 3. Extract from General Skills Sections
        general_sections = SkillExtractor._find_sections(text_lower, [
            r'skills?[:\s]+(.*?)(?=\n\n|\n[A-Z]|education|experience|$)',
            r'competencies?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'capabilities?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'expertise?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'proficiencies?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
            r'key skills?[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)',
        ])
        
        for section in general_sections:
            skills = SkillExtractor._parse_skill_list(section)
            for skill in skills:
                if SkillExtractor._is_valid_skill(skill) and skill not in results['technical'] and skill not in results['soft']:
                    category = SkillExtractor._categorize_dynamically(skill)
                    results[category].add(skill)
        
        # 4. Deep contextual extraction (when sections are sparse)
        if len(results['technical']) < 2 and len(results['soft']) < 2:
            contextual_skills = SkillExtractor._extract_from_context(text_lower)
            for skill in contextual_skills:
                if skill not in results['technical'] and skill not in results['soft'] and skill not in results['general']:
                    category = SkillExtractor._categorize_dynamically(skill)
                    results[category].add(skill)
        
        # 5. Extract experience
        results['experience_years'] = SkillExtractor._extract_experience(text_lower)
        
        # Determine primary category for search prioritization
        if results['technical']:
            search_terms = sorted(results['technical'], key=len, reverse=True)[:5]
            primary_category = 'technical'
        elif results['general']:
            search_terms = sorted(results['general'], key=len, reverse=True)[:5]
            primary_category = 'general'
        elif results['soft']:
            search_terms = sorted(results['soft'], key=len, reverse=True)[:5]
            primary_category = 'soft'
        else:
            # Ultimate fallback
            words = re.findall(r'\b[a-z]{4,}\b', text_lower)
            freq = Counter(words).most_common(5)
            search_terms = [w[0] for w in freq if w[0] not in ['this', 'that', 'with', 'have', 'from', 'they', 'we', 'are', 'was', 'will', 'skills', 'experience', 'ability', 'abilities']]
            if not search_terms:
                search_terms = ['professional']
            primary_category = 'default'
        
        return {
            'skills': list(set(list(results['technical']) + list(results['general']) + list(results['soft']))),
            'technical_skills': list(results['technical']),
            'general_skills': list(results['general']),
            'soft_skills': list(results['soft']),
            'primary_category': primary_category,
            'experience_years': results['experience_years'],
            'raw_text': text_raw,
            'search_query': ' '.join(search_terms[:5]),
            'top_skills': list(results['technical'])[:3] + list(results['general'])[:2]
        }
    
    @staticmethod
    def _categorize_dynamically(skill: str) -> str:
        """Categorize skill dynamically without hardcoded lists"""
        skill_lower = skill.lower()
        
        # Technical indicators (patterns)
        tech_patterns = [
            r'\+\+', r'#', r'\.js', r'\.net', r'\.py', r'css', r'html', r'sql',
            r'database', r'framework', r'library', r'server', r'cloud',
            r'\bjs\b', r'\bts\b', r'\bpy\b', r'\brb\b', r'\bgo\b', r'\bphp\b'
        ]
        
        if any(re.search(pattern, skill_lower) for pattern in tech_patterns):
            return 'technical'
        
        # Soft skill indicators (patterns)
        soft_patterns = [
            r'communication', r'leadership', r'management', r'teamwork',
            r'collaboration', r'problem solving', r'critical thinking',
            r'creativity', r'adaptability', r'organization', r'planning',
            r'presentation', r'negotiation', r'mentoring', r'analytical',
            r'time management', r'multitasking'
        ]
        
        if any(pattern in skill_lower for pattern in soft_patterns):
            return 'soft'
        
        # Heuristic: longer phrases more likely soft skills
        if len(skill.split()) > 1:
            return 'soft'
        
        return 'general'
    
    @staticmethod
    def _find_sections(text: str, patterns: List[str]) -> List[str]:
        """Find text sections matching patterns"""
        sections = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                content = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if content:
                    sections.append(content)
        return sections
    
    @staticmethod
    def _parse_skill_list(text: str) -> Set[str]:
        """Parse comma/bullet separated skills"""
        skills = set()
        parts = re.split(r'[,•|;/\n\r]+', text)
        for part in parts:
            skill = part.strip().lower()
            skill = re.sub(r'\d+\+?\s*years?(?:\s+of)?(?:\s+experience)?', '', skill).strip()
            skill = re.sub(r'^\W+|\W+$', '', skill)
            skill = re.sub(r'\s+', ' ', skill)
            if 2 < len(skill) < 35:
                fillers = ['and', 'or', 'with', 'using', 'etc', 'various', 'including', 'such as']
                if skill not in fillers:
                    skills.add(skill)
        return skills
    
    @staticmethod
    def _is_valid_skill(skill: str) -> bool:
        """Validate skill string"""
        if len(skill) < 2 or len(skill) > 40:
            return False
        if not re.search(r'[a-z]', skill):
            return False
        if skill.isdigit():
            return False
        common_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out']
        if skill in common_words:
            return False
        return True
    
    @staticmethod
    def _extract_from_context(text: str) -> Set[str]:
        """Extract skills from full text context"""
        skills = set()
        
        # Capitalized compound terms
        compound_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        for match in re.finditer(compound_pattern, text):
            term = match.group(1).lower()
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context = text[context_start:context_end].lower()
            skill_indicators = ['skill', 'experience', 'proficient', 'expert', 'using', 'worked']
            if any(ind in context for ind in skill_indicators):
                if len(term) < 40 and ' ' in term:
                    skills.add(term)
        
        # Terms with technical symbols
        tech_symbols = r'\b\w*(?:\+\+|#|\.js|\.net|\.py|\.go|\.rb|\.java)\b'
        for match in re.finditer(tech_symbols, text.lower()):
            skill = match.group().strip('.')
            if skill and len(skill) > 1:
                skills.add(skill)
        
        # N-grams in skill contexts
        skill_contexts = re.findall(r'(?:skills?|experience|proficient|knowledge)(?:\s+(?:in|with|of))?\s+([a-z][a-z\s,]+)', text)
        for context in skill_contexts:
            parts = re.split(r'[,\n\r]', context)
            for part in parts[:5]:
                skill = part.strip()
                if 2 < len(skill) < 30:
                    skills.add(skill)
        
        return skills
    
    @staticmethod
    def _extract_experience(text: str) -> int:
        """Extract years of experience"""
        patterns = [
            r'(\d+)\+?\s*years?(?:\s+of)?(?:\s+professional)?(?:\s+relevant)?\s+experience',
            r'experience\s*[:\s]\s*(\d+)\+?\s*years',
            r'(\d+)\+?\s*yrs?(?:\s+exp)?',
            r'over\s+(\d+)\s+years'
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return 0

class LocalMatcher:
    """Advanced semantic matching"""
    
    @staticmethod
    def calculate_relevance(cv_data: Dict, job: Dict) -> Dict[str, Any]:
        """Calculate detailed relevance with breakdown"""
        if not cv_data:
            return {'total': 0, 'breakdown': {}}
            
        cv_text = cv_data.get('raw_text', '')
        all_skills = cv_data.get('skills', [])
        
        job_title = job.get('title', '').lower()
        job_desc = job.get('description', '').lower()
        job_tags = [t.lower() for t in job.get('tags', [])]
        job_full = f"{job_title} {job_desc} {' '.join(job_tags)}"
        
        scores = {}
        
        # 1. TF-IDF Cosine Similarity
        try:
            if len(cv_text) > 10 and len(job_full) > 10:
                vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
                tfidf = vectorizer.fit_transform([cv_text, job_full])
                cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                scores['semantic'] = min(cos_sim * 100, 100)
            else:
                scores['semantic'] = 0
        except Exception as e:
            print(f"TF-IDF error: {e}")
            scores['semantic'] = 0
        
        # 2. Skill Matches
        if all_skills:
            exact_matches = sum(1 for skill in all_skills if skill.lower() in job_full)
            coverage = (exact_matches / len(all_skills) * 100) if all_skills else 0
            scores['skills'] = min(coverage * 1.5, 100)
        else:
            scores['skills'] = 0
        
        # 3. Title Relevance
        title_score = 0
        if all_skills:
            for skill in all_skills[:15]:
                words = skill.lower().split()
                for word in words:
                    if len(word) > 2 and word in job_title:
                        title_score += 8
        scores['title'] = min(title_score, 100)
        
        # 4. Tag Matches
        if job_tags and all_skills:
            tag_matches = sum(1 for skill in all_skills if any(skill in tag or tag in skill for tag in job_tags))
            scores['tags'] = min(tag_matches * 10, 100)
        else:
            scores['tags'] = 0
        
        # Weighted total
        total = (
            scores.get('semantic', 0) * 0.35 +
            scores.get('skills', 0) * 0.35 +
            scores.get('title', 0) * 0.20 +
            scores.get('tags', 0) * 0.10
        )
        
        # Experience alignment
        exp_years = cv_data.get('experience_years', 0)
        if exp_years >= 5 and any(x in job_title for x in ['senior', 'lead', 'principal']):
            total += 5
        elif exp_years <= 2 and any(x in job_title for x in ['junior', 'entry', 'intern', 'graduate']):
            total += 5
        
        return {
            'total': min(round(total), 100),
            'breakdown': {k: round(v, 1) for k, v in scores.items()}
        }

class JobScraper:
    @staticmethod
    def get_cached_or_fetch(cache_key, fetch_func):
        current_time = time.time()
        if cache_key in job_cache:
            data, timestamp = job_cache[cache_key]
            if current_time - timestamp < CACHE_TIMEOUT:
                return data
        try:
            data = fetch_func()
            if data is None:
                data = []
            job_cache[cache_key] = (data, current_time)
            return data
        except Exception as e:
            print(f"Fetch error for {cache_key}: {e}")
            if cache_key in job_cache:
                return job_cache[cache_key][0]
            return []
    
    @staticmethod
    def scrape_remoteok(query=""):
        def fetch():
            url = "https://remoteok.com/api"
            if query:
                first = query.split()[0] if query else ""
                if first:
                    url += f"?tag={first.lower()}"
            headers = {'User-Agent': 'Mozilla/5.0 (JobMatcher/1.0)', 'Accept': 'application/json'}
            resp = requests.get(url, headers=headers, timeout=10)
            jobs = []
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    start_idx = 1 if len(data) > 0 and isinstance(data[0], dict) and 'id' not in data[0] else 0
                    for item in data[start_idx:]:
                        if isinstance(item, dict) and 'id' in item:
                            desc = BeautifulSoup(item.get('description', ''), 'html.parser').get_text()
                            jobs.append({
                                'id': f"rok_{item['id']}",
                                'title': item.get('position', 'Unknown'),
                                'company': item.get('company', 'Unknown'),
                                'location': item.get('location', 'Remote') if item.get('location') else 'Remote',
                                'description': desc[:400] + '...' if len(desc) > 400 else desc,
                                'url': item.get('url', item.get('apply_url', '')),
                                'remote': True,
                                'salary': item.get('salary', ''),
                                'tags': item.get('tags', []),
                                'source': 'RemoteOK'
                            })
            return jobs
        return JobScraper.get_cached_or_fetch(f'rok_{query}', fetch)
    
    @staticmethod
    def scrape_arbeitnow(remote=None, location="", query=""):
        def fetch():
            url = "https://www.arbeitnow.com/api/job-board-api"
            headers = {'User-Agent': 'Mozilla/5.0 (JobMatcher/1.0)', 'Accept': 'application/json'}
            params = {}
            if remote is not None:
                params['remote'] = 'true' if remote else 'false'
            if query:
                params['search'] = query[:50]
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            jobs = []
            if resp.status_code == 200:
                data = resp.json()
                data_list = data.get('data', [])
                for item in data_list:
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
                        'description': desc[:400] + '...' if len(desc) > 400 else desc,
                        'url': item.get('url', ''),
                        'remote': item.get('remote', False),
                        'salary': f"{item.get('salary_min', '')}-{item.get('salary_max', '')} {item.get('salary_currency', '')}",
                        'tags': tags,
                        'source': 'Arbeitnow',
                        'visa_sponsorship': item.get('visa_sponsorship', False)
                    })
            return jobs
        return JobScraper.get_cached_or_fetch(f'arb_{remote}_{location}_{query}', fetch)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload-cv', methods=['POST'])
def upload_cv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        if not file.filename.endswith(('.pdf', '.docx', '.txt')):
            return jsonify({'error': 'Invalid file type'}), 400
        
        text = None
        if file.filename.endswith('.pdf'):
            try:
                pdf = PdfReader(file)
                text = "".join(page.extract_text() or "" for page in pdf.pages)
            except Exception as e:
                print(f"PDF Error: {e}")
                return jsonify({'error': 'Could not read PDF'}), 400
        elif file.filename.endswith('.docx'):
            try:
                doc = Document(file)
                text = "\n".join(para.text for para in doc.paragraphs)
            except Exception as e:
                print(f"DOCX Error: {e}")
                return jsonify({'error': 'Could not read DOCX'}), 400
        else:
            try:
                text = file.read().decode('utf-8')
            except Exception as e:
                print(f"TXT Error: {e}")
                return jsonify({'error': 'Could not read file'}), 400
        
        if not text:
            return jsonify({'error': 'Empty file or no text found'}), 400
        
        cv_data = SkillExtractor.extract_all_categories(text)
        session['cv_data'] = cv_data
        
        return jsonify({
            'success': True,
            'skills': cv_data.get('skills', []),
            'technical_skills': cv_data.get('technical_skills', []),
            'general_skills': cv_data.get('general_skills', []),
            'soft_skills': cv_data.get('soft_skills', []),
            'primary_category': cv_data.get('primary_category', 'general'),
            'experience_years': cv_data.get('experience_years', 0),
            'search_query': cv_data.get('search_query', ''),
            'top_skills': cv_data.get('top_skills', [])
        })
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': 'Server error processing file'}), 500

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    try:
        remote = request.args.get('remote', 'all')
        location = request.args.get('location', '')
        manual_query = request.args.get('query', '')
        
        cv_data = session.get('cv_data')
        
        # Safe search query construction
        if manual_query:
            search_query = manual_query
        elif cv_data and isinstance(cv_data, dict):
            search_query = cv_data.get('search_query', '')
        else:
            search_query = ''
        
        remote_only = remote == 'true'
        onsite_only = remote == 'false'
        
        jobs = []
        
        try:
            if not onsite_only:
                remote_jobs = JobScraper.scrape_remoteok(search_query)
                if remote_jobs:
                    jobs.extend(remote_jobs)
        except Exception as e:
            print(f"RemoteOK error in route: {e}")
        
        try:
            arb_remote = True if remote_only else (False if onsite_only else None)
            arb_jobs = JobScraper.scrape_arbeitnow(arb_remote, location, search_query)
            if arb_jobs:
                jobs.extend(arb_jobs)
        except Exception as e:
            print(f"Arbeitnow error in route: {e}")
        
        # Remove duplicates safely
        seen = set()
        unique_jobs = []
        for job in jobs:
            if isinstance(job, dict):
                key = f"{job.get('title', '')}-{job.get('company', '')}"
                if key not in seen:
                    seen.add(key)
                    unique_jobs.append(job)
        
        # Safe matching
        if cv_data and isinstance(cv_data, dict):
            matcher = LocalMatcher()
            for job in unique_jobs:
                try:
                    match_result = matcher.calculate_relevance(cv_data, job)
                    job['match_score'] = match_result.get('total')
                    job['match_breakdown'] = match_result.get('breakdown', {})
                except Exception as e:
                    print(f"Matching error for job: {e}")
                    job['match_score'] = None
        else:
            for job in unique_jobs:
                job['match_score'] = None
        
        # Safe sort
        unique_jobs.sort(key=lambda x: (x.get('match_score') or 0, x.get('title', '')), reverse=True)
        
        return jsonify({
            'count': len(unique_jobs),
            'jobs': unique_jobs,
            'search_query_used': search_query,
            'primary_category': cv_data.get('primary_category', 'none') if cv_data and isinstance(cv_data, dict) else 'none'
        })
    except Exception as e:
        print(f"Get jobs error: {str(e)}")
        return jsonify({'error': str(e), 'jobs': [], 'count': 0}), 500

@app.route('/api/clear-cv', methods=['POST'])
def clear_cv():
    try:
        session.pop('cv_data', None)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)