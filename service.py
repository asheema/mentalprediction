import os
import torch
import numpy as np
import json
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import google.generativeai as genai

from model import UserResponse

# Setup
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()
embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
sentiment_classifier = pipeline("sentiment-analysis")

# Load environment variable and configure Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_chat = genai.GenerativeModel("models/gemini-1.5-flash-latest").start_chat()

from dqnagent import DQN  # Assuming this file is present with your model

# Load DQN model
input_dim = embedding_model.get_sentence_embedding_dimension() + 5
output_dim = 7
dqn_model = DQN(input_dim=input_dim, output_dim=output_dim)
dqn_model.load_state_dict(torch.load("dqn_mental_health_finetuned.pth", map_location=torch.device("cpu")), strict=False)
dqn_model.eval()

DEFAULT_OPTIONS = ["Strongly Agree", "Agree", "Neutral", "Disagree"]

def generate_questions(user: UserResponse, category: str, temperature: float):
    prompt = f"""
You are a compassionate mental health AI. Generate 10 warm, personalized multiple-choice questions for the user below, each with **4 options**.
Return ONLY a valid JSON array with format:

[
  {{
    "body": "What kind of work environment do you prefer?",
    "options": ["Quiet and independent", "Team-oriented", "Fast-paced", "Flexible"]
  }},
  ...
]

🛑 DO NOT add explanations, markdown, or text outside the JSON array.

Use this data to personalize:
- Stress Level: {user.stressLevel}
- Gender: {user.gender}
- Category: {category}
- firstName: {user.firstName}
- currentMood: {user.currentMood}
- personalityWords: {user.personalityWords} 
- workStyle: {user.workStyle}
- hobbies: {user.hobbies}
- socialLevel: {user.socialLevel}
"""
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=temperature))

    try:
        cleaned_data = json.loads(response.text)
    except json.JSONDecodeError:
        match = re.search(r'\[\s*{.*?}\s*\]', response.text, re.DOTALL)
        if match:
            cleaned_data = json.loads(match.group(0))
        else:
            raise ValueError("No valid JSON found in Gemini response")

    multiple_choice_questions = []
    for item in cleaned_data:
        body = item.get("body") or next(iter(item))
        options = item.get("options", DEFAULT_OPTIONS)
        multiple_choice_questions.append({"body": body, "options": options})

    # Ensure non-empty list
    if not multiple_choice_questions:
        multiple_choice_questions = add_default_options(["How are you feeling today?"] * 10)

    return multiple_choice_questions

def add_default_options(questions):
    return [{"body": q, "options": DEFAULT_OPTIONS} for q in questions]

def preprocess_for_dqn(summary: str, user: UserResponse, answers: list):
    combined_text = summary + " " + " ".join(answers)
    embedding = embedding_model.encode([combined_text])[0]
    sentiment_score = analyzer.polarity_scores(summary)["compound"]

    age = float(user.age) if user and isinstance(user.age, (int, float)) else 25
    stress = float(user.stressLevel) if user and isinstance(user.stressLevel, (int, float)) else 5

    gender = 0 if user and user.gender and user.gender.lower() == "male" else 1 if user and user.gender and user.gender.lower() == "female" else 2
    employment_map = {"employed": 0, "unemployed": 1, "student": 2, "retired": 3}
    employment = employment_map.get(user.employment.lower(), 4) if user and user.employment else 4

    extra_features = np.array([
        stress / 10,
        age / 100,
        sentiment_score,
        gender,
        employment
    ])

    features = np.concatenate([embedding, extra_features]).astype(np.float32)
    return torch.FloatTensor(features).unsqueeze(0)

def generate_personalized_recommendation(user: UserResponse, answers: list, risk_label: str, temperature: float):
    answer_summary = " ".join(answers)
    prompt = f"""
You are an empathetic mental health coach. The user has answered mental health questions with the following responses:
{answer_summary}

Their assessed mental health risk is: {risk_label}.
Please provide a warm, encouraging, personalized, and conversational recommendation and emotional support message tailored to their situation. Plaese keep it concise and small
"""
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=temperature))
    return response.text.strip()

def generate_mental_state_summary(user, answers, risk_prediction):
    summary_prompt = f"""
    A user has completed a mental health assessment.

    Details:
    - Name: {user.firstName}
    - Age: {user.age}
    - Gender: {user.gender}
    - Responses: {answers}
    - Risk Level: {risk_prediction}

    Provide a brief mental health summary of this user.
    """
    return gemini_chat.send_message(summary_prompt).text
