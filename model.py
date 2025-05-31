from pydantic import BaseModel
from typing import List, Optional, Dict

class UserResponse(BaseModel):
    _id: Optional[str]
    firstName: Optional[str]
    lastName: Optional[str]
    phone: Optional[int]
    userName: Optional[str]
    age: Optional[int]
    email: Optional[str]
    password: Optional[str]
    gender: Optional[str]
    avatar: Optional[str]
    nationality: Optional[str]
    currentMood: Optional[str]
    personalityWords: Optional[List[str]]
    workStyle: Optional[str]
    hobbies: Optional[List[str]]
    socialLevel: Optional[str]
    stressLevel: Optional[str]
    copingMechanisms: Optional[List[str]]
    shortTermGoals: Optional[List[str]]
    areasToImprove: Optional[List[str]]
    careerInterests: Optional[List[str]]
    supportSystem: Optional[List[str]]
    seeksValidation: Optional[str]
    trustLevel: Optional[int]
    decisionStyle: Optional[str]
    createdAt: Optional[str]
    updatedAt: Optional[str]
    employment: Optional[str]
    __v: Optional[int]

class AssessmentRequest(BaseModel):
    category: str
    user: UserResponse
    answers: Optional[List[str]] = None
    temperature: float = 0.7

class AssessmentResponse(BaseModel):
    questions: List[Dict]

class AssessmentResponsee(BaseModel):
    risk_prediction: str
    risk_probability: List[float]
    mental_state_summary: str
    personalized_recommendation: str

class PredictionRequest(BaseModel):
    answers: List[str]
    user: Optional[UserResponse] = None

class PredictionResponse(BaseModel):
    risk_prediction: str
    risk_probability: List[float]

class QuestionAnswer(BaseModel):
    question: str
    selectedOption: str

class AssessmentRequestt(BaseModel):
    category: str
    user: UserResponse
    questionAnswers: Optional[List[QuestionAnswer]] = None
    temperature: float = 0.7
