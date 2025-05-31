from fastapi import APIRouter, HTTPException
from model import AssessmentRequest, AssessmentResponse, AssessmentResponsee
from service import generate_questions, preprocess_for_dqn, generate_personalized_recommendation, generate_mental_state_summary, dqn_model
import torch
from typing import List

router = APIRouter()

@router.post("/api/questions/genearte", response_model=AssessmentResponse)
async def assess_mental_health(request: AssessmentRequest):
    try:
        questions = generate_questions(request.user, request.category, request.temperature)
        return AssessmentResponse(questions=questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/response/generate", response_model=AssessmentResponsee)
async def assess_mental_health_full(request: AssessmentRequest):
    try:
        questions = generate_questions(request.user, request.category, request.temperature)

        if request.answers and len(request.answers) >= 3:
            combined_summary = " ".join(request.answers)
            feature_tensor = preprocess_for_dqn(combined_summary, request.user, request.answers)
            with torch.no_grad():
                logits = dqn_model(feature_tensor)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                predicted_index = int(torch.argmax(logits, dim=1).item())
                risk_labels = ["Low Risk", "Moderate Risk", "High Risk" , "Normal", "slow risk", "Growing Risk", "risky"]
                risk_prediction = risk_labels[predicted_index]
        else:
            risk_prediction = "Unknown"
            probabilities = [0.33, 0.33, 0.34]

        mental_state_summary = generate_mental_state_summary(request.user, request.answers, risk_prediction)

        personalized_recommendation = ""
        if request.answers and len(request.answers) >= 3:
            personalized_recommendation = generate_personalized_recommendation(request.user, request.answers, risk_prediction, request.temperature)
        else:
            personalized_recommendation = "Please answer all the questions to receive personalized recommendations."

        return AssessmentResponsee(
            risk_prediction=risk_prediction,
            risk_probability=probabilities.tolist(),
            mental_state_summary=mental_state_summary,
            personalized_recommendation=personalized_recommendation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
