from fastapi import APIRouter, Body, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# 创建路由
router = APIRouter()

# 定义请求和响应模型
class ModelEvaluationRequest(BaseModel):
    model_id: str
    questions: List[Dict[str, Any]]
    evaluation_metrics: List[str] = ["accuracy", "fluency", "relevance"]

class ModelResponse(BaseModel):
    question_id: str
    model_id: str
    response_content: str
    score: Optional[float] = None

class EvaluationResult(BaseModel):
    model_id: str
    total_score: float
    metrics_scores: Dict[str, float]
    responses: List[ModelResponse]

@router.post("/evaluate", response_model=EvaluationResult)
async def evaluate_model(request: ModelEvaluationRequest):
    """
    评测指定模型在给定题目上的表现
    """
    try:
        responses = []
        for i, question in enumerate(request.questions):
            response = ModelResponse(
                question_id=question.get("id", f"q{i+1}"),
                model_id=request.model_id,
                response_content=f"这是{request.model_id}对问题{question.get('id', f'q{i+1}')}的回答",
                score=0.85
            )
            responses.append(response)
        
        # 计算总分和各指标分数
        metrics_scores = {metric: 0.8 + (i * 0.05) for i, metric in enumerate(request.evaluation_metrics)}
        total_score = sum(metrics_scores.values()) / len(metrics_scores)
        
        return EvaluationResult(
            model_id=request.model_id,
            total_score=total_score,
            metrics_scores=metrics_scores,
            responses=responses
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型评测失败: {str(e)}")

@router.get("/models", response_model=List[Dict[str, Any]])
async def get_available_models():
    """
    获取可用于评测的模型列表
    """
    models = [
        {
            "id": "gpt-3.5-turbo",
            "name": "GPT-3.5 Turbo",
            "provider": "OpenAI",
            "description": "OpenAI的GPT-3.5模型"
        },
        {
            "id": "glm-4",
            "name": "GLM-4",
            "provider": "智谱AI",
            "description": "智谱AI的GLM-4模型"
        },
        {
            "id": "wenxin",
            "name": "文心一言",
            "provider": "百度",
            "description": "百度的文心一言模型"
        },
        {
            "id": "custom-model",
            "name": "自主训练模型",
            "provider": "本地",
            "description": "自主训练的大模型"
        }
    ]
    return models

@router.get("/results/{evaluation_id}", response_model=EvaluationResult)
async def get_evaluation_result(evaluation_id: str):
    """
    获取指定评测的结果
    """
    responses = [
        ModelResponse(
            question_id=f"q{i+1}",
            model_id="gpt-3.5-turbo",
            response_content=f"这是对问题q{i+1}的回答",
            score=0.85
        ) for i in range(5)
    ]
    
    metrics_scores = {"accuracy": 0.82, "fluency": 0.88, "relevance": 0.79}
    total_score = sum(metrics_scores.values()) / len(metrics_scores)
    
    return EvaluationResult(
        model_id="gpt-3.5-turbo",
        total_score=total_score,
        metrics_scores=metrics_scores,
        responses=responses
    )
