from fastapi import APIRouter, Body, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import requests

# Hugging Face API 配置
HF_API_URL = "https://api-inference.huggingface.co/models/your-model-name"  # 替换为你的模型名称
HF_API_TOKEN = "your-huggingface-api-token"  # 替换为你的 Hugging Face API Token

# 创建路由
router = APIRouter()

# 定义请求和响应模型（保持不变）
class ModelEvaluationRequest(BaseModel):
    model_id: str
    questions: List[Dict[str, Any]]  # 每个问题需包含 "content" 和 "correct_answer"

class ModelResponse(BaseModel):
    question_id: str
    model_id: str
    response_content: str
    is_correct: bool  # 新增：判断模型回答是否正确

class EvaluationResult(BaseModel):
    model_id: str
    responses: List[ModelResponse]  # 仅返回每个问题的正确性


@router.post("/evaluate", response_model=EvaluationResult)
async def evaluate_model(request: ModelEvaluationRequest):
    try:
        responses = []
        for i, question in enumerate(request.questions):
            # 调用模型生成回答（假设模型返回文本）
            response_content = model_pipeline(
                question["content"],
                max_length=50,
                num_return_sequences=1
            )[0]["generated_text"]

            # 判断是否正确（示例：直接匹配）
            is_correct = response_content.strip() == question["correct_answer"].strip()

            response = ModelResponse(
                question_id=question.get("id", f"q{i+1}"),
                model_id=request.model_id,
                response_content=response_content,
                is_correct=is_correct
            )
            responses.append(response)

        return EvaluationResult(
            model_id=request.model_id,
            responses=responses
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型评测失败: {str(e)}")
