# PokeDex AI Server

포켓몬 사진 AI 판별 서버

## API 명세서

- AI 서버 API
  - 엔드포인트: POST /pokemon/predictions
    - 사용자가 촬영한 사진을 AI 서버로 전송하여 어떤 포켓몬인지 판별 후 포켓몬 번호 응답
    - 요청 데이터: multipart/form-data
    - 응답 데이터

```json
{
  "number": 1,
  "confidence": 0.14,
  "result": "fail"
}
```

        - result: 신뢰도 50% 미만인 경우 fail, 이상인 경우 success

## 사용 모델

- 모델명: [skshmjn/Pokemon-classifier-gen9-1025](https://huggingface.co/skshmjn/Pokemon-classifier-gen9-1025)
- 출처: Hugging Face
- 라이선스: Apache 2.0
- 설명: ViT(Vision Transformer) 기반으로 Gen 9까지 1025종의 포켓몬 이미지를 분류하는 파인튜닝 모델

## 기술 스택

- Python
- FastAPI
- PyTorch
- Hugging Face Transformers
