import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# 환경 변수 로드
load_dotenv()

def test_azure_openai_connection():
    """Azure OpenAI 연결 및 배포 테스트"""
    
    print("🔍 Azure OpenAI 연결 테스트 시작...")
    
    # 환경 변수 확인
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    
    print(f"📍 엔드포인트: {endpoint}")
    print(f"🔑 API 키: {api_key[:10]}...{api_key[-5:] if api_key else 'None'}")
    print(f"📅 API 버전: {api_version}")
    print(f"💬 Chat 배포: {chat_deployment}")
    print(f"🔍 Embedding 배포: {embedding_deployment}")
    print()
    
    if not all([endpoint, api_key, api_version]):
        print("❌ 필수 환경 변수가 설정되지 않았습니다!")
        return False
    
    # Azure OpenAI 클라이언트 생성
    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        print("✅ Azure OpenAI 클라이언트 생성 성공")
    except Exception as e:
        print(f"❌ 클라이언트 생성 실패: {e}")
        return False
    
    # Chat 모델 테스트
    if chat_deployment:
        print(f"\n💬 Chat 모델 테스트 ({chat_deployment})...")
        try:
            response = client.chat.completions.create(
                model=chat_deployment,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            print("✅ Chat 모델 연결 성공!")
            print(f"   응답: {response.choices[0].message.content}")
        except Exception as e:
            print(f"❌ Chat 모델 연결 실패: {e}")
            print("💡 해결 방법:")
            print("   1. Azure Portal에서 Chat 모델을 배포하세요")
            print("   2. 배포명이 정확한지 확인하세요")
            print(f"   3. 현재 설정: {chat_deployment}")
    
    # Embedding 모델 테스트
    if embedding_deployment:
        print(f"\n🔍 Embedding 모델 테스트 ({embedding_deployment})...")
        try:
            response = client.embeddings.create(
                model=embedding_deployment,
                input="Hello"
            )
            print("✅ Embedding 모델 연결 성공!")
            print(f"   벡터 차원: {len(response.data[0].embedding)}")
        except Exception as e:
            print(f"❌ Embedding 모델 연결 실패: {e}")
            print("💡 해결 방법:")
            print("   1. Azure Portal에서 Embedding 모델을 배포하세요")
            print("   2. 배포명이 정확한지 확인하세요")
            print(f"   3. 현재 설정: {embedding_deployment}")
    
    print("\n🔧 문제 해결 가이드:")
    print("1. Azure OpenAI Studio 접속: https://oai.azure.com")
    print("2. Deployments 탭에서 배포 상태 확인")
    print("3. 필요한 모델이 없다면 'Create new deployment' 클릭")
    print("4. 배포명을 .env 파일과 정확히 일치시키기")
    print("5. 배포 후 5분 정도 기다린 후 재시도")

if __name__ == "__main__":
    test_azure_openai_connection() 