import os
import sys
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st

# 환경 변수 로드
load_dotenv()

# Ollama 설정을 위한 환경 변수들
# .env 파일에 다음 변수들을 설정할 수 있습니다 (선택사항):
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama2:7b

# Chat 모델 (Ollama 사용)
@st.cache_resource
def get_ollama_model():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "llama2:7b")
    
    return Ollama(
        model=model_name,
        base_url=base_url,
        temperature=0.7
    )

def chat_with_ai(user_input):
    """사용자 입력을 받아서 AI 응답을 반환하는 함수"""
    try:
        ollama_model = get_ollama_model()
        response = ollama_model.invoke(user_input)
        return response
    except Exception as e:
        return f"오류가 발생했습니다: {e}\n\n💡 Ollama가 실행 중인지 확인해주세요:\n- ollama serve\n- ollama pull {os.getenv('OLLAMA_MODEL', 'llama2:7b')}"

# Streamlit 웹 앱
st.title("🦙 Ollama 대화 앱")
st.markdown(f"**사용 중인 모델:** {os.getenv('OLLAMA_MODEL', 'llama2:7b')}")
st.markdown(f"**Ollama 서버:** {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")

# 연결 상태 확인
with st.sidebar:
    st.header("🔧 Ollama 설정")
    
    # 모델 선택
    available_models = ["llama2:7b", "llama2:13b", "codellama:7b", "mistral:7b", "gemma:7b"]
    selected_model = st.selectbox(
        "모델 선택",
        available_models,
        index=0 if os.getenv('OLLAMA_MODEL') is None else available_models.index(os.getenv('OLLAMA_MODEL', 'llama2:7b'))
    )
    
    # 환경 변수 업데이트 (세션 상태로)
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = selected_model
    
    if st.session_state.selected_model != selected_model:
        st.session_state.selected_model = selected_model
        os.environ["OLLAMA_MODEL"] = selected_model
        st.cache_resource.clear()  # 캐시 클리어
    
    # 연결 테스트 버튼
    if st.button("🔍 연결 테스트"):
        with st.spinner("Ollama 연결 확인 중..."):
            test_response = chat_with_ai("Hello")
            if "오류가 발생했습니다" in test_response:
                st.error("❌ Ollama 연결 실패")
                st.code(test_response)
            else:
                st.success("✅ Ollama 연결 성공")
    
    st.markdown("---")
    st.header("📋 사용법")
    st.markdown("""
    **Ollama 설치 및 실행:**
    ```bash
    # 1. Ollama 설치
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # 2. 모델 다운로드
    ollama pull llama2:7b
    
    # 3. 서버 실행
    ollama serve
    ```
    """)
    
    if st.button("🗑️ 대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# 세션 상태 초기화 (대화 기록 저장용)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("🦙 Llama가 생각 중..."):
            # 현재 선택된 모델 사용
            os.environ["OLLAMA_MODEL"] = st.session_state.get("selected_model", "llama2:7b")
            response = chat_with_ai(prompt)
        st.markdown(response)
    
    # AI 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": response})