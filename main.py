import os
import sys
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import streamlit as st

# 환경 변수 로드
load_dotenv()

# Azure OpenAI 설정을 위한 환경 변수들
# .env 파일에 다음 변수들을 설정해야 합니다:
# AZURE_OPENAI_API_KEY=your_api_key
# AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
# AZURE_OPENAI_API_VERSION=2023-12-01-preview
# AZURE_OPENAI_CHAT_DEPLOYMENT=your-deployment-name

# Chat 모델 (대화용 - gpt-4o는 chat 모델만 지원)
@st.cache_resource
def get_chat_model():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        temperature=0.7,
        max_tokens=1000
    )

def chat_with_ai(user_input):
    """사용자 입력을 받아서 AI 응답을 반환하는 함수"""
    try:
        chat_model = get_chat_model()
        message = HumanMessage(content=user_input)
        response = chat_model.invoke([message])
        return response.content
    except Exception as e:
        return f"오류가 발생했습니다: {e}"

# Streamlit 웹 앱
st.title("🤖 Azure OpenAI 대화 앱")
st.markdown(f"**사용 중인 모델:** {os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT')}")

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
        with st.spinner("생각 중..."):
            response = chat_with_ai(prompt)
        st.markdown(response)
    
    # AI 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
# 사이드바에 대화 초기화 버튼
with st.sidebar:
    st.header("옵션")
    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

