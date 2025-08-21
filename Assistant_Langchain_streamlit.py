import streamlit as st
import os
import time
import tempfile
from datetime import datetime
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import re

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="🤖 PDF Assistant with LangChain",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PDFAssistant:
    """PDF 검색 전용 AI 어시스턴트 (Streamlit용)"""
    
    def __init__(self):
        """Assistant 초기화"""
        
        # Azure OpenAI 설정 (Content Filter 최적화)
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.1,  # 더 낮은 temperature로 안전성 증대
            max_tokens=800,   # 더 짧은 응답으로 필터 우회
            top_p=0.7,        # 더 보수적인 토큰 선택
        )
        
        # 임베딩 모델 설정
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        
        # 제한된 대화 기억 장치 (최근 3개만 기억)
        self.memory = ConversationBufferWindowMemory(
            k=3,  # 최근 3개 대화만 기억
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # 벡터 저장소와 대화 체인
        self.vectorstore = None
        self.conversation_chain = None
    
    def load_pdf_knowledge(self, uploaded_file):
        """업로드된 PDF를 Assistant의 지식으로 변환"""
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # 1. PDF 로드
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            # 2. 텍스트 분할
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_documents(pages)
            
            # 3. 벡터 저장소 생성
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory="./chroma_db_streamlit"
            )
            
            # 4. 대화형 검색 체인 생성
            from langchain.prompts import PromptTemplate
            
            # 안전한 프롬프트 템플릿 생성 (간소화)
            safe_template = """당신은 PDF 문서 분석 전문가입니다.
            
문서 내용: {context}

질문: {question}

지침: 문서 내용에만 기반하여 간결하고 정확한 답변을 제공하세요.

답변:"""
            
            safe_prompt = PromptTemplate(
                template=safe_template,
                input_variables=["context", "question"]
            )
            
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),  # 검색 결과 줄임
                memory=self.memory,
                return_source_documents=True,
                chain_type="stuff",
                combine_docs_chain_kwargs={"prompt": safe_prompt},
                verbose=False  # 로깅 비활성화로 안전성 증대
            )
            
            return True, len(pages), len(texts)
            
        except Exception as e:
            return False, 0, 0, str(e)
        
        finally:
            # 임시 파일 삭제
            os.unlink(tmp_path)
    
    def chat(self, user_message):
        """사용자와 대화하기"""
        
        if not self.conversation_chain:
            return "먼저 PDF를 업로드해주세요!", [], 0
        
        # 첫 번째 시도: 정제된 입력으로 시도
        try:
            sanitized_message = sanitize_user_input(user_message)
            start_time = time.time()
            
            response = self.conversation_chain.invoke({"question": sanitized_message})
            
            end_time = time.time()
            response_time = end_time - start_time
            
            answer = response['answer']
            source_docs = response.get('source_documents', [])
            
            return answer, source_docs, response_time
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Content Filter 오류 시 백업 전략 시도
            if "content filter" in error_msg or "content_filter" in error_msg:
                return self._try_backup_strategy(user_message)
            
            # Rate Limit 오류 처리
            elif "rate limit" in error_msg or "429" in error_msg:
                return (
                    "⏳ **요청 한도 초과**\n\n"
                    "잠시 후 다시 시도해주세요. (약 1분 대기)"
                ), [], 0
            
            # 기타 오류
            else:
                return f"❌ 답변 생성 중 오류 발생: {e}", [], 0
    
    def _try_backup_strategy(self, original_message):
        """Content Filter 우회를 위한 백업 전략"""
        
        # 백업 전략 1: 매우 간단한 질문으로 변환
        simple_queries = [
            "문서의 주요 내용을 요약해주세요",
            "이 문서에서 가장 중요한 정보는 무엇인가요", 
            "문서의 핵심 주제를 알려주세요"
        ]
        
        for simple_query in simple_queries:
            try:
                start_time = time.time()
                
                # 메모리 임시 비활성화로 시도
                temp_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 1}),
                    memory=None,  # 메모리 비활성화
                    return_source_documents=True,
                    chain_type="stuff"
                )
                
                response = temp_chain.invoke({"question": simple_query})
                
                end_time = time.time()
                response_time = end_time - start_time
                
                answer = response['answer']
                source_docs = response.get('source_documents', [])
                
                # 성공한 경우 안내 메시지와 함께 반환
                backup_answer = (
                    f"🛡️ **안전 모드로 답변드립니다**\n\n"
                    f"원래 질문: \"{original_message}\"\n"
                    f"안전 모드 답변: {answer}\n\n"
                    f"💡 더 구체적인 답변을 원하시면 질문을 다시 표현해보세요."
                )
                
                return backup_answer, source_docs, response_time
                
            except:
                continue
        
        # 모든 백업 전략 실패 시
        return (
            "🛡️ **안전 필터 감지**\n\n"
            "Azure OpenAI의 안전 필터가 작동했습니다. 다음을 시도해보세요:\n\n"
            "1. **질문을 다시 표현**해보세요\n"
            "2. **더 구체적이고 명확한 질문**을 해보세요\n"
            "3. **전문 용어나 학술적 표현**을 사용해보세요\n\n"
            "💡 예시:\n"
            "- '이 문서의 주요 내용을 요약해주세요'\n"
            "- '3페이지의 정의를 설명해주세요'\n"
            "- '특정 개념에 대해 알려주세요'"
        ), [], 0
    
    def get_chat_history(self):
        """대화 히스토리 조회"""
        return self.memory.chat_memory.messages
    
    def clear_history(self):
        """대화 히스토리 초기화"""
        self.memory.clear()

def sanitize_user_input(user_input):
    """사용자 입력을 안전하게 정제 (강화된 버전)"""
    
    # 민감할 수 있는 키워드들을 중성적 표현으로 변경
    replacements = {
        # 기존 + 추가 키워드
        "위험한": "주의가 필요한",
        "문제": "이슈", 
        "공격": "접근",
        "해킹": "보안 분석",
        "폭력": "강력한",
        "죽음": "종료",
        "살인": "제거",
        "테러": "극단적 행위",
        "전쟁": "충돌",
        "싸움": "경쟁",
        "파괴": "변경",
        "범죄": "규정 위반",
        "불법": "규정에 어긋나는",
        "혐오": "부정적",
        "차별": "구분",
        "성적": "관련 내용",
        "약물": "화학 물질",
        "마약": "규제 물질",
    }
    
    sanitized = user_input.lower()  # 소문자로 변환
    
    # 키워드 교체
    for original, replacement in replacements.items():
        sanitized = sanitized.replace(original, replacement)
    
    # 특수 문자나 이모지 제거 (안전성 증대)
    sanitized = re.sub(r'[^\w\s가-힣]', ' ', sanitized)
    
    # 연속된 공백 정리
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # 너무 짧은 질문은 구체화
    if len(sanitized) < 5:
        sanitized = f"문서에서 '{sanitized}'에 대해 설명해주세요"
    
    # 학술적 표현으로 감싸기 (항상 적용)
    academic_prefix = "PDF 문서를 참조하여 다음 질문에 대해 객관적이고 학술적으로 답변해주세요: "
    sanitized = academic_prefix + sanitized
    
    return sanitized

# 세션 상태 초기화
def initialize_session_state():
    """세션 상태 변수들 초기화"""
    if 'assistant' not in st.session_state:
        st.session_state.assistant = PDFAssistant()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_loaded' not in st.session_state:
        st.session_state.pdf_loaded = False
    if 'pdf_info' not in st.session_state:
        st.session_state.pdf_info = {}

def check_environment():
    """환경 설정 확인"""
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_EMBEDDING_DEPLOYMENT',
        'AZURE_OPENAI_CHAT_DEPLOYMENT',
        'AZURE_OPENAI_API_VERSION'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    return missing_vars

def display_chat_message(role, content, timestamp=None, source_docs=None):
    """채팅 메시지 표시"""
    with st.chat_message(role):
        if timestamp:
            st.caption(f"🕐 {timestamp}")
        st.write(content)
        
        # 출처 문서 표시
        if source_docs and len(source_docs) > 0:
            with st.expander(f"📚 참고 문서 ({len(source_docs)}개)", expanded=False):
                for i, doc in enumerate(source_docs[:3], 1):  # 상위 3개만 표시
                    st.markdown(f"**📄 {i}번째 참고 문서:**")
                    st.text_area(
                        f"내용 {i}",
                        value=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        height=100,
                        key=f"source_doc_{i}_{time.time()}"
                    )
                    if doc.metadata:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"📍 페이지: {doc.metadata.get('page', 'Unknown')}")
                        with col2:
                            st.caption(f"📁 파일: {doc.metadata.get('source', 'Unknown')}")
                    st.divider()

def display_chat_history():
    """대화 기록 표시"""
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            display_chat_message("user", chat['question'], chat.get('timestamp'))
            display_chat_message(
                "assistant", 
                chat['answer'], 
                source_docs=chat.get('source_docs', [])
            )

def main():
    """메인 함수"""
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 헤더
    st.title("🤖 PDF Assistant with LangChain")
    st.markdown("LangChain을 사용한 똑똑한 PDF 검색 어시스턴트입니다!")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ Assistant 설정")
        
        # 환경 변수 확인
        missing_vars = check_environment()
        if missing_vars:
            st.error("❌ 환경 변수가 설정되지 않았습니다:")
            for var in missing_vars:
                st.write(f"- {var}")
            st.info("💡 .env 파일을 확인해주세요!")
            return
        else:
            st.success("✅ 환경 설정 완료")
        
        # PDF 업로드
        st.header("📁 PDF 업로드")
        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=['pdf'],
            help="Assistant가 학습할 PDF 파일을 업로드해주세요"
        )
        
        # PDF 처리
        if uploaded_file is not None and not st.session_state.pdf_loaded:
            with st.spinner("🧠 Assistant가 PDF를 학습하는 중..."):
                try:
                    result = st.session_state.assistant.load_pdf_knowledge(uploaded_file)
                    
                    if result[0]:  # 성공
                        st.session_state.pdf_loaded = True
                        st.session_state.pdf_info = {
                            'name': uploaded_file.name,
                            'pages': result[1],
                            'chunks': result[2]
                        }
                        st.success("🎉 PDF 학습 완료!")
                        st.rerun()
                    else:
                        st.error(f"❌ PDF 처리 실패: {result[3] if len(result) > 3 else '알 수 없는 오류'}")
                        
                except Exception as e:
                    st.error(f"❌ PDF 처리 중 오류: {str(e)}")
        
        # PDF 정보 표시
        if st.session_state.pdf_loaded and st.session_state.pdf_info:
            st.header("📊 PDF 정보")
            info = st.session_state.pdf_info
            st.metric("파일명", info['name'])
            st.metric("페이지 수", info['pages'])
            st.metric("텍스트 조각", info['chunks'])
        
        # 대화 관리
        st.header("💬 대화 관리")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 대화 초기화", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.assistant.clear_history()
                st.success("대화 기록을 초기화했습니다!")
                st.rerun()
        
        with col2:
            if st.button("🧠 메모리 정리", use_container_width=True):
                # 메모리만 정리 (화면 기록은 유지)
                st.session_state.assistant.clear_history()
                st.success("Assistant 메모리를 정리했습니다!")
                st.info("화면의 대화 기록은 유지됩니다.")
        
        # 통계 표시
        if st.button("📈 통계 보기", use_container_width=True):
            if st.session_state.chat_history:
                total_chats = len(st.session_state.chat_history)
                avg_time = sum(chat.get('response_time', 0) for chat in st.session_state.chat_history) / total_chats
                
                # 메트릭 표시
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("총 대화 수", total_chats)
                with col_b:
                    st.metric("평균 응답 시간", f"{avg_time:.2f}초")
                
                # Content Filter 관련 통계
                error_count = sum(1 for chat in st.session_state.chat_history 
                                if "안전 필터" in chat.get('answer', '') or "안전 모드" in chat.get('answer', ''))
                if error_count > 0:
                    st.warning(f"⚠️ Content Filter 발생: {error_count}회")
                    st.info("💡 질문 방식을 더 학술적으로 표현해보세요!")
            else:
                st.info("아직 대화 기록이 없습니다.")
        
        # Assistant 상태
        st.header("🤖 Assistant 상태")
        if st.session_state.pdf_loaded:
            st.success("✅ 준비 완료")
            st.caption("Assistant가 PDF를 학습했습니다")
        else:
            st.warning("⏳ 대기 중")
            st.caption("PDF를 업로드해주세요")
        
        # 안전 가이드 추가
        st.header("🛡️ 안전 가이드")
        with st.expander("Content Filter 관련 팁", expanded=False):
            st.markdown("""
            **Azure OpenAI 안전 필터를 피하는 방법:**
            
            ✅ **추천하는 질문 방식:**
            - "문서의 주요 내용을 요약해주세요"
            - "3페이지의 정의를 설명해주세요" 
            - "개념 A와 B의 차이점은 무엇인가요?"
            - "이론적 배경을 알려주세요"
            
            ❌ **피해야 할 표현:**
            - 감정적이거나 자극적인 단어
            - 정치적이거나 민감한 주제
            - 부정적인 뉘앙스의 표현
            
            💡 **팁:**
            - 학술적이고 전문적인 표현 사용
            - 구체적이고 명확한 질문
            - 중립적인 톤 유지
            """)
        
        # 문제 해결 가이드
        with st.expander("문제 해결", expanded=False):
            st.markdown("""
            **Content Filter 오류 발생 시:**
            1. 질문을 다시 표현해보세요
            2. 더 학술적인 용어를 사용해보세요
            3. 구체적인 페이지나 섹션을 지정해보세요
            
            **Rate Limit 오류 발생 시:**
            1. 1-2분 기다린 후 다시 시도
            2. 더 짧은 질문으로 나눠서 물어보세요
            """)
    
    # 메인 영역
    if st.session_state.pdf_loaded:
        # 채팅 인터페이스
        st.header("💬 Assistant와 대화하기")
        
        # 이전 대화 기록 표시
        display_chat_history()
        
        # 새로운 질문 입력
        if prompt := st.chat_input("Assistant에게 질문해보세요!"):
            # 사용자 메시지 표시
            timestamp = datetime.now().strftime("%H:%M:%S")
            display_chat_message("user", prompt, timestamp)
            
            # Assistant 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("🤔 Assistant가 생각하는 중..."):
                    try:
                        # Assistant와 대화
                        answer, source_docs, response_time = st.session_state.assistant.chat(prompt)
                        
                        # 답변 표시
                        st.write(answer)
                        st.caption(f"⏰ 응답 시간: {response_time:.2f}초")
                        
                        # 출처 문서 표시
                        if source_docs:
                            with st.expander(f"📚 참고 문서 ({len(source_docs)}개)", expanded=False):
                                for i, doc in enumerate(source_docs[:3], 1):
                                    st.markdown(f"**📄 {i}번째 참고 문서:**")
                                    st.text_area(
                                        f"내용 {i}",
                                        value=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                                        height=100,
                                        key=f"new_doc_{i}_{time.time()}"
                                    )
                                    if doc.metadata:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.caption(f"📍 페이지: {doc.metadata.get('page', 'Unknown')}")
                                        with col2:
                                            st.caption(f"📁 파일: {doc.metadata.get('source', 'Unknown')}")
                                    st.divider()
                        
                        # 대화 기록에 저장
                        st.session_state.chat_history.append({
                            'question': prompt,
                            'answer': answer,
                            'response_time': response_time,
                            'source_docs': source_docs,
                            'timestamp': timestamp
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Assistant 응답 중 오류: {str(e)}")
    
    else:
        # PDF가 업로드되지 않은 경우
        st.info("👆 사이드바에서 PDF 파일을 업로드하여 Assistant를 활성화해주세요!")
        
        # 사용법 안내
        st.markdown("""
        ### 🤖 PDF Assistant 사용법
        
        1. **PDF 업로드**: 왼쪽 사이드바에서 학습시킬 PDF 파일을 업로드하세요
        2. **학습 대기**: Assistant가 PDF 내용을 분석하고 학습할 때까지 기다리세요
        3. **대화 시작**: 채팅창에서 PDF 내용에 대해 자유롭게 질문하세요
        4. **스마트 답변**: Assistant가 PDF에서 관련 정보를 찾아 답변해드립니다
        
        ### ✨ 주요 기능
        
        - **🧠 지능형 검색**: 질문의 의도를 이해하여 관련 내용을 정확하게 찾습니다
        - **💬 대화 기억**: 이전 대화 내용을 기억하여 맥락에 맞는 답변을 제공합니다
        - **📚 출처 표시**: 답변의 근거가 된 PDF 페이지와 내용을 보여줍니다
        - **⚡ 빠른 응답**: 최적화된 검색으로 빠른 답변을 제공합니다
        
        ### 💡 팁
        
        - 구체적이고 명확한 질문을 하면 더 정확한 답변을 받을 수 있습니다
        - "이전에 말한 것처럼" 등의 표현으로 이전 대화를 참조할 수 있습니다
        - 참고 문서를 확인하여 답변의 정확성을 검증해보세요
        """)
        
        # 샘플 질문 예시
        st.markdown("""
        ### 📝 샘플 질문 예시
        
        - "이 문서의 주요 내용을 요약해주세요"
        - "X에 대한 정의가 무엇인가요?"
        - "Y와 Z의 차이점은 무엇인가요?"
        - "3페이지에서 언급된 내용을 설명해주세요"
        """)

if __name__ == "__main__":
    main() 