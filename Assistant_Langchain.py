# 🤖 LangChain + Azure OpenAI로 Assistant 기능 구현하기
# Assistant API가 아닌 일반 Chat API로 Assistant와 유사한 기능 구현

import os
import time
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

class PDFAssistant:
    """PDF 검색 전용 AI 어시스턴트"""
    
    def __init__(self):
        print("🤖 PDF Assistant를 초기화하는 중...")
        
        # Azure OpenAI 설정
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7
        )
        
        # 임베딩 모델 설정
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        
        # 대화 기억 장치 (Assistant의 상태 유지 기능)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # 벡터 저장소와 대화 체인 (나중에 초기화)
        self.vectorstore = None
        self.conversation_chain = None
        
        print("✅ PDF Assistant 준비 완료!")
    
    def load_pdf_knowledge(self, pdf_path="testpdf.pdf"):
        """PDF를 읽어서 Assistant의 지식으로 만들기"""
        
        print(f"📖 PDF 파일 로딩 중: {pdf_path}")
        
        # 파일 존재 확인
        if not os.path.exists(pdf_path):
            print(f"❌ PDF 파일 '{pdf_path}'를 찾을 수 없습니다!")
            return False
        
        try:
            # 1. PDF 로드
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            print(f"📄 총 {len(pages)}페이지 로드 완료")
            
            # 2. 텍스트 분할
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_documents(pages)
            print(f"✂️ {len(texts)}개 텍스트 조각으로 분할 완료")
            
            # 3. 벡터 저장소 생성
            print("🧠 AI가 이해할 수 있도록 변환 중...")
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            # 4. 대화형 검색 체인 생성 (Assistant의 핵심 기능)
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                return_source_documents=True,
                chain_type="stuff"
            )
            
            print("✅ PDF 지식 로딩 완료! 이제 질문할 수 있습니다.")
            return True
            
        except Exception as e:
            print(f"❌ PDF 로딩 중 오류 발생: {e}")
            return False
    
    def chat(self, user_message):
        """사용자와 대화하기 (Assistant의 핵심 기능)"""
        
        if not self.conversation_chain:
            return "먼저 PDF를 로드해주세요! load_pdf_knowledge() 메서드를 사용하세요."
        
        try:
            print(f"🤔 사용자 질문: {user_message}")
            print("💭 답변을 생각하는 중...")
            
            # 대화형 검색 실행
            response = self.conversation_chain.invoke({"question": user_message})
            
            # 답변과 출처 정보 추출
            answer = response['answer']
            source_docs = response.get('source_documents', [])
            
            # 출처 정보 추가
            if source_docs:
                sources = []
                for i, doc in enumerate(source_docs[:2], 1):  # 상위 2개 출처만
                    page = doc.metadata.get('page', 'Unknown')
                    sources.append(f"페이지 {page + 1}")
                
                answer += f"\n\n📚 출처: {', '.join(sources)}"
            
            return answer
            
        except Exception as e:
            return f"❌ 답변 생성 중 오류 발생: {e}"
    
    def get_chat_history(self):
        """대화 히스토리 조회 (Assistant의 메모리 기능)"""
        return self.memory.chat_memory.messages
    
    def clear_history(self):
        """대화 히스토리 초기화"""
        self.memory.clear()
        print("🗑️ 대화 히스토리가 초기화되었습니다.")

def main():
    """메인 실행 함수"""
    
    print("🚀 PDF Assistant 데모 시작!")
    print("=" * 50)
    
    # 1. Assistant 생성
    assistant = PDFAssistant()
    
    # 2. PDF 지식 로드
    if not assistant.load_pdf_knowledge("testpdf.pdf"):
        print("PDF 로드에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    print("\n" + "=" * 50)
    print("💬 대화를 시작합니다! ('quit' 입력시 종료)")
    print("=" * 50)
    
    # 3. 대화 루프
    while True:
        try:
            user_input = input("\n🙋 질문: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("👋 대화를 종료합니다!")
                break
            
            if user_input.lower() == 'history':
                print("\n📜 대화 히스토리:")
                for msg in assistant.get_chat_history():
                    role = "사용자" if msg.type == "human" else "어시스턴트"
                    print(f"{role}: {msg.content[:100]}...")
                continue
            
            if user_input.lower() == 'clear':
                assistant.clear_history()
                continue
            
            if not user_input:
                print("질문을 입력해주세요!")
                continue
            
            # Assistant와 대화
            response = assistant.chat(user_input)
            print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다!")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
