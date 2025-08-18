# 📚 똑똑한 PDF 검색 프로그램
# 이 프로그램은 PDF 파일을 읽고, AI가 도움을 줘서 내용을 찾아주는 프로그램입니다!

# 필요한 도구들을 가져옵니다 (마치 연필, 지우개를 준비하는 것처럼!)
import os          # 컴퓨터 파일을 다루는 도구
import time        # 시간을 재는 도구 (스톱워치 같은 것)
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI        # AI가 글을 이해하고 대화하는 도구
from langchain_community.document_loaders import PyPDFLoader  # PDF를 읽는 도구
from langchain_text_splitters import CharacterTextSplitter    # 긴 글을 작은 조각으로 나누는 도구
from langchain_community.vectorstores import FAISS         # 메모리에서 검색할 수 있게 저장하는 도구
from langchain.chains import RetrievalQA                   # 검색과 질답을 연결하는 도구
from langchain.memory import ConversationBufferMemory      # 대화 기록을 기억하는 도구
from langchain_core.messages import HumanMessage, AIMessage       # 사람과 AI 메시지 형식
from dotenv import load_dotenv                           # 비밀번호 같은 설정을 읽는 도구

# 🔐 비밀 설정들을 읽어옵니다 (부모님이 숨겨둔 설정 파일)
load_dotenv()

def main():
    # 🚀 프로그램이 시작됩니다!
    try:
        # 📖 1단계: PDF 파일을 찾아서 읽기
        pdf_path = "testpdf.pdf"  # 읽을 PDF 파일 이름
        
        # 파일이 있는지 확인하기 (책이 책상에 있는지 확인하는 것처럼!)
        if not os.path.exists(pdf_path):
            print(f"앗! PDF 파일 '{pdf_path}'를 찾을 수 없어요!")
            print("💡 먼저 'python create_test_pdf.py'를 실행해서 테스트 파일을 만들어주세요!")
            return  # 파일이 없으면 프로그램 끝내기
            
        print(f">> PDF 파일을 읽는 중: {pdf_path}")
        loader = PyPDFLoader(pdf_path)  # PDF 읽는 도구 준비
        pages = loader.load()           # PDF의 모든 페이지 읽기
        print(f">> 총 {len(pages)}페이지를 읽었어요!")

        # ✂️ 2단계: 긴 글을 작은 조각으로 나누기 (큰 피자를 작은 조각으로 자르는 것처럼!)
        text_splitter = CharacterTextSplitter(
            separator="\n\n",      # 문단이 바뀌는 곳에서 나누기
            chunk_size=10000,       # 한 조각은 1000글자 정도로
            chunk_overlap=2000,     # 조각들이 200글자씩 겹치게 (내용이 끊어지지 않게!)
            length_function=len,   # 글자 수 세는 방법
            is_separator_regex=False,  # 복잡한 규칙 사용 안 함
        )
        texts = text_splitter.split_documents(pages)  # 실제로 나누기 실행!
        print(f">> 총 {len(texts)}개의 작은 조각으로 나누었어요!")
        
        # 📊 Rate limit을 피하기 위해 텍스트 조각 수 제한
        max_chunks = 50  # 최대 50개 조각만 사용 (API 호출 제한 때문에)
        if len(texts) > max_chunks:
            print(f"⚠️  텍스트 조각이 너무 많아요! {len(texts)}개 → {max_chunks}개로 제한합니다.")
            texts = texts[:max_chunks]
                
        # 🧠 3단계: AI가 글을 이해할 수 있게 도와주는 도구 준비하기
        print(">> AI가 글을 이해할 수 있게 준비 중...")
        embeddings_model = AzureOpenAIEmbeddings(
            # 마이크로소프트 AI 서비스 설정들 (어른들이 설정해둔 것)
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        # 🤖 4단계: Chat용 AI 모델 준비하기 (대화할 수 있는 똑똑한 AI!)
        print(">> 대화용 AI 모델 준비 중...")
        chat_model = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),  # GPT 모델용 배포 이름
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.1,  # 답변이 일정하게 나오도록 (0에 가까울수록 일정함)
        )

        # 🗃️ 5단계: 메모리에 검색 가능한 데이터베이스 만들기 (빠르게 검색할 수 있어요!)
        print(">> 메모리에 검색 데이터베이스를 만드는 중... (조금 기다려 주세요!)")
        print(f"   📊 처리할 텍스트 조각: {len(texts)}개")
        print("   ⏰ Rate limit 때문에 천천히 처리합니다...")
        
        # Rate limit을 피하기 위해 배치로 처리
        max_retries = 3
        for attempt in range(max_retries):
            try:
                vectorstore = FAISS.from_documents(
                    documents=texts,              # 나눈 글 조각들을
                    embedding=embeddings_model   # AI가 이해할 수 있게 변환해서 메모리에 저장
                )
                print(f">> 짱! {len(texts)}개의 글 조각으로 메모리 검색 데이터베이스를 만들었어요!")
                break  # 성공하면 루프 종료
            except Exception as embed_error:
                if "429" in str(embed_error) or "rate limit" in str(embed_error).lower():
                    if attempt < max_retries - 1:
                        wait_time = 60 * (attempt + 1)  # 1분, 2분, 3분 대기
                        print(f"   ⏳ Rate limit 오류! {wait_time}초 후 재시도... ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"   ❌ {max_retries}번 시도했지만 여전히 Rate limit 오류입니다.")
                        print("   💡 해결방법:")
                        print("   1. 60분 후 다시 시도")
                        print("   2. 더 작은 PDF 파일 사용")
                        print("   3. Azure OpenAI 요금 계층 업그레이드")
                        raise embed_error
                else:
                    raise embed_error

        # 🔗 6단계: 검색과 질답을 연결하는 체인 만들기
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # 가장 비슷한 5개 문서 가져오기
        )
        
        # 💬 대화 기록을 기억할 수 있는 메모리 만들기
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # 🤝 검색과 대화를 연결하는 체인 만들기
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # 🔍 7단계: 이제 Chat 형태로 질문답변을 시작합니다!
        print("\n" + "="*50)
        print("🎉 똑똑한 PDF Chat AI 완성! 이제 대화할 수 있어요!")
        print("❌ 끝내고 싶으면: 'quit' 또는 'exit' 입력")
        print("📜 대화 기록 보기: 'history' 입력")
        print("🔄 대화 기록 초기화: 'clear' 입력")
        print("="*50)
        
        chat_history = []  # 대화 기록을 저장할 리스트
        
        # 💬 Chat 루프: 사용자와 AI가 계속 대화할 수 있게 하기
        while True:  # 무한 반복 (사용자가 그만하겠다고 할 때까지)
            try:
                # 사용자에게 질문 물어보기
                user_input = input("\n� AI에게 질문해보세요: ").strip()
                
                # 사용자가 나가고 싶어하는지 확인하기
                if user_input.lower() in ['quit', 'exit', '종료', '나가기']:
                    # 지금까지 한 대화들을 요약해서 보여주기
                    if chat_history:
                        print(f"\n📊 오늘 대화 요약:")
                        print(f"   • 총 대화 횟수: {len(chat_history)}번")
                        avg_response_time = sum(chat['response_time'] for chat in chat_history) / len(chat_history)
                        print(f"   • 평균 응답 시간: {avg_response_time:.3f}초")
                    print("👋 대화를 마칠게요. 안녕!")
                    break  # 반복문에서 나가기
                
                # 대화 기록 보기
                if user_input.lower() == 'history':
                    if chat_history:
                        print("\n📜 대화 기록:")
                        print("-" * 50)
                        for i, chat in enumerate(chat_history, 1):
                            print(f"{i}. 👤 질문: {chat['question']}")
                            print(f"   🤖 답변: {chat['answer'][:100]}...")
                            print(f"   ⏰ 응답시간: {chat['response_time']:.3f}초")
                            print("-" * 30)
                    else:
                        print("📝 아직 대화 기록이 없어요!")
                    continue
                
                # 대화 기록 초기화
                if user_input.lower() == 'clear':
                    chat_history.clear()
                    memory.clear()
                    print("🔄 대화 기록을 초기화했어요!")
                    continue
                
                # 아무것도 입력하지 않았을 때
                if not user_input:
                    print("❓ 질문을 입력해주세요!")
                    continue  # 다시 질문 물어보기
                
                print(f"\n🤖 AI가 답변을 생각하는 중... 잠깐만 기다려주세요!")
                response_start_time = time.time()  # 응답 시작 시간 기록
                
                # 🤖 AI에게 질문하고 답변 받기
                result = qa_chain.invoke({"query": user_input})
             
                response_end_time = time.time()    # 응답 끝난 시간 기록
                response_time = response_end_time - response_start_time  # 총 걸린 시간 계산
                
                # 📋 AI 답변과 출처 문서 보여주기 (키 이름 확인하여 안전하게 접근)
                print(f"🔍 디버그: 결과 키들 = {list(result.keys())}")  # 디버그용
                
                # result 딕셔너리에서 적절한 키 찾기
                if "result" in result:
                    answer = result["result"]
                elif "answer" in result:
                    answer = result["answer"]
                else:
                    # 첫 번째 값을 답변으로 사용하거나 문자열로 변환
                    answer = str(result.get(list(result.keys())[0], "답변을 찾을 수 없습니다."))
                
                source_docs = result.get("source_documents", [])
                
                print(f"\n🤖 AI 답변:")
                print("="*50)
                print(answer)
                print("="*50)
                
                # 📚 참고한 문서들 보여주기
                if source_docs:
                    print(f"\n📚 참고한 문서들 ({len(source_docs)}개):")
                    for i, doc in enumerate(source_docs, 1):
                        print(f"\n📄 {i}번째 참고 문서:")
                        print(f"    내용: {doc.page_content[:200]}...")
                        if doc.metadata:
                            page_info = doc.metadata.get('page', 'Unknown')
                            source = doc.metadata.get('source', 'Unknown')
                            print(f"   📍 위치: {page_info}페이지 | 파일: {source}")
                
                #  응답 성능 보여주기
                print(f"\n⏰ 응답 시간: {response_time:.3f}초")
                

                # 📝 이번 대화 기록 저장하기
                chat_history.append({
                    'question': user_input,
                    'answer': answer,
                    'response_time': response_time,
                    'source_count': len(source_docs)
                })
                
                # 💭 메모리에도 대화 추가 (수동으로)
                memory.chat_memory.add_user_message(user_input)
                memory.chat_memory.add_ai_message(answer)
                        
            except KeyboardInterrupt:  # Ctrl+C를 눌렀을 때
                print("\n\n👋 대화를 중단합니다. 안녕!")
                break
            except Exception as chat_error:  # 뭔가 잘못됐을 때
                print(f"😓 대화하다가 문제가 생겼어요: {chat_error}")
                print("💡 다시 시도해보세요!")
            
    except Exception as e:  # 프로그램 전체에 문제가 생겼을 때
        print(f"🚨 프로그램에 문제가 생겼어요: {str(e)}")
        print("🔧 어른에게 도움을 요청하세요!")

if __name__ == "__main__":
    print("🚀 똑똑한 PDF Chat AI 프로그램을 시작합니다!")
    print("📖 PDF 파일을 읽고 AI와 대화하면서 원하는 내용을 찾아드려요!")
    main()  # 메인 함수 실행하기
    print("✨ 프로그램이 끝났어요. 수고하셨습니다!") 
