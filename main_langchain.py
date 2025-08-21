# 📚 똑똑한 PDF 검색 프로그램
# 이 프로그램은 PDF 파일을 읽고, AI가 도움을 줘서 내용을 찾아주는 프로그램입니다!

# 필요한 도구들을 가져옵니다 (마치 연필, 지우개를 준비하는 것처럼!)
import os          # 컴퓨터 파일을 다루는 도구
import time        # 시간을 재는 도구 (스톱워치 같은 것)
from langchain_openai import AzureOpenAIEmbeddings        # AI가 글을 이해하는 도구
from langchain_community.document_loaders import PyPDFLoader  # PDF를 읽는 도구
from langchain_text_splitters import CharacterTextSplitter    # 긴 글을 작은 조각으로 나누는 도구
from langchain_chroma import Chroma                       # 검색할 수 있게 저장하는 도구
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
            return  # 파일이 없으면 프로그램 끝내기
            
        print(f">> PDF 파일을 읽는 중: {pdf_path}")
        loader = PyPDFLoader(pdf_path)  # PDF 읽는 도구 준비
        pages = loader.load()           # PDF의 모든 페이지 읽기
        print(f">> 총 {len(pages)}페이지를 읽었어요!")

        # ✂️ 2단계: 긴 글을 작은 조각으로 나누기 (큰 피자를 작은 조각으로 자르는 것처럼!)
        text_splitter = CharacterTextSplitter(
            separator="\n\n",      # 문단이 바뀌는 곳에서 나누기
            chunk_size=1000,       # 한 조각은 1000글자 정도로
            chunk_overlap=200,     # 조각들이 200글자씩 겹치게 (내용이 끊어지지 않게!)
            length_function=len,   # 글자 수 세는 방법
            is_separator_regex=False,  # 복잡한 규칙 사용 안 함
        )
        texts = text_splitter.split_documents(pages)  # 실제로 나누기 실행!
        print(f">> 총 {len(texts)}개의 작은 조각으로 나누었어요!")
                
        # 🧠 3단계: AI가 글을 이해할 수 있게 도와주는 도구 준비하기
        print(">> AI가 글을 이해할 수 있게 준비 중...")
        embeddings_model = AzureOpenAIEmbeddings(
            # 마이크로소프트 AI 서비스 설정들 (어른들이 설정해둔 것)
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        # 🗃️ 4단계: 검색 가능한 데이터베이스 만들기 (도서관 카드 목록 만들기 같은 것!)
        db_path = "./chroma_db"  # 데이터베이스를 저장할 폴더 이름
        
        # 이미 만들어진 데이터베이스가 있는지 확인하기
        if os.path.exists(db_path) and os.listdir(db_path):
            print(">> 이미 만들어진 데이터베이스를 찾았어요! 불러오는 중...")
            db = Chroma(
                persist_directory=db_path,        # 저장된 폴더에서
                embedding_function=embeddings_model  # AI 도구와 함께
            )
            print(f">> 기존 데이터베이스를 성공적으로 불러왔어요!")
        else:
            # 새로운 데이터베이스 만들기
            print(">> 새로운 검색 데이터베이스를 만드는 중... (조금 기다려 주세요!)")
            db = Chroma.from_documents(
                documents=texts,              # 나눈 글 조각들을
                embedding=embeddings_model,   # AI가 이해할 수 있게 변환해서
                persist_directory=db_path     # 컴퓨터에 저장하기
            )
            print(f">> 짠! {len(texts)}개의 글 조각으로 검색 데이터베이스를 만들었어요!")

        # 🔍 5단계: 이제 검색을 시작합니다!
        print("\n" + "="*50)
        print("🎉 검색 데이터베이스 완성! 이제 검색할 수 있어요!")
        print("❌ 끝내고 싶으면: 'quit' 또는 'exit' 입력")
        print("⚙️  더 자세한 검색: 'advanced' 입력")
        print("="*50)
        
        search_history = []  # 검색 기록을 저장할 빈 상자
        
        # 💬 검색 루프: 사용자가 검색어를 계속 입력할 수 있게 하기
        while True:  # 무한 반복 (사용자가 그만하겠다고 할 때까지)
            try:
                # 사용자에게 검색어 물어보기
                query = input("\n🔍 무엇을 찾고 싶나요? ").strip()
                
                # 사용자가 나가고 싶어하는지 확인하기
                if query.lower() in ['quit', 'exit', '종료', '나가기']:
                    # 지금까지 한 검색들을 요약해서 보여주기
                    if search_history:
                        print(f"\n📊 오늘 검색한 것들을 정리해드릴게요:")
                        print(f"   • 총 몇 번 검색했나요? {len(search_history)}번")
                        avg_time = sum(s['time'] for s in search_history) / len(search_history)
                        avg_results = sum(s['results_count'] for s in search_history) / len(search_history)
                        print(f"   • 평균 검색 시간: {avg_time:.3f}초")
                        print(f"   • 평균으로 몇 개씩 찾았나요? {avg_results:.1f}개")
                    print("👋 검색을 마칠게요. 안녕!")
                    break  # 반복문에서 나가기
                
                # 고급 검색 모드 (더 자세한 설정을 원할 때)
                if query.lower() == 'advanced':
                    print("🔧 고급 검색 모드입니다!")
                    k = int(input("몇 개의 결과를 보고 싶나요? (기본값 5개): ") or 5)
                    score_threshold = float(input("얼마나 비슷해야 할까요? (기본값 1.5, 낮을수록 더 비슷): ") or 1.5)
                    query = input("이제 검색어를 입력해주세요: ").strip()
                    if not query:  # 검색어를 안 쓰면 다시 처음으로
                        continue
                else:
                    # 일반 검색 모드 (기본 설정 사용)
                    k = 5              # 5개 결과 보여주기
                    score_threshold = 1.5  # 기본 유사도 기준
                
                # 아무것도 입력하지 않았을 때
                if not query:
                    print("❓ 검색어를 입력해주세요!")
                    continue  # 다시 검색어 물어보기
                
                print(f"\n🔍 '{query}' 찾는 중... 잠깐만 기다려주세요!")
                search_start_time = time.time()  # 검색 시작 시간 기록 (스톱워치 시작!)
                
                # 🤖 AI가 비슷한 내용 찾기 (마치 도서관에서 관련 책 찾는 것처럼!)
                similar_docs_with_scores = db.similarity_search_with_score(query, k=k)
                search_end_time = time.time()    # 검색 끝난 시간 기록 (스톱워치 끝!)
                search_time = search_end_time - search_start_time  # 총 걸린 시간 계산
                
                # 📊 너무 다른 내용은 제외하기 (설정한 기준보다 비슷한 것만 남기기)
                filtered_results = [(doc, score) for doc, score in similar_docs_with_scores if score < score_threshold]
                
                if filtered_results:  # 찾은 결과가 있다면
                    # 📈 성능을 측정해보기 (얼마나 잘 찾았는지 점수 매기기)
                    scores = [score for _, score in filtered_results]  # 모든 점수 모으기
                    avg_score = sum(scores) / len(scores)  # 평균 점수 계산
                    max_score = max(scores)  # 가장 높은 점수
                    min_score = min(scores)  # 가장 낮은 점수 (가장 비슷한 것)
                    
                    # 🌈 다양성 측정 (서로 다른 페이지에서 찾았는지 확인)
                    unique_pages = len(set(doc.metadata.get('page', 0) for doc, _ in filtered_results))
                    diversity = unique_pages / len(filtered_results) if filtered_results else 0
                    
                    # 🎯 정밀도와 재현율 계산 (얼마나 정확하게 찾았는지 측정)
                    good_results = [(doc, score) for doc, score in filtered_results if score < 1.0]  # 정말 좋은 결과들
                    precision = len(good_results) / len(filtered_results) if filtered_results else 0  # 정밀도
                    
                    # 재현율 추정 (전체 중에서 얼마나 많이 찾았는지 - 대략적으로 계산)
                    total_chunks = len(texts) if 'texts' in locals() else 100
                    recall_estimate = len(good_results) / min(total_chunks, 20)
                    
                    # 🎊 검색 결과 성적표 보여주기
                    print(f"📊 검색 성적표:")
                    print(f"   ⏰ 검색 시간: {search_time:.3f}초 (빨랐나요?)")
                    print(f"   📌 걸러진 결과: {len(filtered_results)}개 (전체 {len(similar_docs_with_scores)}개 중)")
                    print(f"   📏 평균 비슷함 정도: {avg_score:.4f} (낮을수록 더 비슷해요)")
                    print(f"   🥇 가장 비슷한 거리: {min_score:.4f}")
                    print(f"   🥉 가장 다른 거리: {max_score:.4f}")
                    print(f"   🎯 정밀도: {precision:.2%} (좋은 결과 {len(good_results)}개/{len(filtered_results)}개)")
                    print(f"   📊 추정 재현율: {recall_estimate:.2%}")
                    print(f"   🌈 다양성: {diversity:.2%} ({unique_pages}개 다른 페이지)")
                    
                    print(f"\n� 찾은 {len(filtered_results)}개 문서들:")
                    print("-" * 60)
                    
                    # 📋 각 결과를 자세히 보여주기
                    for i, (doc, score) in enumerate(filtered_results, 1):
                        # 🚦 얼마나 비슷한지 색깔로 표시하기
                        if score < 0.3:
                            similarity_level = "🟢 매우 비슷해요!"
                        elif score < 0.6:
                            similarity_level = "🟡 비슷해요"
                        elif score < 1.0:
                            similarity_level = "🟠 조금 비슷해요"
                        else:
                            similarity_level = "🔴 별로 안 비슷해요"
                        
                        # 백분율로 바꿔서 이해하기 쉽게 만들기
                        similarity_percent = max(0, (2 - score) / 2 * 100)
                        
                        print(f"\n📄 {i}번째 문서: {similarity_level}")
                        print(f"   📐 비슷함 거리: {score:.4f} (숫자가 작을수록 더 비슷해요)")
                        print(f"   📊 비슷함 백분율: {similarity_percent:.1f}%")
                        print(f"   📝 내용 미리보기: {doc.page_content[:250]}...")
                        
                        if doc.metadata:  # 문서 정보가 있다면
                            page_info = doc.metadata.get('page', 'Unknown')  # 몇 페이지인지
                            source = doc.metadata.get('source', 'Unknown')   # 어느 파일인지
                            print(f"   📍 위치: {page_info}페이지 | 파일: {source}")
                    
                    # 📝 이번 검색 기록 저장하기 (나중에 통계를 보여주기 위해)
                    search_history.append({
                        'query': query,              # 무엇을 검색했는지
                        'time': search_time,         # 얼마나 걸렸는지
                        'results_count': len(filtered_results),  # 몇 개 찾았는지
                        'avg_score': avg_score,      # 평균 점수
                        'precision': precision       # 정확도
                    })
                        
                else:  # 아무것도 못 찾았을 때
                    print("😅 아쉽게도 비슷한 내용을 찾을 수 없어요.")
                    if similar_docs_with_scores:  # 하지만 뭔가는 있었다면
                        best_score = min(score for _, score in similar_docs_with_scores)
                        print(f"   💡 힌트: 가장 비슷한 건 {best_score:.4f}점이에요 (기준: {score_threshold})")
                        print(f"   💭 'advanced' 모드에서 기준을 높이면 더 많이 찾을 수 있어요!")
                    
            except KeyboardInterrupt:  # Ctrl+C를 눌렀을 때
                print("\n\n👋 검색을 중단합니다. 안녕!")
                break
            except Exception as search_error:  # 뭔가 잘못됐을 때
                print(f"😓 검색하다가 문제가 생겼어요: {search_error}")
                print("💡 다시 시도해보세요!")
            
    except Exception as e:  # 프로그램 전체에 문제가 생겼을 때
        print(f"🚨 프로그램에 문제가 생겼어요: {str(e)}")
        print("🔧 어른에게 도움을 요청하세요!")

# 🎯 프로그램 실행 시작점 (여기서부터 모든 게 시작돼요!)
if __name__ == "__main__":
    print("🚀 똑똑한 PDF 검색 프로그램을 시작합니다!")
    print("📖 PDF 파일을 읽고 AI가 도움을 줘서 원하는 내용을 찾아드려요!")
    main()  # 메인 함수 실행하기
    print("✨ 프로그램이 끝났어요. 수고하셨습니다!")
    main() 
