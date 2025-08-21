from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

def create_small_test_pdf():
    """작은 테스트 PDF 파일 생성"""
    filename = "small_test.pdf"
    
    # PDF 문서 생성
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # 페이지 1
    c.setFont("Helvetica", 12)
    y_position = height - 100
    
    content = [
        "테스트 PDF 문서",
        "",
        "이 문서는 AI 테스트용으로 만들어진 작은 PDF 파일입니다.",
        "",
        "주요 내용:",
        "1. 인공지능(AI)은 인간의 지능을 모방하는 기술입니다.",
        "2. 머신러닝은 AI의 하위 분야로, 데이터로부터 학습합니다.",
        "3. 딥러닝은 신경망을 이용한 머신러닝 기법입니다.",
        "",
        "기술 분야:",
        "- 자연어 처리: 컴퓨터가 인간의 언어를 이해합니다.",
        "- 컴퓨터 비전: 이미지와 영상을 분석합니다.",
        "- 로봇공학: 자율적으로 작동하는 기계를 만듭니다.",
        "",
        "응용 분야:",
        "• 의료: 질병 진단과 치료에 도움을 줍니다.",
        "• 금융: 사기 탐지와 투자 분석을 수행합니다.",
        "• 교육: 개인 맞춤형 학습을 제공합니다.",
        "• 교통: 자율주행차 기술을 발전시킵니다.",
    ]
    
    for line in content:
        c.drawString(50, y_position, line)
        y_position -= 20
        
        # 페이지가 꽉 차면 새 페이지 시작
        if y_position < 100:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = height - 100
    
    # 페이지 2 추가
    if y_position > height - 200:  # 아직 공간이 많이 남아있으면 새 페이지
        c.showPage()
        c.setFont("Helvetica", 12)
        y_position = height - 100
    
    additional_content = [
        "추가 정보",
        "",
        "AI의 역사:",
        "- 1950년: 앨런 튜링이 튜링 테스트를 제안했습니다.",
        "- 1956년: 다트머스 회의에서 'AI'라는 용어가 처음 사용되었습니다.",
        "- 1980년대: 전문가 시스템이 상업적으로 성공했습니다.",
        "- 2010년대: 딥러닝 혁신으로 AI가 급속도로 발전했습니다.",
        "",
        "현재 AI 기술:",
        "• ChatGPT: 대화형 AI 모델",
        "• DALL-E: 이미지 생성 AI",
        "• AlphaGo: 바둑 게임 AI",
        "• GPT-4: 대규모 언어 모델",
        "",
        "미래 전망:",
        "AI 기술은 계속 발전하여 우리 삶의 모든 영역에서",
        "더욱 중요한 역할을 할 것으로 예상됩니다.",
    ]
    
    for line in additional_content:
        c.drawString(50, y_position, line)
        y_position -= 20
    
    # PDF 저장
    c.save()
    
    # 파일 크기 확인
    file_size = os.path.getsize(filename)
    print(f"✅ 작은 테스트 PDF 생성 완료!")
    print(f"   📄 파일명: {filename}")
    print(f"   📊 크기: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"   💡 원본 testpdf.pdf 대신 이 파일을 사용하세요!")

if __name__ == "__main__":
    try:
        create_small_test_pdf()
    except ImportError:
        print("❌ reportlab 라이브러리가 필요합니다!")
        print("📦 설치: pip install reportlab")
    except Exception as e:
        print(f"❌ PDF 생성 오류: {e}") 