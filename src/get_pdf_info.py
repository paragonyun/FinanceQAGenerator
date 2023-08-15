import PyPDF2
from glob import glob
import os

PDF_FOLDER = r"C:\Users\JeongSeongYun\Desktop\ChatPDF\hanaproject\example_pdf"

def get_pdf_texts(pdf_folder=PDF_FOLDER):
    # 폴더 내의 모든 PDF 파일 경로 가져오기
    pdf_files = glob(os.path.join(pdf_folder, '*.pdf'))

    # 파일(상품)이름:text 형식의 딕셔너리 저장
    pdf_text_dict = {}

    for pdf_file in pdf_files:
        # PDF 파일 오픈
        pdf_file_obj = open(pdf_file, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        num_pages = len(pdf_reader.pages)
        detected_text = ''

        # 각 페이지에서 텍스트 추출
        for page_num in range(num_pages):
            page_obj = pdf_reader.pages[page_num]
            detected_text += page_obj.extract_text() + '\n\n'

        # 파일 이름에서 확장자 제거하고 딕셔너리에 저장
        filename_without_ext = os.path.splitext(os.path.basename(pdf_file))[0]
        pdf_text_dict[filename_without_ext] = detected_text

        # PDF 파일 닫기
        pdf_file_obj.close()

    return pdf_text_dict

if __name__ == "__main__":
    pdf_texts = get_pdf_texts(PDF_FOLDER)
    for filename, text in pdf_texts.items():
        print(f"📃 {filename}의 내용 📃")
        print(text)
        print("======================================")
