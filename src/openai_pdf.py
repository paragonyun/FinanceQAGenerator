import json
from datetime import datetime
import os

import openai

from src.get_pdf_info import get_pdf_texts

"""
여러개가 존재 하는 경우, 손님이 이해를 확인하고자 하는 상품을 선택하고, 
이를 기반으로 5개를 생성해줘야할듯?
"""

openai.api_key = os.getenv("OPENAI_API_KEY")

texts = get_pdf_texts()

print("상품 이해도를 확인하고자 하는 손님의 상품을 선택해주세요.")
print(f"상품 리스트: {texts.keys()}")

product = str(input())

extracted_texts = str(texts[product])

"""
ChatGPT에게 원하는 역할을 세부적으로 지정하여 
원하는 쿼리를 날립니다.
"""

def get_qa(extracted_texts, age=20, level='초보'):
    global system_msg, query, user_msg
    system_msg = """
                당신은 하나은행의 상품을 잘 이해하고 있는 금융 선생님입니다. 학생의 수준에 맞게 문제를 내고 해설을 잘 내주는 것으로 유명합니다.

                하나은행의 금융 상품 설명서를 보고 주어진 학생의 수준에 맞는 문제와 정답, 그리고 해설을 제공해주세요. 
                해설은 상품 설명서의 어느 부분을 근거로 삼았는지에 대한 내용도 포함해주세요.
                문제를 제출할 때, 주어진 금융 상품 설명서의 내용을 필수로 참고하세요.
                단, 보기는 5개이고 그 중 정답은 1개 입니다.
                문제는 상품에 대한 이해를 도울 수 있는 문제여야 합니다.
                상품에 대한 정확한 이해가 가능해야 합니다.
                각 문제는 서로 다른 것을 물어봐야합니다. 예를 들어, 1번문제엔 금리 관련을, 2번 문제엔 가입조건을, 3번문제엔 예금한도와 같이 서로 다른 주제에 대해 물어봐야 합니다.
                

                다음의 형식에 따라 문제와 보기, 해설을 제공해주세요.
                📃문제1: {1번 문제에 대한 내용}
                🤔보기:
                1번: {1번 보기}
                2번: {2번 보기}
                3번: {3번 보기}
                4번: {4번 보기}
                5번: {5번 보기}
                ✅정답: {정답 번호}
                🚩해설: {상품설명서 내용 기반 해설}

                📃문제2: {2번 문제에 대한 내용}
                🤔보기:
                1번: {1번 보기}
                2번: {2번 보기}
                3번: {3번 보기}
                4번: {4번 보기}
                5번: {5번 보기}
                ✅정답: {정답 번호}
                🚩해설: {상품설명서 내용 기반 해설}

                위 형식 반복으로 5문제 출제.
                """

    query = f"""
            위의 상품 설명에 대한 이해를 도울 수 있는 객관식 문제 5개를 내주세요. 
            단, {age}대이고 금융 {level} 수준에 맞게 난이도를 내주고, 상품에 대한 정확한 이해가 가능해야합니다. 
            그 무엇보다, 상품에 대해 잘못된 지식이 전달되지 않도록 주의해주세요.
            """



    user_msg = extracted_texts + "\n\n" + query

    print("🤖 Creating Questions... 🤖")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k", #TODO: 어떤 모델이 적합할지 테스트 해보기 -> 랭체인으로 줄이는 방법도.. 고려하기..!
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    print(response.choices[0].message.content)

    return response.choices[0].message.content

def get_result_json(QA_output):
    result={}
    result['products'] = []
    for product in texts.keys():
        result['products'].append({

            'fin_product_name': product,
            'system_msg': system_msg,
            'query': query,
            'QA': QA_output

        })

    now =  datetime.now()
    log_time = now.strftime("%y%m%d_%H%M")  # Format as "yymmdd_HHMM"

    result_log_file_name = f"{log_time}.json"

    with open(f"C:/Users/JeongSeongYun/Desktop/ChatPDF/hanaproject/outputs/{result_log_file_name}", "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    texts = get_pdf_texts()                             # 상품설명서로부터 text 추출
    extracted_texts = str(texts.values())               # 내용만 추출
    QA_result = get_qa(extracted_texts=extracted_texts, # 상품설명서 내용
                        age=30,                         # 사용자 연령
                        level='중수')                   # 사용자 금융 지식 수준
    get_result_json(QA_output=QA_result)                # 결과 저장



## TODO: 짧은 PDF도 넣어서 테스트