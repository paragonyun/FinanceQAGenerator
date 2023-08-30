from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

import os
import openai

from src.get_pdf_info import get_pdf_texts

openai.api_key = os.getenv("OPENAI_API_KEY")

template = """
            당신은 하나은행의 상품을 잘 이해하고 있는 금융 선생님입니다. 학생의 수준에 맞게 문제를 내고 해설을 잘 내주는 것으로 유명합니다.

            -주어진 하나은행의 금융 상품 설명서를 보고 주어진 학생의 수준에 맞는 문제와 정답, 그리고 해설을 제공해주세요. 
            -해설은 상품 설명서의 어느 부분을 근거로 삼았는지에 대한 내용도 포함해주세요.
            -문제를 제출할 때, 주어진 금융 상품 설명서의 내용을 필수로 참고하세요.
            -단, 보기는 5개이고 그 중 정답은 1개 입니다.
            -문제는 상품에 대한 이해를 도울 수 있는 문제여야 합니다.
            -상품에 대한 정확한 이해가 가능해야 합니다.
            -각 문제는 서로 다른 것을 물어봐야합니다. 예를 들어, 1번문제엔 금리 관련을, 2번 문제엔 가입조건을, 3번문제엔 예금한도와 같이 서로 다른 주제에 대해 물어봐야 합니다.
            -위의 상품 설명에 대한 이해를 도울 수 있는 객관식 문제 5개를 내주세요. 
            -단, 30대이고 금융 중수 수준에 맞게 난이도를 내주고, 상품에 대한 정확한 이해가 가능해야합니다. 
            -그 무엇보다, 상품에 대해 잘못된 지식이 전달되지 않도록 주의해주세요.

            다음의 형식에 따라 문제와 보기, 해설을 제공해주세요.
            📃문제1: 1번 문제에 대한 내용
            🤔보기:
            1번: 1번 보기
            2번: 2번 보기
            3번: 3번 보기
            4번: 4번 보기
            5번: 5번 보기
            ✅정답: 정답 번호
            🚩해설: 상품설명서 내용 기반 해설

            📃문제2: 2번 문제에 대한 내용
            🤔보기:
            1번: 1번 보기
            2번: 2번 보기
            3번: 3번 보기
            4번: 4번 보기
            5번: 5번 보기
            ✅정답: 정답 번호
            🚩해설: 상품설명서 내용 기반 해설

            위 형식 반복으로 5문제 출제.
            """
query = """
        위의 상품 설명에 대한 이해를 도울 수 있는 객관식 문제 5개를 내주세요. 
        단, 30대이고 금융 중수 수준에 맞게 난이도를 내주고, 상품에 대한 정확한 이해가 가능해야합니다. 
        그 무엇보다, 상품에 대해 잘못된 지식이 전달되지 않도록 주의해주세요.
        """

docs = get_pdf_texts()

print("상품 이해도를 확인하고자 하는 손님의 상품을 선택해주세요.")
print(f"상품 리스트: {docs.keys()}")

product = str(input())

extracted_texts = str(docs[product])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents([extracted_texts])

directory = 'index_store'
vector_index = FAISS.from_documents(texts, OpenAIEmbeddings())
vector_index.save_local(directory)

vector_index = FAISS.load_local('index_store', OpenAIEmbeddings())
retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k":6})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

print("✅ Entering to Model...")
qa_interface = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="map_reduce", 
                                        retriever=retriever, 
                                        return_source_documents=True)
print("🤖 Making..")
result = qa_interface(template)
print("✅ Done!")

print(result["result"])


