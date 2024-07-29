import json
from datetime import datetime
from flask import Flask, request, jsonify, Response
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import GitbookLoader
import requests
from dotenv import load_dotenv
from huggingface_hub import login
from langchain.prompts import PromptTemplate
import pandas as pd
import re
import logging
from tenacity import retry, stop_after_attempt, wait_fixed

# 설정 초기화
app = Flask(__name__)

# 환경 변수 로드
load_dotenv()
# 로깅 설정
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

urls = [
    "https://kaistudio.gitbook.io/kai-studio", # 사용 가이드
    "https://kaistudio.gitbook.io/kai-studio/guide/signup", # 회원 가입하기
    "https://kaistudio.gitbook.io/kai-studio/guide/playground",# 플레이그라운드
    "https://kaistudio.gitbook.io/kai-studio/guide/playground/chat",# 플레이그라운드/대화하기
    "https://kaistudio.gitbook.io/kai-studio/guide/playground/completion",# 플레이그라운드/컴플리션하기
    "https://kaistudio.gitbook.io/kai-studio/guide/playground/undefined",# 플레이그라운드/모델 비교하기
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops",#LLMOps
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/learning",#LLMOps/학습하기
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/learning/finetuning-data",#LLMOps/학습하기/학습데이터 준비 및 업로드
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/learning/finetuning",#LLMOps/학습하기/모델 학습하기
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/deploy",#배포하기
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/search",#AI검색하기
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/search/finder",#AI검색하기/검색기 구성 및 검증
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/search/rag",#AI검색하기/AI 검색 연결하기
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/playground",#플레이그라운드
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/playground/chat",#플레이그라운드/대화하기
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/playground/completion",#플레이그라운드/컴플리션하기
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/api",#서비스 API이용하기
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/api/chat-api",#서비스 API이용하기/Chat API
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/api/chat-completion-api",#서비스 API이용하기/Chat Completion API
    "https://kaistudio.gitbook.io/kai-studio/guide/llmops/api/url-api",#서비스 API이용하기/문서 URL 생성 API
    "https://kaistudio.gitbook.io/kai-studio/guide/usecase",#활용사례
    "https://kaistudio.gitbook.io/kai-studio/manager-guide/undefined",#프로젝트생성하기
    "https://kaistudio.gitbook.io/kai-studio/manager-guide/info",#프로젝트관리하기
    "https://kaistudio.gitbook.io/kai-studio/manager-guide/members",#프로젝트 멤버추가/삭제하기
    "https://kaistudio.gitbook.io/kai-studio/manager-guide/api-key",#API키발급하기
    "https://kaistudio.gitbook.io/kai-studio/general/policy",#운영정책
    "https://kaistudio.gitbook.io/kai-studio/general/catalog",#모델카달로그
    "https://kaistudio.gitbook.io/kai-studio/general/catalog/kt-midm-bitext-s",#kt/midm-bitext-s
    "https://kaistudio.gitbook.io/kai-studio/general/catalog/kt-midm-bitext-s-chat",#kt/midm-bitext-s-chat
    "https://kaistudio.gitbook.io/kai-studio/general/catalog/kt-midm-bitext-e-chat-c08",#kt/midm-bitext-e-chat-c08
    "https://kaistudio.gitbook.io/kai-studio/general/catalog/microsoft-phi-3-medium-4k-instruct",#microsoft/phi-3-medium-4k-instruct
    "https://kaistudio.gitbook.io/kai-studio/general/catalog/openai-gpt-3.5-turbo",#openai/gpt-3.5-turbo
    "https://kaistudio.gitbook.io/kai-studio/general/catalog/meta-llama-3-8b-inst",#meta/llama-3-8b-inst
    "https://kaistudio.gitbook.io/kai-studio/general/catalog/meta-llama-3-70b-inst",#meta/llama-3-70b-inst
    "https://kaistudio.gitbook.io/kai-studio/general/catalog/alibaba-qwen2-7b-instruct",#alibaba/qwen2-7b-instruct

]
docs = [GitbookLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs_list)

# 벡터스토어를 생성
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 벡터스토어 로컬 저장 및 로드
vectorstore.save_local("faiss_index")
vectorstore_load = FAISS.load_local("faiss_index", embeddings=OpenAIEmbeddings(),allow_dangerous_deserialization=True)
# 정보를 검색하고 생성
retriever = vectorstore_load.as_retriever()


# 프롬프트 템플릿
prompt_template = """
# Your role :
당신은 질문자의 의도와 질문의 핵심을 이해하고, 주어진 문서를 통해 질문자의 요구에 가장 적합한 답변을 제공하는 뛰어난 QA 봇입니다.
# Instruction :
당신의 임무는 XML 태그로 구분된 아래의 Retrieved context를 내용을 근거로 질문에 대답하는 것입니다. Constraint를 준수하여 한국어로 대답하세요.
Constraint 내용을 답변으로 쓰지 마세요. 답변에 대한 평가를 하지 마세요. 검색한 문서 결과의 출처(url link) 리스트를 제공하세요.

<retrieved context>
# Retrieved Context:
{context}
</retrieved context>

# Constraint
1. 사용자의 질문의 의도를 이해하고 가장 적절한 답변을 제공해야 합니다.
- 질문의 맥락을 이해하고 질문자가 질문한 이유를 스스로 질문하고,이해한 내용을 바탕으로 적절한 응답을 제공합니다.
2. 검색된 컨텍스트에서 가장 관련성이 높은 콘텐츠(질문과 직접적으로 관련된 핵심 콘텐츠)를 선택하고 이를 사용하여 답변을 생성합니다.
3. 상세하고 논리적인 답변을 생성하세요. 답변을 생성할 때 선택 사항을 나열하는 것뿐만 아니라 문맥에 맞게 재배열하여 자연스러운 흐름의 단락이 되도록 하세요.
4. 사용자 질문에서 '검색 데이터'와 '학습 데이터'는 구분해서 설명하세요.  

# User's Question:
{question}

# Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)
# 모델 및 문서 로더 초기화
from openai import OpenAI

# 언어모델 생성(Create LLM)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    #return "\n\n".join(doc.page_content for doc in docs)
    return "\n\n".join(f"{doc.page_content} (출처: {doc.metadata['source']})" for doc in docs)

# 체인 생성(Create Chain)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 메시지 이벤트 처리
def handle_message(user_message):
    response_text = rag_chain.invoke(user_message)
    
    return response_text

# 팀즈로 payload 전송 실패 할 경우 2초 간격 3Times 시도
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def send_response(response_payload):
    response_json = json.dumps(response_payload, ensure_ascii=False)
    return Response(response=response_json, status=200, mimetype='application/json')

#질의응답 기록 엑셀 저장
def save_to_excel(timestamp, messageid, index, writer, user_message, response_text):
    # Define the filename based on the current date
    filename = f"QAmessages_{datetime.now().strftime('%Y-%m-%d')}.xlsx"

    # Create a DataFrame with the new data
    new_data = pd.DataFrame([{
        "Timestamp": timestamp,
        "MessageId" : messageid,
        "Index": index,
        "Writer": writer,
        "User Message": user_message,
        "Response Text": response_text
    }])

    try:
        # If the file exists, append the new data
        existing_data = pd.read_excel(filename)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        # If the file does not exist, create a new one with the new data
        combined_data = new_data

    # Save the combined data to the Excel file
    combined_data.to_excel(filename, index=False)

@app.route("/api/messages", methods=["POST"])
def messages():
    data = request.json
    user_message = data.get('text')
    user_message_re = re.sub('(<([^>]+)>|&nbsp;)', '', user_message)
    response_text = handle_message(user_message)
    timestamp = datetime.now().strftime('%Y%m%d - %X')
    messageid = data.get("id")
    writer = data.get("from", {}).get("name")
    index = data.get("conversation", {}).get("id")
    response_payload = {
        "type": "message",

        "conversation": {
            "id": data.get("conversation", {}).get("id"),
            "name": data.get("conversation", {}).get("name")
        },
        "recipient": {
            "id": data.get("from", {}).get("id"),
            "name": data.get("from", {}).get("name")
        },
        "text": response_text,
        "replyToId": data.get("id")
    }

    try:
        response = send_response(response_payload)
        save_to_excel(timestamp, messageid, index, writer, user_message_re, response_text)
        logging.info(f"Success: Response sent for message ID {data.get('id')}")
        return response
        
    except Exception as e:
        logging.error(f"Error: Failed to process message ID {data.get('id')}: {e}")
        return jsonify({"status": "failed", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=3968)

