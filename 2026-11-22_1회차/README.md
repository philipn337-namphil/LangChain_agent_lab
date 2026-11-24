# 2026-11-22 · 1회차 실습 정리

러닝스푼즈 「생성형 AI와 LLM」 1회차 실습 노트북 6개를 정리했습니다.
랭체인 기초에서 시작해 RAG 챗봇, 과제(멀티턴·멀티쿼리·도구 선택), 에이전트 3종(Plan & Execute, ReACT, Function Calling)까지 다룹니다.

> **API 키 안내** — 노트북에 있던 키는 모두 지웠습니다. 실행하려면 Colab 왼쪽 🔑(보안 비밀)에 `OPENAI_API_KEY`를 등록하세요.
> 코드는 `userdata.get("OPENAI_API_KEY")`로 키를 읽습니다.

## 한눈에 보기

| # | 노트북 | 핵심 한 줄 |
|---|---|---|
| 01 | [랭체인 시작하기](notebooks/01_랭체인_시작하기.ipynb) | ChatOpenAI → PromptTemplate → 텍스트 분할 → 임베딩 → Chroma/FAISS 검색까지, 랭체인 기본 부품을 하나씩 사용 |
| 02 | [금융 PDF 챗봇](notebooks/02_랭체인_금융_PDF_챗봇.ipynb) | 한국은행 「경제금융용어 700선」 PDF로 RAG 챗봇을 만들고 Gradio로 띄움 |
| 03 | [과제: 멀티턴·멀티쿼리·도구 선택](notebooks/03_과제_멀티턴_멀티쿼리_도구선택.ipynb) | 프롬프트만으로 검색어 재구성, 질문 쪼개기, 도구 매칭을 만드는 빈칸 과제 |
| 04 | [Plan & Execute 에이전트](notebooks/04_Plan_and_Execute_에이전트.ipynb) | LLM이 먼저 `{도구: [검색어들]}` 계획을 JSON으로 세우고, 그 계획대로 검색한 뒤 답변 |
| 05 | [ReACT 에이전트](notebooks/05_ReACT_에이전트_RAG_멀티도구.ipynb) | Thought → Action → Observation 루프를 직접 구현해서, 필요한 만큼 검색을 반복 |
| 06 | [Function Calling](notebooks/06_Function_Calling_실습.ipynb) | OpenAI `tools` 명세로 모델이 호출할 함수와 인자를 고르게 하고, 쇼핑몰 CS 봇으로 확장 |

---

## 01. 랭체인 시작하기

**흐름**
1. `ChatOpenAI`로 LLM 호출, `role/content` 형식으로 이전 대화 넘기기
2. `PromptTemplate`로 `{who}`, `{time1}`/`{time2}` 빈칸이 있는 템플릿 만들기
3. `RecursiveCharacterTextSplitter`로 42MB 뉴스 파일 청킹 (`chunk_size`, `chunk_overlap` 비교)
4. 임베딩 → 코사인 유사도로 6개 문장 중 비슷한 문장 찾기
5. `Chroma` / `FAISS` 벡터 저장소에 넣고 `similarity_search`, `as_retriever(search_kwargs={"k": 1})`

**핵심 코드**
```python
prompt = PromptTemplate.from_template("{who}가 누구인지 설명해주세요")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chroma_db = Chroma.from_documents(texts, embeddings)
retriever = chroma_db.as_retriever(search_kwargs={"k": 1})
```

**결과**
| 실험 | 값 |
|---|---|
| 뉴스 파일 길이 | 18,085,369자 |
| `chunk_size=1000, overlap=0` | 25,281개 (1번 995자, 2번 700자) |
| `chunk_size=1000, overlap=50` | 25,416개 (2번 문서가 749자로 늘어남 = 앞 문서 끝 50자가 겹침) |
| 기사 스크립트 `chunk_size=600` | 22개 |
| "주인공은 어디서 태어났는가?" 검색 | Chroma와 FAISS 모두 "신분 상승의 꿈 / 온성군" 청크를 1등으로 찾음 |
| 꼬르륵 문장 유사도 Top-3 | 굶어서 허기가 지네요(0.838) → 연어구이(0.822) → 배가 고파요 |

**바꾼 부분 · 주의할 점**
- 11월 22일 실행에서 OpenAI 크레딧이 없어서 `ChatOpenAI` 셀이 `429 insufficient_quota`로 실패했습니다.
- 그 뒤 코드를 **무료 모델로 바꿨습니다.** LLM은 `Qwen/Qwen2.5-3B-Instruct`(HuggingFacePipeline), 임베딩은 `jhgan/ko-sroberta-multitask`이고, 키 셀은 주석 처리했습니다.
- ⚠️ 바꾼 뒤 **다시 실행하지 않아서**, 지금 보이는 출력은 OpenAI로 돌렸을 때의 것입니다. 예를 들어 "임베딩 길이 1536"은 OpenAI 기준이고, ko-sroberta는 768이 나와야 합니다.
- `prompt.format(location="서울")`은 일부러 `KeyError: 'who'`를 내는 셀입니다.

## 02. 금융 PDF 챗봇 (RAG)

**흐름**
1. 「2020 경제금융용어 700선」 PDF 다운로드 → `PyPDFLoader.load_and_split()`으로 366개 문서
2. 머리말·목차(0~12번)와 맺음말(마지막)을 지워서 **352개**만 남김
3. `OpenAIEmbeddings(chunk_size=100)` + `Chroma.from_documents` → `_collection.count()` 352
4. `as_retriever(search_kwargs={"k": 2})`로 질문과 비슷한 문서 2개 검색
5. `ChatPromptTemplate`에 `{context}`, `{question}`을 넣고 `gpt-4.1`(temperature 0)로 답변 → `query_rag()`
6. Gradio `Blocks`로 챗봇 화면 띄우기

**핵심 코드**
```python
docs = retriever.invoke(question)
context = "\n\n".join(doc.page_content for doc in docs)
messages = prompt.format_messages(question=question, context=context)
response = llm.invoke(messages)
```

**결과**
- "디커플링이란?", "비트코인", "EC방식" → PDF 내용을 바탕으로 반말로 설명
- "23년 9월 서울 집값" → **"검색 결과에 없어서 답변할 수 없어"** (모르면 모른다고 답함 = 할루시네이션 방지)
- 정치 관련 질문 → 금융 용어가 아니라서 답변 거절
- 임베딩 길이 1536 (OpenAI `text-embedding-ada-002` 계열)

**주의할 점**
- 데이터 정제(앞뒤 불필요한 페이지 제거)가 검색 품질에 직접 영향을 줍니다.
- 다운로드 로그 날짜가 11월 14일이라, 일부 출력은 원본 노트북에 저장돼 있던 결과일 수 있습니다.
- Gradio에서 `type` 미지정 경고가 납니다. `gr.Chatbot(type="messages")`로 바꾸면 사라집니다.

## 03. 과제 — 멀티턴 · 멀티쿼리 · 도구 선택

| 파트 | 목표 | 출력 형식 |
|---|---|---|
| 1. 멀티턴 쿼리 생성기 | 이전 질문·답변을 보고 현재 질문을 완전한 검색어로 바꾸기 ("전주는?" → "전주 여행지 추천") | 검색어 하나 |
| 2. 멀티 쿼리 생성기 | 한 문장에 섞인 여러 질문을 쪼개기 | `['검색어1', '검색어2']` |
| 3. 멀티턴 + 멀티쿼리 | 1과 2를 동시에 | `검색어(번호): 검색어` |
| 4. 도구 매칭 | 질문 조각마다 알맞은 도구 고르기 (없으면 `NoTool`) | `도구이름: 검색어` |

**주의할 점**
- 2·3·4번 프롬프트는 아직 `여기에 어떤 설명을 할까요? 여러분들이 채워보세요.` 빈칸 그대로입니다. 지금 보이는 출력은 원본 노트북에 저장돼 있던 정답 예시입니다.
- 1번은 시스템 프롬프트가 "현재 질문에 대한 **답**을 해주시오"로 되어 있어서, "검색어만 반환"이라는 과제 조건과 다릅니다.
- 2번은 `eval()`로 문자열을 리스트로 바꾸므로, 모델이 형식을 조금만 어겨도 에러가 납니다. `json.loads` + 예외 처리가 더 안전합니다.

## 04. Plan & Execute 에이전트

**흐름**
1. 일본·미국 ICT 보고서 PDF → `chunk_size=300, overlap=100` → FAISS (`text-embedding-3-large`), 각각 129개·144개 청크
2. `Tool(name, description, vectorstore)` 클래스로 `japan_ict`, `usa_ict`, `no_tool` 정의
3. `ToolPlanGenerator`: 최근 2턴 대화 + 현재 질문 → `{"plan": {"도구": ["검색어", ...]}}` JSON 계획 생성 (`gpt-4.1`)
4. `MultiToolRAG.query()`: 계획대로 도구별로 검색(쿼리당 3개)한 뒤, 모은 문서로 최종 답변
5. Gradio UI에 도구 호출 로그 누적 표시

**결과**
| 질문 | 생성된 계획 |
|---|---|
| 일본 ICT 상황이 궁금해 | `japan_ict`: ["일본 ICT 시장 동향 및 현황"] |
| 그 나라의 정치 상황, 한국과 협업, 미국 신사업은? | `japan_ict` 2개 + `usa_ict` 1개 (멀티턴 "그 나라" = 일본, 멀티쿼리 3개) |
| 너는 누구니? | `no_tool` → 검색 없이 답변 |

**주의할 점**
- Gradio 셀에서 `Data incompatible with messages format` 에러가 났습니다. Gradio 6은 대화 기록을 `{"role": ..., "content": ...}` 딕셔너리로 받아야 하는데, 코드는 `(질문, 답변)` 튜플로 넘기고 있습니다.

## 05. ReACT 에이전트

**흐름**
1. 같은 ICT PDF 2개 → `PyMuPDFLoader` → `from_tiktoken_encoder(chunk_size=512)` → Chroma
2. `create_retriever_tool`로 `japan_ict`, `usa_ict` 도구 만들기
3. ReAct 프롬프트: `Question / Thought / Action / Action Input / Observation / Final Answer`
4. `run_react()`가 직접 루프를 돎: LLM 호출(`stop=["\nObservation:"]`) → 정규식으로 Action 파싱 → 도구 실행 결과를 Observation으로 붙임 → `Final Answer`가 나올 때까지 반복 (`max_iters`)
5. `ConversationManager`로 이전 대화를 포함한 Gradio 데모

**핵심 아이디어**
```python
resp = llm.invoke(rendered, stop=["\nObservation:"])  # 검색 결과는 LLM이 지어내지 않게 여기서 멈춤
tool, action_input = _parse_action_and_input(resp.content)
```

**결과**
- "미국·일본 ICT 정책 공통점/차이점 + 미국 블록체인 스타트업 지원" → 두 도구를 번갈아 여러 번 검색한 뒤 종합 답변
- "미국 정부 기구, 법령, 진출 사례 각각 + 일본 AI 정책" → 항목마다 따로 검색 (NTIA, FCC 등)

## 06. Function Calling

**흐름**
1. 가짜 날씨 DB와 `get_weather(location)` → `tools` 명세(JSON Schema) 작성
2. `gpt-4.1`이 "파리의 날씨는 어때?"를 보고 `get_weather({"location": "Paris"})`를 **골라 줌** (실행은 우리 코드가 함)
3. 실행 결과를 `role="tool"` 메시지로 다시 넣고 두 번째 호출 → "현재 파리의 날씨는 흐리고, 기온은 약 14도입니다."
4. 쇼핑몰 CS 봇으로 확장: 고객 프로필 / 주문 / 배송 / 포인트 / 결제 조회 함수 5개 + `execute_function` 매핑
5. Gradio UI (버전에 따라 `messages`/`tuples` 자동 판별), 이어서 FastAPI + 직접 만든 HTML UI 버전

**주의할 점**
- 모델은 함수를 **실행하지 않고 "무엇을 어떤 인자로 부를지"만 정해 줍니다.** 실행과 결과 전달은 코드 몫입니다.
- 고객 ID는 시스템 메시지로 고정해서, 모델이 다른 고객 정보를 조회하지 않게 했습니다.

---

## 에이전트 3종 비교

| | Plan & Execute (04) | ReACT (05) | Function Calling (06) |
|---|---|---|---|
| 계획 시점 | **처음에 한 번** 전체 계획(JSON) | **매 단계** 다음 행동을 생각 | 모델이 호출할 함수를 그때그때 선택 |
| 도구 선택 방식 | 프롬프트로 JSON 출력 → 직접 파싱 | 텍스트(`Action:`)를 정규식으로 파싱 | OpenAI `tools` API가 구조화된 `tool_calls`로 반환 |
| 반복 | 계획대로 한 번 실행 | 답이 나올 때까지 반복 (`max_iters`) | 함수 결과를 넣고 다시 호출 (필요하면 여러 번) |
| 장점 | 호출 수가 적고 빠름, 계획을 로그로 보기 쉬움 | 앞 검색 결과를 보고 다음 검색을 정할 수 있음 | 형식이 깨질 걱정이 적음, 인자 검증이 쉬움 |
| 단점 | 처음 계획이 틀리면 고칠 기회가 없음 | 호출이 많아 느리고 비쌈, 파싱이 깨질 수 있음 | API(모델)에 기능이 있어야 함 |

## 오늘의 에러 모음

| 노트북 | 에러 | 원인 | 해결 |
|---|---|---|---|
| 01 | `RateLimitError 429 insufficient_quota` | OpenAI **API 크레딧 0원** (ChatGPT Plus 구독과 API 결제는 별개) | platform.openai.com → Billing에서 충전, 또는 무료 모델(HuggingFace)로 교체 |
| 03 | `AuthenticationError 401 expired_secret_key` | API 키 만료 | 새 키 발급 후 Secrets에 다시 등록 |
| 03 | `NameError: name 'client' is not defined` | `client = OpenAI(...)` 셀을 먼저 실행하지 않음 (원본에 저장된 출력) | 위에서부터 순서대로 실행 |
| 04 | `gradio.exceptions.Error: Data incompatible with messages format` | Gradio 6은 대화 기록을 role/content 딕셔너리로 받아야 함 | `history.append({"role": "user", ...})` 형식으로 변경 |
| 01·02·04·05 | `DeprecationWarning: langchain-community is being sunset` | `langchain_community` 패키지 지원 종료 예정 (동작은 함) | 장기적으로 `langchain-chroma` 같은 개별 패키지로 이동 |

## 남은 과제

- [ ] 03 과제: 2·3·4번 빈칸 프롬프트 채우고 다시 실행하기
- [ ] 03 과제: 1번 프롬프트를 "검색어만 반환"하도록 고치기
- [ ] 01: Qwen / ko-sroberta로 바꾼 셀을 다시 실행해서 출력 갱신하기 (지금은 OpenAI 시절 출력)
- [ ] 04: Gradio 대화 기록을 messages 형식으로 고치기
- [ ] OpenAI 키 새로 발급 또는 크레딧 충전 (01·03 실행에 필요)
