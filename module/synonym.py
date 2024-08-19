#%%
import re
import openai
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

def generate_synonym(instruction, cred="./assets/cred/chatGPT_api_key.txt"):
    response = chat(instruction, cred)
    response_message = response.choices[0]["message"]["content"]
    response_message = response_message.split('\n')
    
    delimiter = '|'.join([f'{i+1}번.' for i in range(5)])
    
    syn_list = []
    for message in response_message:
        syns = re.split(delimiter, message)[-1].strip()
        syn_list.append(syns)
    return syn_list


def chat(sample, cred="./assets/cred/chatGPT_api_key.txt"):
    openai.api_key_path = cred ### API key 경로 입력
    if sample is None:
        raise ValueError

    prompt = f"""\
    주어진 지시에 대한 적절한 응답을 생성해주세요. 이러한 작업 지침은 ChatGPT 모델에 주어지며, ChatGPT 모델이 지침을 완료하는지 평가합니다.

    요구 사항은 다음과 같습니다:
    1. 응답은 불필요한 내용이 없이 간결하고 명확하게 작성되어야 합니다.
    2. 응답을 생성할 때 사용된 동사와 명사는 그대로 사용하지 않고 다양성을 극대화해야 합니다.
    3. 응답에 사용된 명사들은 동의어나 줄임말을 적극적으로 활용하여 다양성을 극대화해야 합니다.
    4. GPT 언어 모델은 지시를 완료할 수 있어야 합니다.
    5. 응답은 한국어로 작성되어야 합니다.
    6. 출력은 명령어에 대한 적절한 응답이어야 합니다.
    
    주어진 지시에 대한 적절한 응답을 생성해주세요.
    또한, 아래와 같은 형식으로 5개의 동의어를 순서대로 작성해주세요.
    각 응답은 아래와 같이 줄바꿈으로 구분됩니다:
    1번. 동의어
    2번. 동의어
    3번. 동의어
    4번. 동의어
    5번. 동의어
    
    주어진 지시: 다음 단어와 같은 주제와 내용을 다루고 있지만, 겹치는 단어가 최대한 적고 표현방식에서 차이가 있는 동의어를 5개만 알려주세요: "{sample}". 
    """
    
    messages = []
    messages.extend([
        {"role": "system", "content": "You're a helpful assistant that transforms given words into different ones while maintaining the same meaning."},
        {"role": "user", "content": prompt}])

    @retry(
        wait=wait_random_exponential(min=1, max=60), ### API 호출 사이의 시간 간격은 지수분포로부터 샘플링
        stop=stop_after_attempt(100)) ### 오류가 발생하더라도 100번을 시도
    def completions_with_backoff(**kwargs):
        return openai.ChatCompletion.create(**kwargs) ### 실제로 답변을 생성하는 함수
    response = completions_with_backoff(
        model="gpt-4o-mini", messages=messages)
    
    return response
#%%
sample = "3D 창작터 운영 및 관리"
response = chat(sample)
response_message = response.choices[0]["message"]["content"]
response_message = response_message.split('\n')
#%%