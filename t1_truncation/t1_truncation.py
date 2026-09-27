import os
import logging
import time
import httpx2
from dotenv import load_dotenv
from anthropic import Anthropic
import anthropic
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"),
                    timeout=httpx2.Timeout( connect=5.0,read=30.0,write=30.0,pool=30.0)
                   )

for max_output_token in [5,50,500] :

    start = time.perf_counter()

    try :
        response = client.messages.create(
            model = os.getenv("ANTHROPIC_MODEL_NAME"),
            max_tokens = max_output_token,
            messages = [{"role": "user", "content": "用一句话解释递归"}]
        )
    except httpx2.TimeoutException :
        log.info("超时")
        continue# 兜底降级路线

    except anthropic.AuthenticationError:
        log.error("API Key 无效")
        break

    except anthropic.APIStatusError as e:
        log.error("API错误: %s", e)

    end = time.perf_counter()

    output = "".join(
        block.text
        for block in response.content
        if block.type == "text"
    )
    for block in response.content :
        print(block.type)
    if response.stop_reason == "end_turn":
        log.info("模型调用成功")
    elif response.stop_reason == "max_tokens":
        if len(output)==0 :
            log.info("max_output_token过小 模型没有输出")
        else :
            log.info("输出被截断 模型输出超出max_output_token限制")

    elif response.stop_reason == "tool_use":
        log.info("调用工具")
    elif response.stop_reason == "refusal":
        log.info("模型拒绝 记录并走降级")
    else :
        log.info("兜底分支")
    log.info("stop_reason      :%s", response.stop_reason)
    log.info("output_chars     :%s", len(output))
    log.info("input_tokens     :%s", response.usage.input_tokens)
    log.info("output_tokens    :%s", response.usage.output_tokens)
    log.info("raw_output       :%s", repr(output))
    log.info("time             :%.3f s",end-start)
