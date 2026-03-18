import os
import io
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import logging

# ==================== 配置 ====================
ENABLE_MOCK_MODEL = False  # True: 测试前端(不加载模型), False: 正常加载模型
BASE_MODEL_PATH = "./models/Qwen/Qwen2.5-VL-7B-Instruct"
LORA_PATH = "./sft_models/qwen2.5vl_7b_instruct_smt/"
PROMPT_TEXT = "请作为资深SMT质检专家，检测这张连接器图片。"

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(title="MDSC SMT Connector Inspector", version="1.0")

# 挂载静态文件目录（用于提供 index.html）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局模型和 processor
model = None
processor = None


@app.on_event("startup")
async def load_model():
    global model, processor
    if ENABLE_MOCK_MODEL:
        logger.info("🚀 [调试模式] ENABLE_MOCK_MODEL=True，跳过基座模型和 LoRA 的加载...")
        model = "mock_model"
        processor = "mock_processor"
        return

    logger.info(f"🚀 [初始化] 正在加载基座模型: {BASE_MODEL_PATH} ...")
    try:
        # 设备自适应选择：判断 GPU, MPS (Mac), 否则 CPU
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # NOTE: 对于 Qwen2.5-VL，Apple MPS 可能会由于单 buffer 超过 4G 限制 
            # (total bytes of NDArray > 2**32) 导致推理报错。默认使用 cpu 最稳妥。
            device = "cpu"
            dtype = torch.float32
        else:
            device = "cpu"
            dtype = torch.float32

        logger.info(f"检测到可用设备: {device}, 使用数值精度: {dtype}")

        # 加载基座模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=dtype,
            device_map={"": device},
            attn_implementation="sdpa",
        )

        processor = AutoProcessor.from_pretrained(
            BASE_MODEL_PATH,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28
        )
        logger.info(f"✅ 基座加载成功！运行在 {device} 设备上。")

        # 挂载 LoRA
        logger.info(f"🔗 [初始化] 正在挂载 LoRA: {LORA_PATH} ...")
        model = PeftModel.from_pretrained(model, LORA_PATH)
        model.eval()
        logger.info(f"✅ LoRA 挂载成功！")

    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise RuntimeError("Failed to load model")


def run_inference(image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()


@app.post("/inspect")
async def inspect_smt_image(file: UploadFile = File(...)):
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="仅支持 JPG/PNG/BMP/WEBP 格式图片"
        )

    try:
        import time
        start_time = time.time()
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 推理或模拟输出
        if ENABLE_MOCK_MODEL:
            import random
            import asyncio
            await asyncio.sleep(1.5) # 模拟推理耗时
            if random.choice([True, False]):
                result = "检测结论：PASS - 良品\n详细分析：连接器整体外观良好。两侧的接地焊片平整贴合，信号引脚排列整齐，无连锡或变形。"
            else:
                result = "检测结论：FAIL - 卡扣变形\n详细分析：连接器卡扣部位存在明显的物理变形，塑料主体扭曲，可能导致锁紧力不足。"
        else:
            result = run_inference(image)

        # 解析模型返回的新格式 (PASS/FAIL)
        # 支持 "检测结论：PASS" 或者直接包含 "FAIL" 等字眼
        if "FAIL" in result:
            status = "FAIL"
        elif "PASS" in result:
            status = "PASS"
        else:
            # 兜底逻辑
            status = "UNKNOWN"

        end_time = time.time()
        duration = round(end_time - start_time, 2)

        return JSONResponse({
            "success": True,
            "ai_judgment": result,
            "status": status,
            "duration": duration
        })

    except Exception as e:
        logger.error(f"推理出错: {e}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
