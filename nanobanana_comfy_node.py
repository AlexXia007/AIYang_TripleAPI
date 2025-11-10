import json
import time
import urllib.error
import urllib.request
import urllib.parse
import io
import re
import hashlib
import hmac
import threading
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import torch


DEFAULT_SERVER_URL = "https://api.nanobananaapi.ai"
ENDPOINT_PATH = "/api/v1/nanobanana/generate"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

# 豆包API相关常量
# API端点路径常量
# DOUBAO_ENDPOINT_PATH = "/images/generations"  # 不再使用，用户填写完整接口地址

# 火山引擎API签名相关
VOLCENGINE_HOST = "visual.volcengineapi.com"
VOLCENGINE_REGION = "cn-north-1"
VOLCENGINE_SERVICE = "cv"


def _create_volcengine_signature(
    access_key_id: str,
    secret_access_key: str,
    method: str,
    uri: str,
    query_string: str,
    headers: Dict[str, str],
    body: str,
    timestamp: str
) -> str:
    """创建火山引擎API签名 - 基于官方示例"""
    
    # 计算请求体哈希
    body_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()
    
    # 规范化头部 - 按照官方示例的顺序
    signed_headers_list = ["content-type", "host", "x-content-sha256", "x-date"]
    signed_headers_str = ";".join(signed_headers_list)
    
    # 构建规范头部字符串
    canonical_headers_list = [
        f"content-type:{headers.get('Content-Type', '')}",
        f"host:{headers.get('Host', '')}",
        f"x-content-sha256:{body_hash}",
        f"x-date:{timestamp}"
    ]
    canonical_headers_str = "\n".join(canonical_headers_list)
    
    # 创建规范请求
    canonical_request = "\n".join([
        method.upper(),
        uri,
        query_string,
        canonical_headers_str,
        "",  # 空行
        signed_headers_str,
        body_hash
    ])
    
    # 计算规范请求的哈希
    canonical_request_hash = hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
    
    # 创建待签名字符串
    date_part = timestamp[:8]  # YYYYMMDD
    credential_scope = f"{date_part}/{VOLCENGINE_REGION}/{VOLCENGINE_SERVICE}/request"
    
    string_to_sign = "\n".join([
        "HMAC-SHA256",
        timestamp,
        credential_scope,
        canonical_request_hash
    ])
    
    # 派生签名密钥
    k_date = hmac.new(secret_access_key.encode('utf-8'), date_part.encode('utf-8'), hashlib.sha256).digest()
    k_region = hmac.new(k_date, VOLCENGINE_REGION.encode('utf-8'), hashlib.sha256).digest()
    k_service = hmac.new(k_region, VOLCENGINE_SERVICE.encode('utf-8'), hashlib.sha256).digest()
    k_signing = hmac.new(k_service, "request".encode('utf-8'), hashlib.sha256).digest()
    
    # 计算签名
    signature = hmac.new(k_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
    
    # 创建Authorization头
    authorization = f"HMAC-SHA256 Credential={access_key_id}/{credential_scope}, SignedHeaders={signed_headers_str}, Signature={signature}"
    
    return authorization


def _calculate_aspect_ratio(width: int, height: int) -> str:
    """计算图片的宽高比，返回最接近的NanoBanana支持的比例"""
    # 验证输入参数
    if width is None or height is None or width <= 0 or height <= 0:
        print(f"[警告] 无效的图片尺寸 width={width}, height={height}，使用默认比例 1:1")
        return "1:1"
    
    ratio = width / height
    print(f"[调试] 计算宽高比: {width}/{height} = {ratio:.6f}")
    
    # 定义NanoBanana支持的比例及其阈值
    supported_ratios = {
        "1:1": 1.0,        # 正方形
        "9:16": 9/16,      # 0.5625 (竖屏手机)
        "16:9": 16/9,      # 1.777... (横屏宽屏)
        "3:4": 3/4,        # 0.75 (竖屏)
        "4:3": 4/3,        # 1.333... (横屏)
        "3:2": 3/2,        # 1.5 (横屏)
        "2:3": 2/3,        # 0.666... (竖屏)
        "5:4": 5/4,        # 1.25 (横屏)
        "4:5": 4/5,        # 0.8 (竖屏)
        "21:9": 21/9,      # 2.333... (超宽屏)
    }
    
    # 找到差值最小的比例
    min_diff = float('inf')
    best_ratio = "1:1"  # 默认值
    
    for ratio_name, target_ratio in supported_ratios.items():
        diff = abs(ratio - target_ratio)
        print(f"[调试] 比例 {ratio_name} ({target_ratio:.6f}): 差值 = {diff:.6f}")
        if diff < min_diff:
            min_diff = diff
            best_ratio = ratio_name
    
    print(f"[调试] 最终匹配比例: {best_ratio} (差值 = {min_diff:.6f})")
    return best_ratio


def _get_image_size_with_exif(image: Image.Image) -> Tuple[int, int]:
    """获取图片的实际尺寸，考虑EXIF方向信息
    
    当图片有EXIF方向信息（orientation）时，需要根据方向信息调整宽高。
    例如：如果orientation=6（顺时针旋转90度），则实际显示时需要交换宽高。
    
    如果没有EXIF信息或EXIF中没有orientation信息，则返回图片的默认尺寸。
    
    Args:
        image: PIL Image对象
        
    Returns:
        (width, height): 实际显示的尺寸
    """
    width, height = image.size
    
    # 检查EXIF方向信息
    try:
        exif = image.getexif()
        orientation = exif.get(274)  # EXIF标签274是Orientation
        if orientation:
            # orientation值说明：
            # 1 = 正常（0度）- 不需要交换
            # 3 = 旋转180度 - 不需要交换（尺寸不变）
            # 6 = 顺时针旋转90度（需要交换宽高）
            # 8 = 逆时针旋转90度（需要交换宽高）
            if orientation in [6, 8]:  # 需要旋转90度或270度
                # 交换宽高
                width, height = height, width
        # 如果没有orientation信息或orientation为None/1/3，使用原始尺寸（已赋值，无需修改）
    except Exception:
        # 如果获取EXIF失败或图片没有EXIF信息，使用原始尺寸（已赋值，无需修改）
        pass
    
    return width, height


def _encode_url_for_request(url: str) -> str:
    """对URL进行编码处理，处理中文字符
    
    智能判断是否需要编码：
    1. 如果路径已经编码（包含%XX格式），不再次编码，避免破坏签名
    2. 如果路径包含中文字符但未编码，进行编码
    3. 对于签名URL（通常已编码），直接返回原URL
    
    Args:
        url: 原始URL
        
    Returns:
        编码后的URL（如果需要编码）或原URL（如果已编码或无需编码）
    """
    try:
        parsed = urllib.parse.urlparse(url)
        
        # 检查路径是否已经编码（如果包含%XX格式，说明已经编码）
        # 对于已经编码的URL，直接返回原URL，避免双重编码破坏签名
        if '%' in parsed.path:
            # 路径已经编码，直接返回原URL（特别是OSS签名URL）
            return url
        
        # 检查路径是否包含中文字符（需要编码）
        # 判断是否包含非ASCII字符（包括中文）
        try:
            # 尝试解码路径，如果包含非ASCII字符，quote会进行编码
            decoded_path = urllib.parse.unquote(parsed.path)
            # 检查是否包含中文字符或其他非ASCII字符
            has_non_ascii = any(ord(c) > 127 for c in decoded_path)
            
            if has_non_ascii:
                # 包含中文字符或其他非ASCII字符，需要进行编码
                encoded_path = urllib.parse.quote(parsed.path, safe='/:')
                # 重新组装URL
                safe_url = urllib.parse.urlunparse((
                    parsed.scheme, 
                    parsed.netloc, 
                    encoded_path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment
                ))
                return safe_url
            else:
                # 不包含非ASCII字符，无需编码
                return url
        except Exception:
            # 如果解码失败，尝试直接检查原始路径
            has_non_ascii = any(ord(c) > 127 for c in parsed.path)
            if has_non_ascii:
                # 包含非ASCII字符，进行编码
                encoded_path = urllib.parse.quote(parsed.path, safe='/:')
                safe_url = urllib.parse.urlunparse((
                    parsed.scheme, 
                    parsed.netloc, 
                    encoded_path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment
                ))
                return safe_url
            else:
                return url
    except Exception as e:
        # 如果编码失败，返回原URL（避免破坏签名URL）
        return url


def _get_image_dimensions_from_url(url: str) -> Tuple[int, int]:
    """从图片URL获取图片尺寸
    
    采用多级备选方案：
    1. 优先使用GET请求（适用于大多数情况，包括签名URL）
    2. 如果GET请求失败，尝试HEAD请求作为备选方案
    3. 支持中文路径的URL编码
    
    Args:
        url: 图片URL
        
    Returns:
        (width, height): 图片尺寸，如果获取失败返回 (None, None)
    """
    # 对URL进行编码处理，避免中文路径导致的问题
    safe_url = _encode_url_for_request(url)
    
    # 方案1: 优先尝试GET请求（适用于大多数情况，包括签名URL）
    try:
        request = urllib.request.Request(
            url=safe_url,
            method="GET",
            headers={"User-Agent": DEFAULT_USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            # 检查是否为图片
            content_type = resp.headers.get('Content-Type', '')
            if content_type and 'image' not in content_type.lower():
                print(f"非图片类型: {content_type}")
                return None, None
            
            # 尝试多种方法获取图片尺寸
            
            # 方法1: 尝试读取部分数据（8KB）
            try:
                img_data = resp.read(8192)  # 读取前8KB
                with Image.open(io.BytesIO(img_data)) as im:
                    return _get_image_size_with_exif(im)  # (width, height)
            except Exception as e1:
                print(f"[GET请求] 方法1失败: {str(e1)}")
                
            # 方法2: 重新请求，读取完整图片
            try:
                request2 = urllib.request.Request(
                    url=safe_url,
                    method="GET",
                    headers={"User-Agent": DEFAULT_USER_AGENT}
                )
                with urllib.request.urlopen(request2, timeout=30) as resp2:
                    img_data = resp2.read()  # 读取完整图片
                    with Image.open(io.BytesIO(img_data)) as im:
                        return _get_image_size_with_exif(im)  # (width, height)
            except Exception as e2:
                print(f"[GET请求] 方法2失败: {str(e2)}")
                
            # 方法3: 尝试从HTTP头获取尺寸信息
            try:
                content_length = resp.headers.get('Content-Length')
                if content_length:
                    print(f"图片大小: {content_length} bytes")
                # 某些服务器可能在头信息中包含尺寸
                return None, None
            except Exception as e3:
                print(f"[GET请求] 方法3失败: {str(e3)}")
                
    except urllib.error.HTTPError as e:
        # HTTP错误（如403、404等），尝试HEAD请求作为备选方案
        error_code = e.code
        error_msg = str(e)
        print(f"[GET请求] HTTP错误 {error_code}: {error_msg}，尝试HEAD请求作为备选方案")
        
        # 方案2: HEAD请求备选方案
        try:
            head_request = urllib.request.Request(
                url=safe_url,
                method="HEAD",
                headers={"User-Agent": DEFAULT_USER_AGENT}
            )
            with urllib.request.urlopen(head_request, timeout=30) as head_resp:
                # 检查是否为图片
                content_type = head_resp.headers.get('Content-Type', '')
                if content_type and 'image' not in content_type.lower():
                    print(f"[HEAD请求] 非图片类型: {content_type}")
                    return None, None
                
                # HEAD请求通常不返回响应体，但某些服务器可能在头信息中包含尺寸
                # 如果HEAD请求成功，尝试使用GET请求获取实际数据
                print(f"[HEAD请求] 请求成功，尝试使用GET请求获取图片数据")
                
                # 重新尝试GET请求（可能HEAD请求验证了URL有效性）
                try:
                    get_request = urllib.request.Request(
                        url=safe_url,
                        method="GET",
                        headers={"User-Agent": DEFAULT_USER_AGENT}
                    )
                    with urllib.request.urlopen(get_request, timeout=30) as get_resp:
                        img_data = get_resp.read(8192)  # 读取前8KB
                        with Image.open(io.BytesIO(img_data)) as im:
                            return _get_image_size_with_exif(im)
                except Exception as get_e:
                    print(f"[HEAD备选方案] GET请求仍然失败: {str(get_e)}")
                    return None, None
                    
        except urllib.error.HTTPError as head_e:
            # HEAD请求也失败
            print(f"[HEAD请求] HTTP错误 {head_e.code}: {str(head_e)}")
            return None, None
        except Exception as head_e:
            # HEAD请求其他错误
            print(f"[HEAD请求] 错误: {str(head_e)}")
            return None, None
            
    except Exception as e:
        # 其他错误（网络错误、超时等）
        try:
            error_msg = str(e)
            print(f"获取图片尺寸失败: {error_msg}")
        except UnicodeEncodeError:
            print(f"获取图片尺寸失败: {repr(e)}")
        return None, None


def _determine_image_size(image_size_mode: str, image_urls: Optional[List[str]]) -> str:
    """确定最终的image_size参数"""
    
    # 如果用户选择了特定比例，直接使用
    if image_size_mode != "auto":
        print(f"用户选择固定比例: {image_size_mode}")
        return image_size_mode
    
    # AUTO模式：需要从输入图片计算比例
    if not image_urls or len(image_urls) == 0:
        print("AUTO模式但未提供输入图片，使用默认比例 1:1")
        return "1:1"
    
    # 获取第一张输入图片的尺寸
    first_image_url = image_urls[0]
    print(f"[AUTO模式] 分析输入图片: {first_image_url}")
    
    width, height = _get_image_dimensions_from_url(first_image_url)
    
    if width is None or height is None:
        print(f"[AUTO模式] 无法获取图片尺寸，使用默认比例 1:1")
        return "1:1"
    
    # 验证尺寸有效性
    if width <= 0 or height <= 0:
        print(f"[AUTO模式] 警告: 获取到无效的图片尺寸 {width}×{height}，使用默认比例 1:1")
        return "1:1"
    
    # 输出获取到的尺寸信息
    print(f"[AUTO模式] 成功获取图片尺寸: {width}×{height}")
    print(f"[AUTO模式] 图片宽高比: {width/height:.6f} ({width}:{height})")
    
    # 计算最接近的比例
    calculated_ratio = _calculate_aspect_ratio(width, height)
    print(f"[AUTO模式] 最终选择比例: {calculated_ratio}")
    
    return calculated_ratio


def _build_request_body(
    prompt: str,
    generation_type: str,
    num_images: int,
    image_urls: Optional[List[str]],
    seed: Optional[int],
    watermark: Optional[str],
    image_size_mode: str = "auto",
) -> Dict[str, Any]:
    if not prompt:
        raise ValueError("prompt 不能为空")
    if generation_type not in {"IMAGETOIAMGE", "TEXTTOIAMGE"}:
        raise ValueError("type 只能是 TEXTTOIAMGE 或 IMAGETOIAMGE")
    if not (1 <= num_images <= 4):
        raise ValueError("numImages 必须在 1 到 4 之间")

    body: Dict[str, Any] = {
        "prompt": prompt,
        "type": generation_type,
        "numImages": num_images,
        "callBackUrl": "https://api.nanobananaapi.ai/callback",  # 必填的回调URL，使用官方默认值
    }
    
    print(f"banana请求api参数: {json.dumps(body, ensure_ascii=False)}")

    # 处理图片尺寸
    final_image_size = _determine_image_size(image_size_mode, image_urls)
    body["image_size"] = final_image_size

    if watermark:
        body["watermark"] = watermark
    if seed is not None:
        try:
            body["seed"] = int(seed)
        except Exception:
            pass

    if generation_type == "IMAGETOIAMGE":
        if not image_urls:
            raise ValueError("IMAGETOIAMGE 模式下必须提供 image_urls")
        body["imageUrls"] = image_urls

    return body


def _send_request(
    server_url: str,
    api_key: str,
    body: Dict[str, Any],
    timeout_seconds: int = 60,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Dict[str, Any]:
    # 如果server_url已经包含完整路径，则不添加ENDPOINT_PATH
    if "/api/v1/nanobanana/generate" in server_url:
        url = server_url.rstrip("/")
    else:
        url = server_url.rstrip("/") + ENDPOINT_PATH
    data_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "User-Agent": user_agent or DEFAULT_USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "Connection": "keep-alive",
    }

    request = urllib.request.Request(url=url, data=data_bytes, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
            resp_text = resp.read().decode("utf-8", errors="replace")
            print(f"banana api响应: {resp_text}")
            try:
                return json.loads(resp_text)
            except json.JSONDecodeError:
                return {"raw": resp_text}
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(err_text)
        except json.JSONDecodeError:
            payload = {"error": err_text}
        payload["http_status"] = e.code
        return payload
    except urllib.error.URLError as e:
        return {"error": str(e)}


def _clean_url_token(token: str) -> Optional[str]:
    s = (token or "").strip()
    if not s:
        return None
    # 支持以 @ 开头的输入格式
    if s.startswith("@"):
        s = s.lstrip("@").strip()
    # 去除可能的包裹符 <...>
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()
    # 仅接受 http/https
    if not (s.startswith("http://") or s.startswith("https://")):
        return None
    return s


def _parse_image_urls_field(text: str) -> Optional[List[str]]:
    """解析任意输入（JSON 数组/单字符串/按换行或逗号或空白分隔，支持 @ 前缀）为 URL 列表。"""
    txt = (text or "").strip()
    if not txt:
        return None
    urls: List[str] = []
    # 优先尝试 JSON
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, list):
            for u in parsed:
                if isinstance(u, str):
                    cu = _clean_url_token(u)
                    if cu:
                        urls.append(cu)
            seen = set()
            return [u for u in urls if not (u in seen or seen.add(u))] or None
        elif isinstance(parsed, str):
            cu = _clean_url_token(parsed)
            return ([cu] if cu else None)
    except Exception:
        pass
    # 自由格式：按换行/逗号/空白分割
    tokens = [t for t in re.split(r"[\r\n,\s]+", txt) if t.strip()]
    for t in tokens:
        cu = _clean_url_token(t)
        if cu:
            urls.append(cu)
    if not urls:
        return None
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


class NanoBananaGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "type": (["TEXTTOIAMGE", "IMAGETOIAMGE"], {"default": "TEXTTOIAMGE"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image_size_mode": (["auto", "1:1", "9:16", "16:9", "3:4", "4:3", "3:2", "2:3", "5:4", "4:5", "21:9"], {"default": "auto"}),
            },
            "optional": {
                "image_urls_json": ("STRING", {"default": "[]", "multiline": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 99999999}),
                "watermark": ("STRING", {"default": ""}),
                "server": ("STRING", {"default": DEFAULT_SERVER_URL}),
                "timeout_seconds": ("INT", {"default": 60, "min": 5, "max": 300}),
                "wait_for_result": ("BOOLEAN", {"default": False}),
                "poll_interval_seconds": ("INT", {"default": 3, "min": 1, "max": 60}),
                "poll_timeout_seconds": ("INT", {"default": 240, "min": 5, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("task_id", "response_json", "result_image")
    FUNCTION = "generate"
    CATEGORY = "NanoBanana"

    def generate(
        self,
        prompt: str,
        type: str,
        num_images: int,
        api_key: str,
        image_urls_json: str = "[]",
        seed: int = -1,
        watermark: str = "",
        image_size_mode: str = "auto",
        server: str = DEFAULT_SERVER_URL,
        timeout_seconds: int = 60,
        wait_for_result: bool = False,
        poll_interval_seconds: int = 3,
        poll_timeout_seconds: int = 240,  # 与 INPUT_TYPES 保持一致
    ) -> Tuple[str, str, Any]:
        if not api_key:
            raise ValueError("未提供 API Key。请在节点中填写 api_key。")

        image_urls: Optional[List[str]] = _parse_image_urls_field(image_urls_json)

        # 限制种子值范围
        processed_seed = None
        if seed is not None and seed >= 0:
            # 如果种子值超出范围，取模限制在合理范围内
            processed_seed = int(seed) % 100000000 if seed > 99999999 else int(seed)
        
        body = _build_request_body(
            prompt=prompt,
            generation_type=type,
            num_images=num_images,
            image_urls=image_urls,
            seed=processed_seed,
            watermark=watermark or None,
            image_size_mode=image_size_mode,
        )

        response = _send_request(
            server_url=server,
            api_key=api_key,
            body=body,
            timeout_seconds=timeout_seconds,
            user_agent=DEFAULT_USER_AGENT,
        )

        task_id: str = ""
        try:
            task_id = str(response.get("data", {}).get("taskId", ""))
            print(f"NanoBanana: API响应解析 - taskId: {task_id}")
            if not task_id:
                print(f"NanoBanana: 完整响应内容: {json.dumps(response, ensure_ascii=False, indent=2)}")
        except Exception as e:
            task_id = ""
            print(f"NanoBanana: 解析taskId时出错: {e}")
            print(f"NanoBanana: 原始响应: {json.dumps(response, ensure_ascii=False, indent=2)}")

        # 如果用户选择等待结果且拿到了 task_id，则进入轮询
        if wait_for_result and task_id:
            print(f"NanoBanana: 开始轮询任务 {task_id}")
            final_info = _poll_task(
                server_url=server,
                api_key=api_key,
                task_id=task_id,
                interval_seconds=poll_interval_seconds,
                timeout_seconds=poll_timeout_seconds,
                user_agent=DEFAULT_USER_AGENT,
            )
            print(f"NanoBanana: 轮询完成，结果: {json.dumps(final_info, ensure_ascii=False, indent=2)}")
            
            # 提取结果图片地址（支持多张）
            result_image_urls: List[str] = []
            try:
                data_obj = final_info.get("data") or {}
                if isinstance(data_obj, dict):
                    # 1) data.response.*
                    resp_obj = data_obj.get("response") or {}
                    if isinstance(resp_obj, dict):
                        candidates: List[str] = []
                        # 优先收集数组字段
                        for key in ("resultImageUrls", "resultImages", "images", "imageUrls", "image_urls", "result_image_urls", "data"):
                            v = resp_obj.get(key)
                            if isinstance(v, list):
                                for item in v:
                                    if isinstance(item, str) and item.strip():
                                        candidates.append(item)
                                    elif isinstance(item, dict) and "url" in item:
                                        # 处理字典格式的图片项
                                        url = item.get("url")
                                        if isinstance(url, str) and url.strip():
                                            candidates.append(url)
                        # 兼容单字段
                        for key in ("resultImageUrl", "imageUrl", "image_url", "result_image_url"):
                            v = resp_obj.get(key)
                            if isinstance(v, str) and v.strip():
                                candidates.append(v)
                        # 去重并保序
                        seen: set = set()
                        result_image_urls = [u for u in candidates if not (u in seen or seen.add(u))]
                    # 2) data.* 顶层数组字段
                    if not result_image_urls:
                        for key in ("images", "resultImages", "imageUrls", "image_urls", "result_image_urls", "data"):
                            v = data_obj.get(key)
                            if isinstance(v, list):
                                for item in v:
                                    if isinstance(item, str) and item.strip():
                                        result_image_urls.append(item)
                                    elif isinstance(item, dict) and "url" in item:
                                        # 处理字典格式的图片项
                                        url = item.get("url")
                                        if isinstance(url, str) and url.strip():
                                            result_image_urls.append(url)
                        # 兼容处理 - 处理所有图片，不只是第一张
                        if not result_image_urls:
                            images_list = data_obj.get("images") or data_obj.get("resultImages")
                            if isinstance(images_list, list) and images_list:
                                # 处理所有图片项，不只是第一张
                                for item in images_list:
                                    if isinstance(item, str) and item.strip():
                                        result_image_urls.append(item)
                                    elif isinstance(item, dict) and "url" in item:
                                        # 如果item是字典且包含url字段
                                        url = item.get("url")
                                        if isinstance(url, str) and url.strip():
                                            result_image_urls.append(url)
            except Exception as e:
                print(f"NanoBanana: 解析图片URL失败: {str(e)}")
                pass
            
            print(f"NanoBanana: 解析到图片URLs: {result_image_urls}")
            
            # 下载并转换为 ComfyUI IMAGE 批次张量
            result_image = None
            if result_image_urls:
                images_tensors: List[torch.Tensor] = []
                for idx, url in enumerate(result_image_urls):
                    try:
                        print(f"NanoBanana: 下载图片 {idx+1}/{len(result_image_urls)}: {url}")
                        # 保持每张图片的原始尺寸，不强制调整
                        tensor_img = _download_image_to_tensor_with_size(url, target_size=None, timeout_seconds=min(timeout_seconds, 60))
                        images_tensors.append(tensor_img)
                        # 打印每张图片的实际尺寸（使用编码后的URL）
                        try:
                            safe_url = _encode_url_for_request(url)
                            with urllib.request.urlopen(urllib.request.Request(safe_url, headers={"User-Agent": DEFAULT_USER_AGENT}), timeout=min(timeout_seconds, 60)) as resp:
                                img_bytes = resp.read()
                            with Image.open(io.BytesIO(img_bytes)) as im:
                                width, height = _get_image_size_with_exif(im)
                                print(f"NanoBanana: 图片{idx+1}实际尺寸: {width}x{height}")
                        except Exception as size_e:
                            print(f"NanoBanana: 无法读取图片{idx+1}尺寸: {size_e}")
                    except Exception as e:
                        print(f"NanoBanana: 图片下载失败 {url}: {str(e)}")
                        continue
                if images_tensors:
                    try:
                        result_image = torch.cat(images_tensors, dim=0)
                        print(f"NanoBanana: 成功下载 {len(images_tensors)} 张图片")
                    except Exception as e:
                        print(f"NanoBanana: 图片拼接失败: {str(e)}")
                        # 退化到第一张
                        result_image = images_tensors[0]
                else:
                    print("NanoBanana: 所有图片下载失败")
            else:
                print("NanoBanana: 未找到图片URL")
            
            response_json = json.dumps(final_info, ensure_ascii=False)
            return (task_id, response_json, result_image)
        else:
            # 如果不等待结果，返回任务ID和响应，但图片为None
            # 提示用户需要等待结果才能获取图片
            response_json = json.dumps(response, ensure_ascii=False)
            if not wait_for_result:
                print("NanoBanana: 未启用等待结果，请勾选 wait_for_result 选项以获取图片")
            elif not task_id:
                print("NanoBanana: 未获取到任务ID，可能API调用失败")
            return (task_id, response_json, None)


def _get_task_info(
    server_url: str,
    api_key: str,
    task_id: str,
    timeout_seconds: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Dict[str, Any]:
    # 处理server_url，如果包含完整路径则提取基础URL
    base = server_url.rstrip("/")
    if "/api/v1/nanobanana/generate" in base:
        base = base.replace("/api/v1/nanobanana/generate", "")
    
    query = urllib.parse.urlencode({"taskId": task_id})
    url = f"{base}/api/v1/nanobanana/record-info?{query}"
    print(f"NanoBanana: 轮询URL: {url}")

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "User-Agent": user_agent or DEFAULT_USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "Connection": "keep-alive",
    }

    request = urllib.request.Request(url=url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
            resp_text = resp.read().decode("utf-8", errors="replace")
            print(f"banana 轮询api响应: {resp_text}")
            try:
                result = json.loads(resp_text)
                print(f"NanoBanana: 解析后的响应: {json.dumps(result, ensure_ascii=False)}")
                return result
            except json.JSONDecodeError:
                return {"raw": resp_text}
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(err_text)
        except json.JSONDecodeError:
            payload = {"error": err_text}
        payload["http_status"] = e.code
        return payload
    except urllib.error.URLError as e:
        return {"error": str(e)}


def _poll_task(
    server_url: str,
    api_key: str,
    task_id: str,
    interval_seconds: int,
    timeout_seconds: int,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Dict[str, Any]:
    start_ts = time.time()
    last_payload: Dict[str, Any] = {}
    consecutive_timeouts = 0  # 连续超时计数
    max_consecutive_timeouts = 3  # 最多允许连续3次超时
    
    while True:
        try:
            # 使用传入的 timeout_seconds 参数作为单个请求的超时时间
            payload = _get_task_info(
                server_url=server_url,
                api_key=api_key,
                task_id=task_id,
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
            )
            last_payload = payload
            
            # 检查是否是错误响应（包括超时错误）
            if "error" in payload:
                error_msg = payload.get("error", "")
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    consecutive_timeouts += 1
                    print(f"NanoBanana: 轮询请求超时 ({consecutive_timeouts}/{max_consecutive_timeouts}): {error_msg}")
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        print(f"NanoBanana: 连续{consecutive_timeouts}次轮询请求超时，停止轮询")
                        return {
                            "error": f"连续轮询请求超时({consecutive_timeouts}次)", 
                            "last": payload, 
                            "consecutive_timeouts": consecutive_timeouts
                        }
                else:
                    # 其他错误，也记录但允许重试
                    print(f"NanoBanana: 轮询请求错误: {error_msg}")
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        return {
                            "error": f"连续轮询错误", 
                            "last": payload, 
                            "consecutive_timeouts": consecutive_timeouts
                        }
            else:
                consecutive_timeouts = 0  # 重置超时计数（成功响应）
            
            # HTTP错误检查
            if payload.get("http_status") and payload["http_status"] >= 500:
                consecutive_timeouts += 1
                print(f"NanoBanana: HTTP {payload['http_status']} 错误 ({consecutive_timeouts}/{max_consecutive_timeouts})")
                if consecutive_timeouts >= max_consecutive_timeouts:
                    return {
                        "error": f"HTTP {payload['http_status']} 错误", 
                        "last": payload, 
                        "consecutive_timeouts": consecutive_timeouts
                    }
            
        except Exception as e:
            # 捕获所有未预期的异常（包括socket.timeout等）
            consecutive_timeouts += 1
            print(f"NanoBanana: 轮询请求异常 ({consecutive_timeouts}/{max_consecutive_timeouts}): {str(e)}")
            if consecutive_timeouts >= max_consecutive_timeouts:
                return {
                    "error": f"轮询异常: {str(e)}", 
                    "last": last_payload, 
                    "consecutive_timeouts": consecutive_timeouts
                }
            # 等待后继续尝试
            time.sleep(max(1, interval_seconds))
            continue
        
        # 尝试解析状态
        status = 0
        data = {}
        try:
            data = payload.get("data") or {}
            if not isinstance(data, dict):
                data = payload if isinstance(payload, dict) else {}
            
            status = int(data.get("successFlag", 0))
            print(f"NanoBanana: 轮询状态 {status}, 数据: {json.dumps(data, ensure_ascii=False)}")
        except Exception as e:
            status = 0
            print(f"NanoBanana: 解析状态失败: {str(e)}, 原始响应: {json.dumps(payload, ensure_ascii=False)}")

        # 检查任务是否完成
        if status in (1, 2, 3):
            print(f"NanoBanana: 任务完成，状态: {status}")
            return payload

        # 检查总超时
        elapsed = time.time() - start_ts
        if elapsed >= timeout_seconds:
            print(f"NanoBanana: 轮询超时 ({timeout_seconds}秒)，最后状态: {status}, 已耗时: {elapsed:.1f}秒")
            return {
                "error": "poll_timeout", 
                "data": data, 
                "last": last_payload, 
                "timeout_seconds": timeout_seconds,
                "elapsed": elapsed
            }

        # 等待下一次轮询
        time.sleep(max(1, interval_seconds))


def _download_image_to_tensor(url: str, timeout_seconds: int = 30):
    """下载图片并转换为 ComfyUI IMAGE（torch.Tensor，形状 [1, H, W, 3]，值域 0-1）。"""
    # 对URL进行编码处理
    safe_url = _encode_url_for_request(url)
    request = urllib.request.Request(
        url=safe_url,
        method="GET",
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Connection": "keep-alive",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
        img_bytes = resp.read()
    with Image.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")
        arr = np.array(im, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
    # [H, W, C] -> add batch dim -> [1, H, W, C]
    tensor = tensor.unsqueeze(0).contiguous()
    return tensor


def _download_image_to_tensor_with_size(
    url: str,
    target_size: Optional[Tuple[int, int]] = None,  # (W, H)
    return_size: bool = False,
    timeout_seconds: int = 30,
):
    """下载图片并转换为 ComfyUI IMAGE；可选对齐到 target_size；
    return_size=True 时返回 (tensor, (W, H))
    """
    # 对URL进行编码处理
    safe_url = _encode_url_for_request(url)
    request = urllib.request.Request(
        url=safe_url,
        method="GET",
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Connection": "keep-alive",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
        img_bytes = resp.read()
    with Image.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")
        if target_size is not None:
            # target_size 为 (W, H)
            im = im.resize((int(target_size[0]), int(target_size[1])), Image.BICUBIC)
        width, height = _get_image_size_with_exif(im)
        arr = np.array(im, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
    tensor = tensor.unsqueeze(0).contiguous()
    if return_size:
        return tensor, (width, height)
    return tensor


def _create_white_image_tensor(width: int = 512, height: int = 512) -> torch.Tensor:
    """创建一个白色图片张量，用于错误情况下的返回。"""
    # 创建白色图片 [1, H, W, 3]，值域 0-1
    white_array = np.ones((height, width, 3), dtype=np.float32)
    tensor = torch.from_numpy(white_array).unsqueeze(0).contiguous()
    return tensor

def _create_fallback_or_white_image_tensor(
    fallback_image_url: str,
    timeout_seconds: int = 60,
) -> torch.Tensor:
    """
    优先返回前端提供的链接图片；若无或下载失败，则返回白色底图。
    """
    try:
        if fallback_image_url and isinstance(fallback_image_url, str) and fallback_image_url.strip():
            return _download_image_to_tensor_with_size(
                fallback_image_url.strip(),
                timeout_seconds=timeout_seconds,
            )
    except Exception:
        pass
    return _create_white_image_tensor()


def _extract_image_urls_from_response(response_data: Dict[str, Any]) -> List[str]:
    """从API响应中提取图片URLs"""
    urls: List[str] = []
    
    try:
        # 处理NanoBanana响应格式
        if "data" in response_data and isinstance(response_data["data"], dict):
            data_obj = response_data["data"]
            
            # 1) data.response.*
            resp_obj = data_obj.get("response") or {}
            if isinstance(resp_obj, dict):
                candidates: List[str] = []
                # 优先收集数组字段
                for key in ("resultImageUrls", "resultImages", "images", "imageUrls", "image_urls", "result_image_urls", "data"):
                    v = resp_obj.get(key)
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and item.strip():
                                candidates.append(item)
                            elif isinstance(item, dict) and "url" in item:
                                url = item.get("url")
                                if isinstance(url, str) and url.strip():
                                    candidates.append(url)
                # 兼容单字段
                for key in ("resultImageUrl", "imageUrl", "image_url", "result_image_url"):
                    v = resp_obj.get(key)
                    if isinstance(v, str) and v.strip():
                        candidates.append(v)
                # 去重并保序
                seen: set = set()
                urls.extend([u for u in candidates if not (u in seen or seen.add(u))])
            
            # 2) data.* 顶层数组字段
            if not urls:
                for key in ("images", "resultImages", "imageUrls", "image_urls", "result_image_urls", "data"):
                    v = data_obj.get(key)
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and item.strip():
                                urls.append(item)
                            elif isinstance(item, dict) and "url" in item:
                                url = item.get("url")
                                if isinstance(url, str) and url.strip():
                                    urls.append(url)
                # 兼容处理 - 处理所有图片，不只是第一张
                if not urls:
                    images_list = data_obj.get("images") or data_obj.get("resultImages")
                    if isinstance(images_list, list) and images_list:
                        for item in images_list:
                            if isinstance(item, str) and item.strip():
                                urls.append(item)
                            elif isinstance(item, dict) and "url" in item:
                                url = item.get("url")
                                if isinstance(url, str) and url.strip():
                                    urls.append(url)
        
        # 处理豆包响应格式
        elif "data" in response_data and isinstance(response_data["data"], list):
            for item in response_data["data"]:
                if isinstance(item, dict) and "url" in item:
                    url = item.get("url")
                    if isinstance(url, str) and url.strip():
                        urls.append(url)
        
        # 处理即梦响应格式
        elif "data" in response_data and isinstance(response_data["data"], dict):
            data_obj = response_data["data"]
            if "image_urls" in data_obj and isinstance(data_obj["image_urls"], list):
                urls.extend([url for url in data_obj["image_urls"] if isinstance(url, str) and url.strip()])
            elif "image_url" in data_obj and isinstance(data_obj["image_url"], str):
                urls.append(data_obj["image_url"])
    
    except Exception as e:
        print(f"提取图片URLs失败: {str(e)}")
    
    return urls


def _send_doubao_request(
    api_key: str,
    prompt: str,
    size: str = "1K",
    model: str = "doubao-seedream-4-0-250828",
    watermark: bool = True,
    seed: Optional[int] = None,
    image_urls: Optional[List[str]] = None,
    sequential_generation: str = "disabled",
    num_images: int = 1,
    timeout_seconds: int = 60,
    server_url: str = "https://ark.cn-beijing.volces.com/api/v3/images/generations",
) -> Dict[str, Any]:
    """发送豆包 API 请求 - 使用HTTP请求方式，支持转发站"""
    try:
        # 构建API URL - 用户填写的是完整的接口地址
        url = server_url.rstrip('/')
        
        # 构建请求体
        body = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": "url",
            "watermark": watermark,
            "stream": False
        }
        
        # 添加图片输入（如果提供）
        if image_urls and len(image_urls) > 0:
            if len(image_urls) == 1:
                body["image"] = image_urls[0]  # 单图模式
            else:
                body["image"] = image_urls  # 多图模式
        
        # 添加组图生成控制
        if sequential_generation == "auto":
            body["sequential_image_generation"] = "auto"
            
            # 计算实际可生成的最大图片数量
            input_image_count = len(image_urls) if image_urls else 0
            max_allowed_images = 15 - input_image_count
            
            if max_allowed_images <= 0:
                print(f"TripleAPI: 豆包API错误 - 输入图片数量({input_image_count})已达到15张限制")
                return {"error": f"输入图片数量({input_image_count})已达到15张限制"}
            
            body["sequential_image_generation_options"] = {
                "max_images": min(num_images, max_allowed_images)
            }
            
            print(f"TripleAPI: 豆包组图生成 - 输入图片:{input_image_count}张, 请求生成:{num_images}张, 实际生成:{min(num_images, max_allowed_images)}张")
        else:
            body["sequential_image_generation"] = sequential_generation
        
        # 添加 seed 参数（如果提供）
        if seed is not None and seed >= 0:
            limited_seed = seed % 100000000 if seed > 99999999 else seed
            body["seed"] = limited_seed
        
        print(f"豆包请求API参数: {json.dumps(body, ensure_ascii=False, indent=2)}")
        
        # 构建请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": DEFAULT_USER_AGENT,
        }
        
        # 发送HTTP请求
        data_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url=url, data=data_bytes, headers=headers, method="POST")
        
        with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
            resp_text = resp.read().decode("utf-8", errors="replace")
            try:
                response_data = json.loads(resp_text)
                print(f"豆包API响应: {response_data}")
                return response_data
            except json.JSONDecodeError:
                return {"raw": resp_text}
        
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(err_text)
        except json.JSONDecodeError:
            payload = {"error": err_text}
        payload["http_status"] = e.code
        print(f"豆包API HTTP错误: {payload}")
        return payload
    except urllib.error.URLError as e:
        error_msg = str(e)
        print(f"豆包API URL错误: {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = str(e)
        print(f"豆包API未知错误: {error_msg}")
        return {"error": error_msg}


def _send_hidream_request_multi(
    req_key: str,
    prompt: str,
    hidream_type: str = "txt2img",
    image_urls: Optional[List[str]] = None,
    seed: Optional[int] = None,
    scale: float = 0.5,
    timeout_seconds: int = 60,
    server_url: str = "https://visual.volcengineapi.com",
) -> Dict[str, Any]:
    """发送即梦 API 4.0 请求 - 支持多图输入（最多10张）"""
    
    # 从req_key解析access_key_id和secret_access_key
    if ":" not in req_key:
        return {"error": "req_key格式错误，应为 'access_key_id:secret_access_key'"}
    
    access_key_id, secret_access_key = req_key.split(":", 1)
    
    # API基本信息
    method = "POST"
    uri = "/"
    query_params = {
        "Action": "CVSync2AsyncSubmitTask",
        "Version": "2022-08-31"
    }
    query_string = "&".join([f"{k}={v}" for k, v in sorted(query_params.items())])
    
    # 构建请求体
    body_data = {
        "req_key": "jimeng_t2i_v40",
        "prompt": prompt,
        "scale": scale,
    }
    
    # 添加 seed 参数（如果提供）
    if seed is not None and seed >= 0:
        limited_seed = seed % 100000000 if seed > 99999999 else seed
        body_data["seed"] = limited_seed
    
    # 如果是图生图类型，添加图片URLs（支持多图，最多10张）
    if hidream_type in ["img2img", "shopImg2img", "vton"] and image_urls:
        # 限制最多10张图片
        limited_urls = image_urls[:10] if len(image_urls) > 10 else image_urls
        body_data["image_urls"] = limited_urls
        if len(image_urls) > 10:
            print(f"即梦: 输入图片超过10张，已截取前10张: {len(limited_urls)}张")
    
    body = json.dumps(body_data)
    print(f"即梦请求api参数: {body}")
    
    # 创建时间戳
    now = datetime.utcnow()
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    
    # 计算请求体哈希
    body_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()
    
    # 从server_url中提取host
    from urllib.parse import urlparse
    parsed_url = urlparse(server_url)
    host = parsed_url.hostname or VOLCENGINE_HOST
    
    # 构建请求头
    headers = {
        "Host": host,
        "Content-Type": "application/json",
        "X-Date": timestamp,
        "X-Content-Sha256": body_hash,
        "User-Agent": DEFAULT_USER_AGENT,
    }
    
    # 创建签名
    authorization = _create_volcengine_signature(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        method=method,
        uri=uri,
        query_string=query_string,
        headers=headers,
        body=body,
        timestamp=timestamp
    )
    
    headers["Authorization"] = authorization
    
    # 构建完整URL
    url = f"{server_url.rstrip('/')}{uri}?{query_string}"
    
    # 发送请求
    request = urllib.request.Request(
        url=url,
        data=body.encode("utf-8"),
        headers=headers,
        method=method
    )
    
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
            resp_text = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(resp_text)
            except json.JSONDecodeError:
                return {"raw": resp_text}
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(err_text)
        except json.JSONDecodeError:
            payload = {"error": err_text}
        payload["http_status"] = e.code
        return payload
    except urllib.error.URLError as e:
        return {"error": str(e)}


def _send_hidream_request(
    req_key: str,
    prompt: str,
    hidream_type: str = "txt2img",
    image_url: Optional[str] = None,
    seed: Optional[int] = None,
    scale: float = 0.5,
    timeout_seconds: int = 60,
    server_url: str = "https://visual.volcengineapi.com",
) -> Dict[str, Any]:
    """发送即梦 API 4.0 请求 - 提交任务"""
    
    # 从req_key解析access_key_id和secret_access_key
    # 格式应该是: "access_key_id:secret_access_key"
    if ":" not in req_key:
        return {"error": "req_key格式错误，应为 'access_key_id:secret_access_key'"}
    
    access_key_id, secret_access_key = req_key.split(":", 1)
    
    # API基本信息
    method = "POST"
    uri = "/"
    query_params = {
        "Action": "CVSync2AsyncSubmitTask",
        "Version": "2022-08-31"
    }
    query_string = "&".join([f"{k}={v}" for k, v in sorted(query_params.items())])
    
    # 构建请求体
    body_data = {
        "req_key": "jimeng_t2i_v40",
        "prompt": prompt,
        "scale": scale,
    }
    
    # 添加 seed 参数（如果提供）
    if seed is not None and seed >= 0:
        # 限制种子值范围
        limited_seed = seed % 100000000 if seed > 99999999 else seed
        body_data["seed"] = limited_seed
    
    # 如果是图生图类型，添加图片URL
    if hidream_type in ["img2img", "shopImg2img", "vton"] and image_url:
        body_data["image_urls"] = [image_url]
    
    body = json.dumps(body_data)
    
    # 创建时间戳
    now = datetime.utcnow()
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    
    # 计算请求体哈希
    body_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()
    
    # 从server_url中提取host
    from urllib.parse import urlparse
    parsed_url = urlparse(server_url)
    host = parsed_url.hostname or VOLCENGINE_HOST
    
    # 构建请求头
    headers = {
        "Host": host,
        "Content-Type": "application/json",
        "X-Date": timestamp,
        "X-Content-Sha256": body_hash,
        "User-Agent": DEFAULT_USER_AGENT,
    }
    
    # 创建签名
    authorization = _create_volcengine_signature(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        method=method,
        uri=uri,
        query_string=query_string,
        headers=headers,
        body=body,
        timestamp=timestamp
    )
    
    headers["Authorization"] = authorization
    
    # 构建完整URL
    url = f"{server_url.rstrip('/')}{uri}?{query_string}"
    
    # 发送请求
    request = urllib.request.Request(
        url=url,
        data=body.encode("utf-8"),
        headers=headers,
        method=method
    )
    
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
            resp_text = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(resp_text)
            except json.JSONDecodeError:
                return {"raw": resp_text}
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(err_text)
        except json.JSONDecodeError:
            payload = {"error": err_text}
        payload["http_status"] = e.code
        return payload
    except urllib.error.URLError as e:
        return {"error": str(e)}


def _check_hidream_status(
    req_key: str,
    task_id: str,
    timeout_seconds: int = 30,
    server_url: str = "https://visual.volcengineapi.com",
) -> Dict[str, Any]:
    """查询即梦 API 4.0 任务状态"""
    
    # 从req_key解析access_key_id和secret_access_key
    # 格式应该是: "access_key_id:secret_access_key"
    if ":" not in req_key:
        return {"error": "req_key格式错误，应为 'access_key_id:secret_access_key'"}
    
    access_key_id, secret_access_key = req_key.split(":", 1)
    
    # API基本信息
    method = "POST"
    uri = "/"
    query_params = {
        "Action": "CVSync2AsyncGetResult",
        "Version": "2022-08-31"
    }
    query_string = "&".join([f"{k}={v}" for k, v in sorted(query_params.items())])
    
    # 构建请求体
    body_data = {
        "req_key": "jimeng_t2i_v40",
        "task_id": task_id,
        "req_json": json.dumps({
            "return_url": True,
            "logo_info": {
                "add_logo": False
            }
        }, ensure_ascii=False)
    }
    
    body = json.dumps(body_data)
    
    # 创建时间戳
    now = datetime.utcnow()
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    
    # 计算请求体哈希
    body_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()
    
    # 从server_url中提取host
    from urllib.parse import urlparse
    parsed_url = urlparse(server_url)
    host = parsed_url.hostname or VOLCENGINE_HOST
    
    # 构建请求头
    headers = {
        "Host": host,
        "Content-Type": "application/json",
        "X-Date": timestamp,
        "X-Content-Sha256": body_hash,
        "User-Agent": DEFAULT_USER_AGENT,
    }
    
    # 创建签名
    authorization = _create_volcengine_signature(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        method=method,
        uri=uri,
        query_string=query_string,
        headers=headers,
        body=body,
        timestamp=timestamp
    )
    
    headers["Authorization"] = authorization
    
    # 构建完整URL
    url = f"{server_url.rstrip('/')}{uri}?{query_string}"
    
    # 发送请求
    request = urllib.request.Request(
        url=url,
        data=body.encode("utf-8"),
        headers=headers,
        method=method
    )
    
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
            resp_text = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(resp_text)
            except json.JSONDecodeError:
                return {"raw": resp_text}
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(err_text)
        except json.JSONDecodeError:
            payload = {"error": err_text}
        payload["http_status"] = e.code
        return payload
    except urllib.error.URLError as e:
        return {"error": str(e)}


def _call_nanobanana_api(
    prompt: str,
    nanobanana_type: str,
    nanobanana_num_images: int,
    nanobanana_image_size_mode: str,
    nanobanana_api_key: str,
    image_urls_input: str,
    seed: int,
    nanobanana_watermark: str,
    nanobanana_server_url: str,
    fallback_image_url: str,
    poll_interval: int,
    poll_timeout: int,
    max_retries: int = 3,
    retry_interval: int = 2,
) -> Tuple[Any, str, str, str]:
    """调用NanoBanana API的并行执行函数（支持重试）"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"TripleAPI: NanoBanana第{attempt + 1}次重试...")
            else:
                print(f"TripleAPI: 开始调用NanoBanana API，类型: {nanobanana_type}")
            
            nanobanana_node = NanoBananaGenerate()
            task_id, response_json, result_image = nanobanana_node.generate(
                prompt=prompt,
                type=nanobanana_type,
                num_images=nanobanana_num_images,
                api_key=nanobanana_api_key,
                image_urls_json=image_urls_input or "[]",
                seed=seed,
                watermark=nanobanana_watermark,
                image_size_mode=nanobanana_image_size_mode,
                server=nanobanana_server_url,
                timeout_seconds=poll_timeout,
                wait_for_result=True,
                poll_interval_seconds=poll_interval,
                poll_timeout_seconds=poll_timeout,
            )
            
            # 检查响应是否包含错误（API调用失败的情况）
            has_error = False
            error_msg = None
            try:
                if response_json:
                    response_data = json.loads(response_json) if isinstance(response_json, str) else response_json
                    # 检查是否有错误字段或HTTP错误状态
                    if isinstance(response_data, dict):
                        if "error" in response_data or "http_status" in response_data:
                            has_error = True
                            error_msg = response_data.get("error", f"HTTP错误: {response_data.get('http_status', 'unknown')}")
            except Exception:
                pass  # 如果解析失败，继续检查其他条件
            
            # 如果result_image为None且task_id为空，说明API调用失败，应该重试
            if result_image is None and (not task_id or has_error):
                if has_error:
                    error_msg = error_msg or "API返回错误响应"
                else:
                    error_msg = "未获取到任务ID，API调用可能失败"
                # 抛出异常以触发重试逻辑
                raise Exception(error_msg)
            
            if result_image is not None:
                status_info = f"NanoBanana: 成功(第{attempt + 1}次尝试); "
                print(f"TripleAPI: NanoBanana成功，图片形状: {result_image.shape}")
            else:
                # 这种情况应该是正常情况（比如用户选择不等待结果）
                status_info = f"NanoBanana: 成功但无图片(第{attempt + 1}次尝试); "
                print("TripleAPI: NanoBanana成功但未返回图片")
            
            return result_image, task_id, response_json, status_info
            
        except Exception as e:
            last_error = e
            print(f"TripleAPI: NanoBanana第{attempt + 1}次尝试失败: {str(e)}")
            if attempt < max_retries - 1:
                print(f"TripleAPI: NanoBanana将在{retry_interval}秒后重试...")
                time.sleep(retry_interval)  # 固定重试间隔
    
    # 所有重试都失败了
    status_info = f"NanoBanana: 失败(已重试{max_retries}次) - {str(last_error)}; "
    print(f"TripleAPI: NanoBanana最终失败: {str(last_error)}")
    result_image = _create_fallback_or_white_image_tensor(
        fallback_image_url=fallback_image_url,
        timeout_seconds=min(poll_timeout, 60),
    )
    response_json = json.dumps({"error": str(last_error), "retries": max_retries}, ensure_ascii=False)
    return result_image, "", response_json, status_info


def _call_doubao_api(
    prompt: str,
    doubao_type: str,
    doubao_api_key: str,
    image_urls_input: str,
    seed: int,
    doubao_model: str,
    doubao_size: str,
    doubao_watermark: bool,
    doubao_sequential_generation: str,
    doubao_num_images: int,
    fallback_image_url: str,
    poll_timeout: int,
    max_retries: int = 3,
    retry_interval: int = 2,
    doubao_server_url: str = "https://ark.cn-beijing.volces.com",
) -> Tuple[Any, str, str]:
    """调用豆包API的并行执行函数（支持重试）"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"TripleAPI: 豆包第{attempt + 1}次重试...")
            else:
                print(f"TripleAPI: 开始调用豆包 API，类型: {doubao_type}")
            
            # 从统一的图片输入中解析URL用于豆包
            doubao_image_urls = []
            if doubao_type in ["img2img", "multi_img_fusion"]:
                if image_urls_input:
                    parsed_urls = _parse_image_urls_field(image_urls_input)
                    if parsed_urls:
                        if doubao_type == "img2img":
                            # 图生图模式：只使用第一张图片
                            doubao_image_urls = [parsed_urls[0]]
                            print(f"TripleAPI: 豆包图生图模式，使用第一张图片: {doubao_image_urls[0]}")
                        elif doubao_type == "multi_img_fusion":
                            # 多图融合模式：使用所有图片
                            doubao_image_urls = parsed_urls
                            print(f"TripleAPI: 豆包多图融合模式，使用所有图片: {len(doubao_image_urls)}张")
                
                if not doubao_image_urls:
                    status_info = "豆包: 图生图模式需要提供图片URL; "
                    print("TripleAPI: 豆包图生图模式需要提供图片URL")
                    doubao_image = _create_fallback_or_white_image_tensor(
                        fallback_image_url=fallback_image_url,
                        timeout_seconds=min(poll_timeout, 60),
                    )
                    doubao_response = json.dumps({"error": "图生图模式需要提供图片URL"}, ensure_ascii=False)
                    return doubao_image, doubao_response, status_info
                else:
                    # 有图片URL，继续API调用
                    print(f"TripleAPI: 豆包使用图片URLs: {doubao_image_urls}")
                    doubao_result = _send_doubao_request(
                        api_key=doubao_api_key,
                        prompt=prompt,
                        size=doubao_size,
                        model=doubao_model,
                        watermark=doubao_watermark,
                        seed=(None if seed < 0 else seed),
                        image_urls=doubao_image_urls,
                        sequential_generation=doubao_sequential_generation,
                        num_images=doubao_num_images,
                        timeout_seconds=poll_timeout,
                        server_url=doubao_server_url,
                    )
                    
                    doubao_response = json.dumps(doubao_result, ensure_ascii=False)
                    print(f"TripleAPI: 豆包API响应: {doubao_result}")
                    
                    # 检查响应是否包含错误（API调用失败的情况）
                    has_error = False
                    error_msg = None
                    if isinstance(doubao_result, dict):
                        if "error" in doubao_result or "http_status" in doubao_result:
                            has_error = True
                            error_msg = doubao_result.get("error", f"HTTP错误: {doubao_result.get('http_status', 'unknown')}")
                    
                    # 如果API返回错误，抛出异常以触发重试逻辑
                    if has_error:
                        raise Exception(error_msg or "API返回错误响应")
                    
                    # 提取图片URL并下载（支持多张组图）
                    if "data" in doubao_result and isinstance(doubao_result["data"], list) and doubao_result["data"]:
                        image_urls = []
                        for item in doubao_result["data"]:
                            if isinstance(item, dict) and "url" in item:
                                image_urls.append(item["url"])
                        
                        if image_urls:
                            try:
                                print(f"TripleAPI: 豆包下载{len(image_urls)}张图片")
                                images_tensors = []
                                target_size = None
                                
                                for idx, image_url in enumerate(image_urls):
                                    print(f"TripleAPI: 豆包下载图片 {idx+1}/{len(image_urls)}: {image_url}")
                                    if target_size is None:
                                        tensor, wh = _download_image_to_tensor_with_size(image_url, return_size=True, timeout_seconds=min(poll_timeout, 60))
                                        target_size = wh
                                        images_tensors.append(tensor)
                                    else:
                                        tensor = _download_image_to_tensor_with_size(image_url, target_size=target_size, timeout_seconds=min(poll_timeout, 60))
                                        images_tensors.append(tensor)
                                
                                if images_tensors:
                                    doubao_image = torch.cat(images_tensors, dim=0)
                                    status_info = f"豆包: 成功({doubao_type}, {len(images_tensors)}张图片, 第{attempt + 1}次尝试); "
                                    print(f"TripleAPI: 豆包成功，图片形状: {doubao_image.shape}")
                                else:
                                    status_info = "豆包: 所有图片下载失败; "
                                    doubao_image = _create_fallback_or_white_image_tensor(
                                        fallback_image_url=fallback_image_url,
                                        timeout_seconds=min(poll_timeout, 60),
                                    )
                            except Exception as e:
                                status_info = f"豆包: 图片下载失败 - {str(e)}; "
                                print(f"TripleAPI: 豆包图片下载失败: {str(e)}")
                                doubao_image = _create_fallback_or_white_image_tensor(
                                    fallback_image_url=fallback_image_url,
                                    timeout_seconds=min(poll_timeout, 60),
                                )
                        else:
                            status_info = "豆包: 响应中未找到图片URL; "
                            print("TripleAPI: 豆包响应中未找到图片URL")
                            doubao_image = _create_fallback_or_white_image_tensor(
                                fallback_image_url=fallback_image_url,
                                timeout_seconds=min(poll_timeout, 60),
                            )
                    else:
                        status_info = "豆包: API响应格式错误; "
                        print("TripleAPI: 豆包API响应格式错误")
                        doubao_image = _create_fallback_or_white_image_tensor(
                            fallback_image_url=fallback_image_url,
                            timeout_seconds=min(poll_timeout, 60),
                        )
            else:
                # 文生图模式，不需要图片
                print("TripleAPI: 豆包文生图模式")
                doubao_result = _send_doubao_request(
                    api_key=doubao_api_key,
                    prompt=prompt,
                    size=doubao_size,
                    model=doubao_model,
                    watermark=doubao_watermark,
                    seed=(None if seed < 0 else seed),
                    image_urls=None,
                    sequential_generation=doubao_sequential_generation,
                    num_images=doubao_num_images,
                    timeout_seconds=poll_timeout,
                    server_url=doubao_server_url,
                )
                
                doubao_response = json.dumps(doubao_result, ensure_ascii=False)
                print(f"TripleAPI: 豆包API响应: {doubao_result}")
                
                # 检查响应是否包含错误（API调用失败的情况）
                has_error = False
                error_msg = None
                if isinstance(doubao_result, dict):
                    if "error" in doubao_result or "http_status" in doubao_result:
                        has_error = True
                        error_msg = doubao_result.get("error", f"HTTP错误: {doubao_result.get('http_status', 'unknown')}")
                
                # 如果API返回错误，抛出异常以触发重试逻辑
                if has_error:
                    raise Exception(error_msg or "API返回错误响应")
                
                # 提取图片URL并下载（支持多张组图）
                if "data" in doubao_result and isinstance(doubao_result["data"], list) and doubao_result["data"]:
                    image_urls = []
                    for item in doubao_result["data"]:
                        if isinstance(item, dict) and "url" in item:
                            image_urls.append(item["url"])
                    
                    if image_urls:
                        try:
                            print(f"TripleAPI: 豆包下载{len(image_urls)}张图片")
                            images_tensors = []
                            target_size = None
                            
                            for idx, image_url in enumerate(image_urls):
                                print(f"TripleAPI: 豆包下载图片 {idx+1}/{len(image_urls)}: {image_url}")
                                if target_size is None:
                                    tensor, wh = _download_image_to_tensor_with_size(image_url, return_size=True, timeout_seconds=min(poll_timeout, 60))
                                    target_size = wh
                                    images_tensors.append(tensor)
                                else:
                                    tensor = _download_image_to_tensor_with_size(image_url, target_size=target_size, timeout_seconds=min(poll_timeout, 60))
                                    images_tensors.append(tensor)
                            
                            if images_tensors:
                                doubao_image = torch.cat(images_tensors, dim=0)
                                status_info = f"豆包: 成功(txt2img, {len(images_tensors)}张图片, 第{attempt + 1}次尝试); "
                                print(f"TripleAPI: 豆包成功，图片形状: {doubao_image.shape}")
                            else:
                                status_info = "豆包: 所有图片下载失败; "
                                doubao_image = _create_fallback_or_white_image_tensor(
                                    fallback_image_url=fallback_image_url,
                                    timeout_seconds=min(poll_timeout, 60),
                                )
                        except Exception as e:
                            status_info = f"豆包: 图片下载失败 - {str(e)}; "
                            print(f"TripleAPI: 豆包图片下载失败: {str(e)}")
                            doubao_image = _create_fallback_or_white_image_tensor(
                                fallback_image_url=fallback_image_url,
                                timeout_seconds=min(poll_timeout, 60),
                            )
                    else:
                        status_info = "豆包: 响应中未找到图片URL; "
                        print("TripleAPI: 豆包响应中未找到图片URL")
                        doubao_image = _create_fallback_or_white_image_tensor(
                            fallback_image_url=fallback_image_url,
                            timeout_seconds=min(poll_timeout, 60),
                        )
                else:
                    status_info = "豆包: API响应格式错误; "
                    print("TripleAPI: 豆包API响应格式错误")
                    doubao_image = _create_fallback_or_white_image_tensor(
                        fallback_image_url=fallback_image_url,
                        timeout_seconds=min(poll_timeout, 60),
                    )
            
            return doubao_image, doubao_response, status_info
            
        except Exception as e:
            last_error = e
            print(f"TripleAPI: 豆包第{attempt + 1}次尝试失败: {str(e)}")
            if attempt < max_retries - 1:
                print(f"TripleAPI: 豆包将在{retry_interval}秒后重试...")
                time.sleep(retry_interval)  # 固定重试间隔
    
    # 所有重试都失败了
    status_info = f"豆包: 失败(已重试{max_retries}次) - {str(last_error)}; "
    print(f"TripleAPI: 豆包最终失败: {str(last_error)}")
    doubao_image = _create_fallback_or_white_image_tensor(
        fallback_image_url=fallback_image_url,
        timeout_seconds=min(poll_timeout, 60),
    )
    doubao_response = json.dumps({"error": str(last_error), "retries": max_retries}, ensure_ascii=False)
    return doubao_image, doubao_response, status_info


def _call_hidream_api(
    prompt: str,
    hidream_type: str,
    hidream_req_key: str,
    image_urls_input: str,
    seed: int,
    hidream_scale: float,
    fallback_image_url: str,
    poll_interval: int,
    poll_timeout: int,
    max_retries: int = 3,
    retry_interval: int = 2,
    hidream_server_url: str = "https://visual.volcengineapi.com",
) -> Tuple[Any, str, str, str]:
    """调用即梦API的并行执行函数（支持重试）"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"TripleAPI: 即梦第{attempt + 1}次重试...")
            else:
                print(f"TripleAPI: 开始调用即梦 API，类型: {hidream_type}")
            
            # 从统一的图片输入中解析URLs用于即梦（支持多图，最多10张）
            hidream_image_urls = []
            if image_urls_input:
                parsed_urls = _parse_image_urls_field(image_urls_input)
                if parsed_urls:
                    # 限制最多10张图片
                    hidream_image_urls = parsed_urls[:10] if len(parsed_urls) > 10 else parsed_urls
                    if len(parsed_urls) > 10:
                        print(f"TripleAPI: 即梦输入图片超过10张，已截取前10张: {len(hidream_image_urls)}张")
                    print(f"TripleAPI: 即梦使用图片URLs: {hidream_image_urls}")
            
            hidream_result = _send_hidream_request_multi(
                req_key=hidream_req_key,
                prompt=prompt,
                hidream_type=hidream_type,
                image_urls=hidream_image_urls if hidream_image_urls else None,
                seed=(None if seed < 0 else seed),
                scale=hidream_scale,
                timeout_seconds=poll_timeout,
                server_url=hidream_server_url,
            )
            
            hidream_response = json.dumps(hidream_result, ensure_ascii=False)
            print(f"TripleAPI: 即梦API响应: {hidream_result}")
            
            # 检查响应是否包含错误（API调用失败的情况）
            has_error = False
            error_msg = None
            if isinstance(hidream_result, dict):
                # 检查HTTP错误状态
                if "http_status" in hidream_result:
                    has_error = True
                    error_msg = f"HTTP错误: {hidream_result.get('http_status', 'unknown')}"
                # 检查即梦API错误码（非10000表示错误）
                elif "code" in hidream_result and hidream_result["code"] != 10000:
                    has_error = True
                    error_msg = hidream_result.get("message", f"API错误码: {hidream_result.get('code', 'unknown')}")
                # 检查是否有error字段
                elif "error" in hidream_result:
                    has_error = True
                    error_msg = hidream_result.get("error", "API返回错误")
            
            # 如果API返回错误，抛出异常以触发重试逻辑
            if has_error:
                raise Exception(error_msg or "API返回错误响应")
            
            # 根据API响应处理结果 (即梦API 4.0使用10000作为成功码)
            if "code" in hidream_result and hidream_result["code"] == 10000:
                # 成功响应，检查是否有task_id需要轮询
                if "data" in hidream_result and "task_id" in hidream_result["data"]:
                    hidream_task_id = str(hidream_result["data"]["task_id"])
                    
                    # 轮询任务状态
                    print(f"即梦: 开始轮询任务 {hidream_task_id}")
                    start_time = time.time()
                    poll_count = 0
                    while time.time() - start_time < poll_timeout:
                        poll_count += 1
                        try:
                            print(f"即梦: 第{poll_count}次查询状态...")
                            status_result = _check_hidream_status(
                                req_key=hidream_req_key,
                                task_id=hidream_task_id,
                                timeout_seconds=poll_timeout,
                                server_url=hidream_server_url,
                            )
                            
                            print(f"即梦: 状态查询响应: {json.dumps(status_result, ensure_ascii=False)}")
                            
                            # 检查状态查询响应是否包含HTTP错误（需要重试整个API调用）
                            if isinstance(status_result, dict) and "http_status" in status_result:
                                http_status = status_result.get("http_status", 0)
                                if http_status >= 500:  # 5xx错误表示服务器错误，应该重试
                                    error_msg = f"状态查询HTTP错误: {http_status}"
                                    raise Exception(error_msg)
                            
                            if "code" in status_result and status_result["code"] == 10000:
                                if "data" in status_result and "status" in status_result["data"]:
                                    task_status = status_result["data"]["status"]
                                    print(f"即梦: 任务状态: {task_status}")
                                    
                                    if task_status == "done":
                                        # 任务完成，提取图片URLs（支持多张图片）
                                        image_urls = []
                                        if "image_urls" in status_result["data"] and status_result["data"]["image_urls"]:
                                            image_urls = status_result["data"]["image_urls"]
                                        elif "image_url" in status_result["data"]:
                                            image_urls = [status_result["data"]["image_url"]]
                                        
                                        print(f"即梦: 提取到{len(image_urls)}个图片URL")
                                        
                                        if image_urls:
                                            try:
                                                # 下载所有图片并合并（类似豆包的处理方式）
                                                print(f"TripleAPI: 即梦下载{len(image_urls)}张图片")
                                                images_tensors = []
                                                target_size = None
                                                
                                                for idx, image_url in enumerate(image_urls):
                                                    print(f"TripleAPI: 即梦下载图片 {idx+1}/{len(image_urls)}: {image_url}")
                                                    if target_size is None:
                                                        tensor, wh = _download_image_to_tensor_with_size(image_url, return_size=True, timeout_seconds=min(poll_timeout, 60))
                                                        target_size = wh
                                                        images_tensors.append(tensor)
                                                    else:
                                                        tensor = _download_image_to_tensor_with_size(image_url, target_size=target_size, timeout_seconds=min(poll_timeout, 60))
                                                        images_tensors.append(tensor)
                                                
                                                # 合并所有图片为一张
                                                if images_tensors:
                                                    hidream_image = torch.cat(images_tensors, dim=0)
                                                    print(f"TripleAPI: 即梦成功合并{len(images_tensors)}张图片")
                                                else:
                                                    hidream_image = _create_fallback_or_white_image_tensor(
                                                        fallback_image_url=fallback_image_url,
                                                        timeout_seconds=min(poll_timeout, 60),
                                                    )
                                                
                                                status_info = f"即梦: 成功(第{attempt + 1}次尝试); "
                                                hidream_response = json.dumps(status_result, ensure_ascii=False)
                                                print("即梦: 图片下载成功!")
                                                return hidream_image, hidream_response, hidream_task_id, status_info
                                            except Exception as e:
                                                status_info = f"即梦: 图片下载失败 - {str(e)}; "
                                                hidream_image = _create_fallback_or_white_image_tensor(
                                                    fallback_image_url=fallback_image_url,
                                                    timeout_seconds=min(poll_timeout, 60),
                                                )
                                                print(f"即梦: 图片下载失败 - {str(e)}")
                                                return hidream_image, hidream_response, hidream_task_id, status_info
                                        else:
                                            status_info = "即梦: 响应中未找到图片URL; "
                                            hidream_image = _create_fallback_or_white_image_tensor(
                                                fallback_image_url=fallback_image_url,
                                                timeout_seconds=min(poll_timeout, 60),
                                            )
                                            print("即梦: 响应中未找到图片URL")
                                            return hidream_image, hidream_response, hidream_task_id, status_info
                                    elif task_status in ["not_found", "expired", "failed"]:
                                        error_msg = status_result.get("message", f"任务{task_status}")
                                        status_info = f"即梦: {error_msg}; "
                                        hidream_image = _create_fallback_or_white_image_tensor(
                                            fallback_image_url=fallback_image_url,
                                            timeout_seconds=min(poll_timeout, 60),
                                        )
                                        print(f"即梦: 任务失败 - {error_msg}")
                                        return hidream_image, hidream_response, hidream_task_id, status_info
                                    # 如果是 in_queue, generating 或其他状态，继续等待
                                    else:
                                        print(f"即梦: 任务进行中，状态: {task_status}，继续等待...")
                                else:
                                    print(f"即梦: 状态响应格式异常: {status_result}")
                                    status_info = f"即梦: 状态响应格式异常; "
                                    hidream_image = _create_fallback_or_white_image_tensor(
                                        fallback_image_url=fallback_image_url,
                                        timeout_seconds=min(poll_timeout, 60),
                                    )
                                    return hidream_image, hidream_response, hidream_task_id, status_info
                            else:
                                # API返回错误
                                error_msg = status_result.get("message", "API查询失败")
                                status_info = f"即梦: 查询错误 - {error_msg}; "
                                hidream_image = _create_fallback_or_white_image_tensor(
                                    fallback_image_url=fallback_image_url,
                                    timeout_seconds=min(poll_timeout, 60),
                                )
                                print(f"即梦: API查询错误 - {error_msg}")
                                return hidream_image, hidream_response, hidream_task_id, status_info
                            
                            time.sleep(poll_interval)
                            
                        except Exception as e:
                            status_info = f"即梦: 状态查询错误 - {str(e)}; "
                            hidream_image = _create_fallback_or_white_image_tensor(
                                fallback_image_url=fallback_image_url,
                                timeout_seconds=min(poll_timeout, 60),
                            )
                            print(f"即梦: 状态查询异常 - {str(e)}")
                            return hidream_image, hidream_response, hidream_task_id, status_info
                    else:
                        # 超时
                        status_info = "即梦: 轮询超时; "
                        hidream_image = _create_fallback_or_white_image_tensor(
                            fallback_image_url=fallback_image_url,
                            timeout_seconds=min(poll_timeout, 60),
                        )
                        return hidream_image, hidream_response, hidream_task_id, status_info
                
                # 检查是否直接返回了图片URLs（某些接口可能直接返回）
                elif "data" in hidream_result and ("image_url" in hidream_result["data"] or "image_urls" in hidream_result["data"]):
                    image_urls = []
                    if "image_urls" in hidream_result["data"] and hidream_result["data"]["image_urls"]:
                        image_urls = hidream_result["data"]["image_urls"]
                    elif "image_url" in hidream_result["data"]:
                        image_urls = [hidream_result["data"]["image_url"]]
                    
                    if image_urls:
                        try:
                            # 下载所有图片并合并（类似豆包的处理方式）
                            print(f"TripleAPI: 即梦直接返回{len(image_urls)}张图片")
                            images_tensors = []
                            target_size = None
                            
                            for idx, image_url in enumerate(image_urls):
                                print(f"TripleAPI: 即梦下载图片 {idx+1}/{len(image_urls)}: {image_url}")
                                if target_size is None:
                                    tensor, wh = _download_image_to_tensor_with_size(image_url, return_size=True, timeout_seconds=min(poll_timeout, 60))
                                    target_size = wh
                                    images_tensors.append(tensor)
                                else:
                                    tensor = _download_image_to_tensor_with_size(image_url, target_size=target_size, timeout_seconds=min(poll_timeout, 60))
                                    images_tensors.append(tensor)
                            
                            # 合并所有图片为一张
                            if images_tensors:
                                hidream_image = torch.cat(images_tensors, dim=0)
                                print(f"TripleAPI: 即梦成功合并{len(images_tensors)}张图片")
                            else:
                                hidream_image = _create_fallback_or_white_image_tensor(
                                    fallback_image_url=fallback_image_url,
                                    timeout_seconds=min(poll_timeout, 60),
                                )
                            
                            status_info = f"即梦: 成功(第{attempt + 1}次尝试); "
                            return hidream_image, hidream_response, "", status_info
                        except Exception as e:
                            status_info = f"即梦: 图片下载失败 - {str(e)}; "
                            hidream_image = _create_fallback_or_white_image_tensor(
                                fallback_image_url=fallback_image_url,
                                timeout_seconds=min(poll_timeout, 60),
                            )
                            return hidream_image, hidream_response, "", status_info
                else:
                    status_info = "即梦: API响应格式错误; "
                    hidream_image = _create_fallback_or_white_image_tensor(
                        fallback_image_url=fallback_image_url,
                        timeout_seconds=min(poll_timeout, 60),
                    )
                    return hidream_image, hidream_response, "", status_info
            else:
                # 这种情况不应该发生，因为已经在前面检查过了，但为了安全起见保留
                error_msg = hidream_result.get("message", "未知错误")
                status_info = f"即梦: API错误 - {error_msg}; "
                hidream_image = _create_fallback_or_white_image_tensor(
                    fallback_image_url=fallback_image_url,
                    timeout_seconds=min(poll_timeout, 60),
                )
                # 如果code不是10000，抛出异常以触发重试
                raise Exception(f"API错误: {error_msg}")
                
        except Exception as e:
            last_error = e
            print(f"TripleAPI: 即梦第{attempt + 1}次尝试失败: {str(e)}")
            if attempt < max_retries - 1:
                print(f"TripleAPI: 即梦将在{retry_interval}秒后重试...")
                time.sleep(retry_interval)  # 固定重试间隔
    
    # 所有重试都失败了
    status_info = f"即梦: 失败(已重试{max_retries}次) - {str(last_error)}; "
    print(f"TripleAPI: 即梦最终失败: {str(last_error)}")
    hidream_image = _create_fallback_or_white_image_tensor(
        fallback_image_url=fallback_image_url,
        timeout_seconds=min(poll_timeout, 60),
    )
    hidream_response = json.dumps({"error": str(last_error), "retries": max_retries}, ensure_ascii=False)
    return hidream_image, hidream_response, "", status_info


class TripleAPIGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                # API网址配置（按顺序：网址在前，KEY在后）
                "nanobanana_server_url": ("STRING", {"default": DEFAULT_SERVER_URL, "multiline": False}),
                "nanobanana_api_key": ("STRING", {"default": "", "multiline": False}),
                "doubao_server_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3/images/generations", "multiline": False}),
                "doubao_api_key": ("STRING", {"default": "", "multiline": False}),
                "hidream_server_url": ("STRING", {"default": "https://visual.volcengineapi.com", "multiline": False}),
                "hidream_req_key": ("STRING", {"default": "", "multiline": False}),
                # 平台开关
                "enable_nanobanana": ("BOOLEAN", {"default": True}),
                "enable_doubao": ("BOOLEAN", {"default": True}),
                "enable_hidream": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 99999999}),
            },
            "optional": {
                # 通用图片输入（统一所有API使用）
                "image_urls_input": ("STRING", {"default": "", "multiline": True}),
                
                # 通用轮询和超时设置
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 60}),
                "poll_timeout": ("INT", {"default": 240, "min": 5, "max": 3600}),
                
                # 错误回退图片（错误时优先返回该URL图片，否则返白图）
                "fallback_image_url": ("STRING", {"default": "", "multiline": False}),
                
                # NanoBanana 参数
                "nanobanana_type": (["TEXTTOIAMGE", "IMAGETOIAMGE"], {"default": "TEXTTOIAMGE"}),
                "nanobanana_num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "nanobanana_image_size_mode": (["auto", "1:1", "9:16", "16:9", "3:4", "4:3", "3:2", "2:3", "5:4", "4:5", "21:9"], {"default": "auto"}),
                "nanobanana_watermark": ("STRING", {"default": ""}),
                
                # 豆包参数
                "doubao_model": ("STRING", {"default": "doubao-seedream-4-0-250828"}),
                "doubao_size": (["1K", "2K", "4K"], {"default": "1K"}),
                "doubao_watermark": ("BOOLEAN", {"default": True}),
                "doubao_type": (["txt2img", "img2img", "multi_img_fusion"], {"default": "txt2img"}),
                "doubao_sequential_generation": (["disabled", "auto"], {"default": "disabled"}),
                "doubao_num_images": ("INT", {"default": 1, "min": 1, "max": 15}),
                
                # 即梦参数
                "hidream_type": (["txt2img", "img2img", "shopImg2img", "vton"], {"default": "txt2img"}),
                "hidream_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
                
                # 重试参数
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 5}),
                "retry_interval": ("INT", {"default": 2, "min": 1, "max": 60}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("nanobanana_image", "doubao_image", "hidream_image", "nanobanana_task_id", "nanobanana_response", "doubao_response", "hidream_response", "hidream_task_id", "status_info", "nanobanana_image_urls", "doubao_image_urls", "hidream_image_urls")
    FUNCTION = "triple_generate"
    CATEGORY = "AIYang_TripleAPI"

    def triple_generate(
        self,
        prompt: str,
        # API网址配置（按顺序：网址在前，KEY在后）
        nanobanana_server_url: str = DEFAULT_SERVER_URL,
        nanobanana_api_key: str = "",
        doubao_server_url: str = "https://ark.cn-beijing.volces.com/api/v3/images/generations",
        doubao_api_key: str = "",
        hidream_server_url: str = "https://visual.volcengineapi.com",
        hidream_req_key: str = "",
        # 平台开关
        enable_nanobanana: bool = True,
        enable_doubao: bool = True,
        enable_hidream: bool = True,
        seed: int = -1,
        # 通用图片输入
        image_urls_input: str = "",
        # 错误回退图片
        fallback_image_url: str = "",
        # 通用轮询和超时设置
        poll_interval: int = 3,
        poll_timeout: int = 240,
        # NanoBanana 参数
        nanobanana_type: str = "TEXTTOIAMGE",
        nanobanana_num_images: int = 1,
        nanobanana_image_size_mode: str = "auto",
        nanobanana_watermark: str = "",
        # 豆包参数
        doubao_model: str = "doubao-seedream-4-0-250828",
        doubao_size: str = "1K",
        doubao_watermark: bool = True,
        doubao_type: str = "txt2img",
        doubao_sequential_generation: str = "disabled",
        doubao_num_images: int = 1,
        # 即梦参数
        hidream_type: str = "txt2img",
        hidream_scale: float = 0.5,
        # 重试参数
        max_retries: int = 3,
        retry_interval: int = 2,
    ) -> Tuple[Any, Any, Any, str, str, str, str, str, str, str, str, str]:
        
        if not prompt:
            raise ValueError("提示词不能为空")
        
        # 限制种子值范围
        if seed is not None and seed > 99999999:
            seed = seed % 100000000
        
        # 初始化结果变量
        nanobanana_image = None
        doubao_image = None
        hidream_image = None
        nanobanana_task_id = ""
        nanobanana_response = ""
        doubao_response = ""
        hidream_response = ""
        hidream_task_id = ""
        status_info = ""
        
        # 初始化图片URL数组变量
        nanobanana_image_urls = "[]"
        doubao_image_urls = "[]"
        hidream_image_urls = "[]"
        
        # 并行调用三个API
        print("TripleAPI: 开始并行调用三个API...")
        
        # 准备并行任务
        tasks = []
        
        # NanoBanana任务
        if enable_nanobanana and nanobanana_api_key:
            tasks.append(('nanobanana', _call_nanobanana_api, {
                'prompt': prompt,
                'nanobanana_type': nanobanana_type,
                'nanobanana_num_images': nanobanana_num_images,
                'nanobanana_image_size_mode': nanobanana_image_size_mode,
                'nanobanana_api_key': nanobanana_api_key,
                'image_urls_input': image_urls_input,
                'seed': seed,
                'nanobanana_watermark': nanobanana_watermark,
                'nanobanana_server_url': nanobanana_server_url,
                'fallback_image_url': fallback_image_url,
                'poll_interval': poll_interval,
                'poll_timeout': poll_timeout,
                'max_retries': max_retries,  # 使用用户配置的重试次数
                'retry_interval': retry_interval,  # 使用用户配置的重试间隔
            }))
        else:
            status_info += "NanoBanana: 已禁用或未提供API密钥; "
            print("TripleAPI: NanoBanana已禁用或未提供API密钥")
            nanobanana_image = _create_fallback_or_white_image_tensor(
                fallback_image_url=fallback_image_url,
                timeout_seconds=min(poll_timeout, 60),
            )
            nanobanana_task_id = ""
            nanobanana_response = json.dumps({"error": "API已禁用或未提供密钥"}, ensure_ascii=False)
            nanobanana_image_urls = json.dumps(["api_disabled"], ensure_ascii=False)
        
        # 豆包任务
        if enable_doubao and doubao_api_key:
            tasks.append(('doubao', _call_doubao_api, {
                'prompt': prompt,
                'doubao_type': doubao_type,
                'doubao_api_key': doubao_api_key,
                'image_urls_input': image_urls_input,
                'seed': seed,
                'doubao_model': doubao_model,
                'doubao_size': doubao_size,
                'doubao_watermark': doubao_watermark,
                'doubao_sequential_generation': doubao_sequential_generation,
                'doubao_num_images': doubao_num_images,
                'fallback_image_url': fallback_image_url,
                'poll_timeout': poll_timeout,
                'max_retries': max_retries,  # 使用用户配置的重试次数
                'retry_interval': retry_interval,  # 使用用户配置的重试间隔
                'doubao_server_url': doubao_server_url,
            }))
        else:
            status_info += "豆包: 已禁用或未提供API密钥; "
            print("TripleAPI: 豆包已禁用或未提供API密钥")
            doubao_image = _create_fallback_or_white_image_tensor(
                fallback_image_url=fallback_image_url,
                timeout_seconds=min(poll_timeout, 60),
            )
            doubao_response = json.dumps({"error": "API已禁用或未提供密钥"}, ensure_ascii=False)
            doubao_image_urls = json.dumps(["api_disabled"], ensure_ascii=False)
        
        # 即梦任务
        if enable_hidream and hidream_req_key:
            tasks.append(('hidream', _call_hidream_api, {
                'prompt': prompt,
                'hidream_type': hidream_type,
                'hidream_req_key': hidream_req_key,
                'image_urls_input': image_urls_input,
                'seed': seed,
                'hidream_scale': hidream_scale,
                'fallback_image_url': fallback_image_url,
                'poll_interval': poll_interval,
                'poll_timeout': poll_timeout,
                'max_retries': max_retries,  # 使用用户配置的重试次数
                'retry_interval': retry_interval,  # 使用用户配置的重试间隔
                'hidream_server_url': hidream_server_url,
            }))
        else:
            status_info += "即梦: 已禁用或未提供API密钥; "
            print("TripleAPI: 即梦已禁用或未提供API密钥")
            hidream_image = _create_fallback_or_white_image_tensor(
                fallback_image_url=fallback_image_url,
                timeout_seconds=min(poll_timeout, 60),
            )
            hidream_response = json.dumps({"error": "API已禁用或未提供密钥"}, ensure_ascii=False)
            hidream_task_id = ""
            hidream_image_urls = json.dumps(["api_disabled"], ensure_ascii=False)
        
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交所有任务
            future_to_api = {
                executor.submit(func, **kwargs): api_name 
                for api_name, func, kwargs in tasks
            }
            
            # 等待所有任务完成并收集结果
            for future in as_completed(future_to_api):
                api_name = future_to_api[future]
                try:
                    result = future.result()
                    
                    if api_name == 'nanobanana':
                        nanobanana_image, nanobanana_task_id, nanobanana_response, api_status = result
                        status_info += api_status
                        # 提取NanoBanana图片URLs
                        try:
                            response_data = json.loads(nanobanana_response)
                            urls = _extract_image_urls_from_response(response_data)
                            if urls:
                                nanobanana_image_urls = json.dumps(urls, ensure_ascii=False)
                                print(f"TripleAPI: NanoBanana提取到{len(urls)}个URL")
                            else:
                                nanobanana_image_urls = json.dumps(["api_error"], ensure_ascii=False)
                        except Exception as e:
                            print(f"TripleAPI: NanoBanana URL提取失败: {str(e)}")
                            nanobanana_image_urls = json.dumps(["api_error"], ensure_ascii=False)
                    elif api_name == 'doubao':
                        doubao_image, doubao_response, api_status = result
                        status_info += api_status
                        # 提取豆包图片URLs
                        try:
                            response_data = json.loads(doubao_response)
                            urls = _extract_image_urls_from_response(response_data)
                            if urls:
                                doubao_image_urls = json.dumps(urls, ensure_ascii=False)
                                print(f"TripleAPI: 豆包提取到{len(urls)}个URL")
                            else:
                                doubao_image_urls = json.dumps(["api_error"], ensure_ascii=False)
                        except Exception as e:
                            print(f"TripleAPI: 豆包URL提取失败: {str(e)}")
                            doubao_image_urls = json.dumps(["api_error"], ensure_ascii=False)
                    elif api_name == 'hidream':
                        hidream_image, hidream_response, hidream_task_id, api_status = result
                        status_info += api_status
                        # 提取即梦图片URLs
                        try:
                            response_data = json.loads(hidream_response)
                            urls = _extract_image_urls_from_response(response_data)
                            if urls:
                                hidream_image_urls = json.dumps(urls, ensure_ascii=False)
                                print(f"TripleAPI: 即梦提取到{len(urls)}个URL")
                            else:
                                hidream_image_urls = json.dumps(["api_error"], ensure_ascii=False)
                        except Exception as e:
                            print(f"TripleAPI: 即梦URL提取失败: {str(e)}")
                            hidream_image_urls = json.dumps(["api_error"], ensure_ascii=False)
                        
                except Exception as e:
                    print(f"TripleAPI: {api_name} 并行执行异常: {str(e)}")
                    if api_name == 'nanobanana':
                        nanobanana_image = _create_fallback_or_white_image_tensor(
                            fallback_image_url=fallback_image_url,
                            timeout_seconds=min(poll_timeout, 60),
                        )
                        nanobanana_task_id = ""
                        nanobanana_response = json.dumps({"error": str(e)}, ensure_ascii=False)
                        status_info += f"NanoBanana: 并行执行异常 - {str(e)}; "
                        nanobanana_image_urls = json.dumps(["api_error"], ensure_ascii=False)
                    elif api_name == 'doubao':
                        doubao_image = _create_fallback_or_white_image_tensor(
                            fallback_image_url=fallback_image_url,
                            timeout_seconds=min(poll_timeout, 60),
                        )
                        doubao_response = json.dumps({"error": str(e)}, ensure_ascii=False)
                        status_info += f"豆包: 并行执行异常 - {str(e)}; "
                        doubao_image_urls = json.dumps(["api_error"], ensure_ascii=False)
                    elif api_name == 'hidream':
                        hidream_image = _create_fallback_or_white_image_tensor(
                            fallback_image_url=fallback_image_url,
                            timeout_seconds=min(poll_timeout, 60),
                        )
                        hidream_task_id = ""
                        hidream_response = json.dumps({"error": str(e)}, ensure_ascii=False)
                        status_info += f"即梦: 并行执行异常 - {str(e)}; "
                        hidream_image_urls = json.dumps(["api_error"], ensure_ascii=False)
        
        print("TripleAPI: 所有API并行执行完成")
        
        # 确保所有图片都不为空
        if nanobanana_image is None:
            print("TripleAPI: NanoBanana图片为空，使用白图占位")
            nanobanana_image = _create_fallback_or_white_image_tensor(
                fallback_image_url=fallback_image_url,
                timeout_seconds=min(poll_timeout, 60),
            )
        if doubao_image is None:
            print("TripleAPI: 豆包图片为空，使用白图占位")
            doubao_image = _create_fallback_or_white_image_tensor(
                fallback_image_url=fallback_image_url,
                timeout_seconds=min(poll_timeout, 60),
            )
        if hidream_image is None:
            print("TripleAPI: 即梦图片为空，使用白图占位")
            hidream_image = _create_fallback_or_white_image_tensor(
                fallback_image_url=fallback_image_url,
                timeout_seconds=min(poll_timeout, 60),
            )
        
        print(f"TripleAPI: 最终状态信息: {status_info.strip()}")
        return (
            nanobanana_image,
            doubao_image,
            hidream_image,
            nanobanana_task_id,
            nanobanana_response,
            doubao_response,
            hidream_response,
            hidream_task_id,
            status_info.strip(),
            nanobanana_image_urls,
            doubao_image_urls,
            hidream_image_urls
        )


NODE_CLASS_MAPPINGS = {
    "NanoBananaGenerate": NanoBananaGenerate,
    "TripleAPIGenerate": TripleAPIGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaGenerate": "NanoBanana Generate",
    "TripleAPIGenerate": "Triple API Generate (NanoBanana + Doubao + HiDream)",
}


