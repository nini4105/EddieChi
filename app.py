import re
import torch
import sympy as sp
import gradio as gr
from ctransformers import AutoModelForCausalLM
from huggingface_hub import login

#你的 Hugging Face API Token（如果需要驗證）
HUGGING_FACE_API_KEY = "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"
login(token=HUGGING_FACE_API_KEY)

#使用 GGML 量化的 Llama 2 7B（4-bit 量化），適合 CPU
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGML",
    model_type="llama",
    gpu_layers=0  #GitHub Codespaces 沒有 GPU，請設為 0
)

def classify_math_problem(problem):
    """使用 Llama 2 進行數學問題分類"""
    print(f"正在使用 Llama 2 分析問題類型，請稍候...")

    prompt = f"""
    You are a **strict keyword classification system**. Your task is to **only classify** the given text based on the following keywords:
    
    - If the text contains **"derivative"**, respond **only** with: `Derivative`
    - If the text contains **"integral"**, respond **only** with: `Integral`
    - If the text contains **"limit"**, respond **only** with: `Limit`
    
    **STRICT RULES:**
    - **Do NOT explain.**
    - **Do NOT add extra words or formatting.**
    - **Your response must be exactly one of the following: `Derivative`, `Integral`, or `Limit`.**
    
    **Text to classify:**
    ```{problem}```

    **Output:**
    """

    #讓 LLM 進行分類
    result = model(prompt).strip()

    #Debugging：顯示完整輸出
    print(f"DEBUG - LLM 完整輸出：\n{result}")

    #確保只返回三種可能的答案
    valid_types = ["Derivative", "Integral", "Limit"]
    for t in valid_types:
        if t.lower() in result.lower():
            return t.lower()

    return "unknown"

def extract_math_expression(user_input):
    """從使用者輸入中提取純數學表達式與極限值"""
    print(f"DEBUG - 原始輸入：{user_input}")

    #嘗試提取 `as x approaches N` 或 `x → N`
    x_value = None
    limit_match = re.search(r"(?:as x approaches|x\s*→)\s*(-?\d+(\.\d+)?)", user_input, re.IGNORECASE)
    if limit_match:
        x_value = float(limit_match.group(1))  # 轉換為數字
        user_input = re.sub(r"(?:as x approaches|x\s*→)\s*(-?\d+(\.\d+)?)", "", user_input, flags=re.IGNORECASE).strip()

    #只保留數學符號與變數
    user_input = re.sub(r"(find|the|derivative|integral|limit|of|compute|as|approaches)", "", user_input, flags=re.IGNORECASE).strip()
    
    #**處理商數 (Fraction) 表達式，確保 `/` 轉換為 SymPy 可解析格式**
    user_input = user_input.replace("^", "**").replace("/", " / ")

    #確保數字與變數之間有 `*`（例如 `2x` 轉換成 `2*x`）
    user_input = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", user_input)

    math_expr = user_input.strip()
    print(f"DEBUG - 修正後的數學表達式：{math_expr}")
    print(f"檢測到 x → {x_value}" if x_value is not None else "無檢測到趨近值")

    return math_expr, x_value

def solve_calculus(problem_type, expression, x_value=None):
    """使用 SymPy 解決微積分問題"""
    x = sp.Symbol('x')

    try:
        if problem_type == "derivative":
            result = sp.diff(expression, x)
        elif problem_type == "integral":
            result = sp.integrate(expression, x)
        elif problem_type == "limit":
            if x_value is not None:
                print(f"DEBUG - 計算極限表達式：{expression}, x → {x_value}")
                result = sp.limit(expression, x, x_value)
                print(f"SymPy 成功計算極限，結果為：{result}")
            else:
                return "無法解析極限，請提供 x 的趨近值"
        else:
            return "無法識別的問題類型"
        
        return result
    except Exception as e:
        print(f"SymPy 計算失敗，錯誤訊息：{e}")
        return f"錯誤：{str(e)}"

def process_question(user_input):
    """處理使用者輸入的數學問題"""
    problem_type = classify_math_problem(user_input)
    print(f"直接偵測關鍵字：{problem_type}")

    if problem_type == "unknown":
        return "無法識別問題類型，請輸入有效的數學問題"

    #提取數學表達式 & 檢測 `x` 趨近值
    math_expr, x_value = extract_math_expression(user_input)
    print(f"提取數學表達式：{math_expr}")

    #讓 SymPy 解決數學問題
    try:
        expr = sp.sympify(math_expr, evaluate=False)
        solution = solve_calculus(problem_type, expr, x_value)
        return f"計算結果：{solution}"
    except Exception as e:
        return f"解析錯誤：{e}"

def main():
    print("=== 微積分 AI 助手（使用 Llama 2 GGML） ===")

    #測試 SymPy 是否能解析數學表達式
    sympy_test_expr = "x**3 + 2*x**2 + x"
    try:
        expr = sp.sympify(sympy_test_expr, evaluate=False)
        print(f"SymPy 測試成功：{expr}")
    except Exception as e:
        print(f"SymPy 測試失敗：{e}")

    #**Gradio 介面**
    gr.Interface(
        fn=process_question,
        inputs=gr.Textbox(label="輸入你的數學問題"),
        outputs=gr.Textbox(label="計算結果"),
        title="微積分 AI 助手",
        description="輸入數學問題，例如：\n- Compute the derivative of x^3 + 2x^2 + x\n- Compute the integral of 3x^2 + 4x + 1\n- Compute the limit of (x^2 - 1)/(x - 1) as x approaches 1"
    ).launch(share=True)

if __name__ == "__main__":
    main()
