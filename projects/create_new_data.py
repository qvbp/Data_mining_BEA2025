# import json
# import uuid
# import random
# import torch
# from typing import List, Dict, Any
# from tqdm import tqdm
# import time
# import os
# import sys

# # 避免Flash Attention问题
# sys.modules['flash_attn'] = None
# sys.modules['flash_attn.flash_attn_interface'] = None
# sys.modules['flash_attn.bert_padding'] = None

# # 导入需要的库
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# def generate_unique_id() -> str:
#     """生成唯一的对话ID"""
#     return f"{random.randint(1, 5000)}-{uuid.uuid4()}"

# def generate_tutoring_data_prompt() -> str:
#     """生成用于创建单个数学教学对话数据的提示"""
#     prompt = """
#     Please generate a realistic mathematics tutoring conversation in the following format:
#     {
#         "conversation_history": "Tutor: Hi, could you please provide a step-by-step solution for the question below? The question is: [INSERT MATH PROBLEM HERE] \n Student: [STUDENT'S SOLUTION WITH POSSIBLE ERRORS] \n [STUDENT'S ANSWER] \n Tutor: [FOLLOW-UP QUESTION OR COMMENT] \n Student: [STUDENT'S RESPONSE] \n Tutor: [ANOTHER FOLLOW-UP QUESTION OR GUIDANCE] \n Student: [STUDENT'S FINAL RESPONSE]"
#     }

#     Requirements:
#     1. Create realistic mathematics problems suitable for students in grades 5-12
#     2. Topics can include arithmetic, algebra, geometry, word problems, or basic probability
#     3. The student's solution should sometimes contain common calculation or conceptual errors
#     4. The tutor should ask questions to guide the student toward discovering errors or improving their understanding
#     5. Keep the dialogue natural and educational
#     6. Make sure the conversation flows logically
#     7. The problem should be challenging but appropriate for the implied age level
#     8. Ensure the generated JSON format is correct and can be parsed
#     9. Include a variety of difficulty levels and problem types across generations
#     10. Make some conversations shorter and some longer to reflect real tutoring scenarios

#     Here are examples of the kind of conversations to generate:

#     Example 1:
#     "conversation_history": "Tutor: Hi, could you please provide a step-by-step solution for the question below? The question is: Yulia was able to net $44 in profit this week. Her lemonade stand made a gross revenue of $47 this week. She was also able to babysit and made $31. However, her lemonade stand is expensive to run because she needs to buy lemons, sugar, and sunscreen. How much did she spend to operate her lemonade stand, in dollars? \n Student: Yulia's total earnings from both her lemonade stand and babysitting are $44 + $47 + $31 = $122.\nLet x be the amount she spent to run her lemonade stand.\nThen her profit is $47 - x.\nAnd we know that her total profit is $44.\nSo we can set up an equation: $47 - x = $44\nSolving for x, we get: x = $3\nTherefore, Yulia spent $3 to operate her lemonade stand. \n 3 \n Tutor: tell me step by step how you worked this out please? \n Student: First, I added up Yulia's total earnings from her lemonade stand and babysitting, which was $44 + $47 + $31 = $122. Then, I let x be the amount she spent to run her lemonade stand. After that, I figured out her profit was $47 - x. Since we know her total profit was $44, I set up an equation: $47 - x = $44. Finally, I solved for x and got x = $3. So, Yulia spent $3 to operate her lemonade stand."

#     Example 2:
#     "conversation_history": "Tutor: Hi, could you please provide a step-by-step solution for the question below? The question is: A small store made 50 posters to advertise their products. Two-fifths of them are small posters while half of them are medium posters. The rest are large posters. How many large posters are there? \n Student: Two-fifths of the posters are small, which means 3/5 of them are either medium or large.\nLet x be the total number of medium and large posters.\nSo 2/5x is the number of small posters.\nAnd 2/5x + 50 = x is the total number of posters.\nSolving for x, we get x = 80.\nTherefore, there are 80 - 50 = 30 large posters.\n 30 \n Tutor: You have the first part correct at 20 \n Student: Yes, I got that right. But then I got confused about the total number of posters and I ended up with the wrong answer. \n Tutor: So half of 50 are medium added to 20 \n Student: Yes, that's right. So the total number of posters would be 70 and the number of large posters would be 70 - 50 = 20."

#     Example 3:
#     "conversation_history": "Tutor: Look at the board. \n Student: okayyy \n Tutor: What is 12+3=? \n Student: the Ar is 13"
    
#     Example 4:
#     "conversation_history": "Tutor: Hi! Let's work on solving quadratic equations today. Can you solve x² - 5x + 6 = 0? \n Student: I'll use the quadratic formula: x = (-b ± √(b² - 4ac))/2a. Here, a=1, b=-5, and c=6. So x = (5 ± √(25 - 24))/2 = (5 ± √1)/2 = (5 ± 1)/2. Therefore, x = 3 or x = 2. \n Tutor: Great job! Can you also solve this by factoring? \n Student: Let me try. I need to find two numbers that multiply to 6 and add to -5. Those would be -2 and -3. So x² - 5x + 6 = (x-2)(x-3) = 0, which means x = 2 or x = 3. \n Tutor: Excellent! You've shown two different methods to solve the same problem."

#     Please generate one complete data entry as a valid JSON object with a unique and realistic mathematics tutoring conversation.
#     """
#     return prompt



# def load_llama_model_8bit(model_path="Llama-2-7b-chat-hf", gpu_id=1):
#     """使用8位量化加载LLaMA模型，减少内存使用"""
#     print(f"Loading model from {model_path} to GPU #{gpu_id}...")
    
#     # 检查GPU可用性和内存状态
#     if not torch.cuda.is_available():
#         print("CUDA不可用，将使用CPU")
#         device = "cpu"
#     else:
#         # 获取可用GPU数量
#         gpu_count = torch.cuda.device_count()
#         print(f"系统中可见的GPU数量: {gpu_count}")
        
#         # 打印每个GPU的内存使用情况
#         for i in range(gpu_count):
#             free_mem, total_mem = torch.cuda.mem_get_info(i)
#             free_mem_gb = free_mem / (1024**3)
#             total_mem_gb = total_mem / (1024**3)
#             print(f"GPU #{i}: 可用内存 {free_mem_gb:.2f} GB / 总内存 {total_mem_gb:.2f} GB")
        
#         if gpu_id >= gpu_count:
#             print(f"警告：指定的GPU #{gpu_id}不存在，共有{gpu_count}个GPU")
#             print(f"将使用GPU #0")
#             gpu_id = 0
        
#         # 清理指定GPU的缓存
#         torch.cuda.empty_cache()
        
#         device = f"cuda:{gpu_id}"
#         print(f"Using GPU #{gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
#     # 加载分词器
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
    
#     # 对于7B模型，可以尝试直接使用FP16加载，通常内存足够
#     if "7b" in model_path.lower() or "7-b" in model_path.lower():
#         try:
#             print("检测到7B模型，尝试直接以FP16加载...")
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 torch_dtype=torch.float16,
#                 device_map={"": gpu_id}
#             )
#             print("模型已成功加载(FP16)")
#             return model, tokenizer, device
#         except Exception as e:
#             print(f"FP16加载失败，将尝试8位量化: {e}")
#             # 如果失败，继续尝试8位量化
    
#     # 配置8位量化
#     quantization_config = BitsAndBytesConfig(
#         load_in_8bit=True,
#         llm_int8_threshold=6.0,  # 默认阈值
#         llm_int8_has_fp16_weight=False,
#         bnb_4bit_compute_dtype=torch.float16
#     )
    
#     try:
#         # 使用8位量化加载模型
#         print("使用8位量化加载模型...")
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             quantization_config=quantization_config,
#             device_map={"": gpu_id}  # 将所有模块映射到指定GPU
#         )
#         print("模型已成功加载(8位量化)")
#     except Exception as e:
#         print(f"8位量化加载失败: {e}")
        
#         # 尝试安装bitsandbytes
#         try:
#             print("尝试安装bitsandbytes...")
#             import subprocess
#             subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes>=0.37.0", "--quiet"])
            
#             # 重新尝试加载
#             print("重新尝试加载模型...")
#             quantization_config = BitsAndBytesConfig(
#                 load_in_8bit=True
#             )
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 quantization_config=quantization_config,
#                 device_map={"": gpu_id}
#             )
#             print("模型已成功加载(8位量化)")
#         except Exception as e2:
#             print(f"安装bitsandbytes并重新加载失败: {e2}")
            
#             # 如果都失败了，尝试使用普通的FP16加载
#             print("尝试使用普通FP16加载...")
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path, 
#                 torch_dtype=torch.float16
#             )
#             model = model.to(device)
#             print("模型以FP16精度加载")
    
#     return model, tokenizer, device

# def generate_with_llama(model, tokenizer, prompt, device, max_new_tokens=2000):
#     """使用LLaMA模型生成回应"""
#     # 检测模型类型并使用适当的提示格式
#     model_name = getattr(model, "name_or_path", "").lower()
#     if not model_name:
#         model_name = tokenizer.name_or_path.lower()
    
#     # 为LLaMA-2-Chat或LLaMA-3格式化提示
#     system_prompt = "You are a helpful, harmless, and precise AI assistant that generates high-quality educational dialogue data."
    
#     if "llama-3" in model_name or "llama3" in model_name:
#         # Llama-3格式
#         formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
#         print("使用Llama-3提示格式")
#     else:
#         # Llama-2格式
#         formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
#         print("使用Llama-2提示格式")
    
#     # 编码提示并移至指定设备
#     inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
#     # 生成回应
#     with torch.no_grad():
#         output = model.generate(
#             inputs.input_ids,
#             attention_mask=inputs.attention_mask,
#             max_new_tokens=max_new_tokens,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id
#         )
    
#     # 解码并返回仅包含新生成的token
#     response_text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
#     return response_text.strip()

# def extract_json_from_text(text):
#     """从文本中提取JSON对象"""
#     # 尝试直接解析整个文本
#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         pass
    
#     # 尝试找到文本中的JSON对象
#     try:
#         # 寻找第一个{和最后一个}之间的内容
#         start_idx = text.find('{')
#         end_idx = text.rfind('}') + 1
        
#         if start_idx != -1 and end_idx != 0:
#             json_str = text[start_idx:end_idx]
#             return json.loads(json_str)
#     except (json.JSONDecodeError, ValueError):
#         pass
    
#     # 如果标准方法都失败，尝试更复杂的JSON修复（针对小错误）
#     try:
#         if start_idx != -1 and end_idx != 0:
#             json_str = text[start_idx:end_idx]
            
#             # 常见错误修复: 单引号替换为双引号
#             json_str = json_str.replace("'", '"')
            
#             # 尝试修复未转义的引号
#             json_str = json_str.replace('\\"', '"').replace('\\"', '"')
            
#             return json.loads(json_str)
#     except (json.JSONDecodeError, ValueError):
#         pass
    
#     # 最后返回None表示无法提取JSON
#     return None

# def save_generated_data(data: List[Dict[str, Any]], output_file: str = "generated_tutoring_data.json"):
#     """保存生成的数据到JSON文件"""
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)
#     print(f"数据已保存到 {output_file}")

# def main(num_samples: int = 100, model_path: str = "Llama-2-7b-chat-hf", gpu_id: int = 1):
#     """主函数，生成并保存数据"""
#     # 打印设备信息，帮助调试
#     print("PyTorch版本:", torch.__version__)
#     print("CUDA是否可用:", torch.cuda.is_available())
#     print("可见GPU数量:", torch.cuda.device_count())
#     for i in range(torch.cuda.device_count()):
#         print(f"GPU #{i}: {torch.cuda.get_device_name(i)}")
    
#     # 显示正在使用的模型信息
#     model_name = os.path.basename(model_path)
#     print(f"使用模型: {model_name}")
    
#     # 清理显存
#     torch.cuda.empty_cache()
    
#     # 加载模型到指定GPU
#     model, tokenizer, device = load_llama_model_8bit(model_path, gpu_id)
    
#     # 对于7B模型可以使用更高的批处理大小
#     batch_size = 1
#     if "7b" in model_path.lower() or "7-b" in model_path.lower():
#         batch_size = 1  # 仍然使用1，但对于7B模型，您可以考虑增加到2
    
#     # 准备生成数据
#     all_data = []
#     success_count = 0
    
#     # 创建输出目录（如果不存在）
#     output_dir = "generated_data_only_conversations"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 设置文件用于保存失败的生成结果
#     failed_file = os.path.join(output_dir, f"failed_generations_{model_name}.txt")
#     with open(failed_file, 'w', encoding='utf-8') as f:
#         f.write(f"# 记录使用 {model_name} 的失败生成结果\n\n")
    
#     # 设置临时文件保存每次成功的生成
#     temp_file = os.path.join(output_dir, f"temp_generated_data_{model_name}.json")
    
#     # 使用tqdm显示进度
#     for i in tqdm(range(num_samples), desc=f"Generating samples using {model_name}"):
#         # 生成提示
#         prompt = generate_tutoring_data_prompt()
        
#         try:
#             # 使用模型生成回应
#             response = generate_with_llama(model, tokenizer, prompt, device)
            
#             # 尝试解析生成的JSON
#             data_entry = extract_json_from_text(response)
            
#             if data_entry:
#                 # 替换对话ID
#                 data_entry["conversation_id"] = generate_unique_id()
#                 all_data.append(data_entry)
#                 success_count += 1
                
#                 # 每次成功后保存临时文件
#                 with open(temp_file, 'w', encoding='utf-8') as f:
#                     json.dump(all_data, f, ensure_ascii=False, indent=4)
                
#                 print(f"Successfully generated sample {success_count}/{num_samples}")
#             else:
#                 # 记录失败的生成结果
#                 with open(failed_file, 'a', encoding='utf-8') as f:
#                     f.write(f"## Generation {i+1}\n\n```\n{response}\n```\n\n")
#                 print(f"Failed to parse JSON from generation {i+1}")
            
#             # 添加一些延迟以避免过热
#             # 7B模型处理更快，可以减少延迟
#             if "7b" in model_path.lower() or "7-b" in model_path.lower():
#                 time.sleep(1)  # 对于7B模型减少延迟
#             else:
#                 time.sleep(2)
            
#         except Exception as e:
#             print(f"Error during generation {i+1}: {e}")
#             # 记录错误
#             with open(failed_file, 'a', encoding='utf-8') as f:
#                 f.write(f"## Error in Generation {i+1}\n\n```\n{str(e)}\n```\n\n")
            
#             # 添加一些延迟以冷却
#             time.sleep(3)
    
#     # 保存所有成功的数据
#     if all_data:
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         output_file = os.path.join(output_dir, f"generated_tutoring_data_{model_name}_{success_count}samples_{timestamp}.json")
#         save_generated_data(all_data, output_file)
#         print(f"成功生成 {success_count}/{num_samples} 条数据")
#         print(f"成功率: {success_count/num_samples*100:.2f}%")
#     else:
#         print("没有成功生成任何数据")

# if __name__ == "__main__":
#     # 模型路径和GPU ID设置
#     model_path = "/mnt/cfs/huangzhiwei/models/Llama-2-7b-chat-hf"  # 使用完整路径
#     gpu_id = 1  # 使用1号GPU
    
#     # 设置生成样本数量
#     num_samples = 100
    
#     # 记录开始时间
#     start_time = time.time()
    
#     try:
#         main(num_samples=num_samples, model_path=model_path, gpu_id=gpu_id)
#     except KeyboardInterrupt:
#         print("\n程序被用户中断")
#     except Exception as e:
#         print(f"\n程序发生错误: {e}")
#     finally:
#         # 计算总运行时间
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         hours, remainder = divmod(elapsed_time, 3600)
#         minutes, seconds = divmod(remainder, 60)
#         print(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")




import json
import uuid
import random
import torch
from typing import List, Dict, Any
from tqdm import tqdm
import time
import os
import sys
import re

# 避免Flash Attention问题
sys.modules['flash_attn'] = None
sys.modules['flash_attn.flash_attn_interface'] = None
sys.modules['flash_attn.bert_padding'] = None

# 导入需要的库
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def generate_unique_id() -> str:
    """生成唯一的对话ID"""
    return f"{random.randint(1, 5000)}-{uuid.uuid4()}"

def generate_tutoring_data_prompt() -> str:
    """生成用于创建单个数学教学对话数据的提示"""
    prompt = """
    Please generate a realistic mathematics tutoring conversation in EXACTLY the following JSON format:
    {
        "conversation_id": "327-4c3b8d62-3b0b-4b8e-8b0a-7b3b8b62aa3b",
        "conversation_history": "Tutor: [QUESTION] \n Student: [ANSWER-WITH-ERROR] \n Tutor: [FOLLOW-UP] \n Student: [RESPONSE-MAINTAINS-ERROR]"
    }

    CORE REQUIREMENTS:
    1. TUTOR PROTOCOL (MUST IMPLEMENT ALL):
       A) Variable Clarification:
          - Ask direct questions about key values 
          → "How many pounds per serving?"
          → "What's the initial quantity?"
       B) Step-by-Step Verification:
          - Demand process explanations
          → "Walk me through your calculation"
          → "Show each operation separately"
       C) Calculation Basis Check:
          - Verify percentage/base values
          → "What base value are you using for this 20%?"
          → "Is this increase applied to original or previous result?"
       D) Error Discovery Guidance:
          - Use targeted questioning
          → "What was the first harvest yield?"
          → "How does this compare to initial assumptions?"
       E) Component Separation:
          - Require individual calculations
          → "Calculate meat and cheese costs separately"
          → "Show area and perimeter computations dividedly"
       F) Conceptual Verification:
          - Probe understanding through "why" questions
          → "Why is 4.8 not rounded to 5?"
          → "Explain why we use πr² instead of 2πr"
          
    2. STUDENT ERROR REQUIREMENTS:
       ■ Every student response MUST contain ≥1 errors from:
         - Arithmetic miscalculations
         - Base value confusion
         - Formula misapplication
         - Unit conversion errors
       ■ Errors must persist through dialogue
       
    3. Additional Requirements:
       - Create realistic mathematics problems suitable for students in grades 5-12
       - Topics can include arithmetic, algebra, geometry, word problems, or basic probability
       - Keep the dialogue natural and educational
       - Make sure the conversation flows logically
       - The problem should be challenging but appropriate for the implied age level
       - VERY IMPORTANT: Ensure the generated JSON format is EXACTLY as shown above
       - The entire conversation should be a single string value for the "conversation_history" key
       - Include variety in problem difficulty and types

    Example of the EXACT JSON format to generate:

    Example 1:
    {
        "conversation_id": "478-5d2f8a1c-f190-4b8e-a54d-985fqwe789d",
        "conversation_history": "Tutor: Calculate 18% tip on $45 meal\nStudent: 45 × 0.18 = 8.1 → $53.1 total\nTutor: Show each calculation step\nStudent: Meal:45 × 1.18 = 53.1\nTutor: Break down the components\nStudent: 45 + (45×0.18) = 45+8.1=53.1\nTutor: Verify percentage base\nStudent: 18% is always from original price"
    }

    Example 2:
    {
        "conversation_id": "327-4c3b8d62-3b0b-4b8e-8b0a-7b3b8b62aa3b",
        "conversation_history": "Tutor: Hi, could you please provide a step-by-step solution for the question below? The question is: Yulia was able to net $44 in profit this week. Her lemonade stand made a gross revenue of $47 this week. She was also able to babysit and made $31. However, her lemonade stand is expensive to run because she needs to buy lemons, sugar, and sunscreen. How much did she spend to operate her lemonade stand, in dollars? \n Student: Yulia's total earnings from both her lemonade stand and babysitting are $44 + $47 + $31 = $122.\nLet x be the amount she spent to run her lemonade stand.\nThen her profit is $47 - x.\nAnd we know that her total profit is $44.\nSo we can set up an equation: $47 - x = $44\nSolving for x, we get: x = $3\nTherefore, Yulia spent $3 to operate her lemonade stand. \n Tutor: Tell me step by step how you worked this out please? \n Student: First, I added up Yulia's total earnings from her lemonade stand and babysitting, which was $44 + $47 + $31 = $122. Then, I let x be the amount she spent to run her lemonade stand. After that, I figured out her profit was $47 - x. Since we know her total profit was $44, I set up an equation: $47 - x = $44. Finally, I solved for x and got x = $3. So, Yulia spent $3 to operate her lemonade stand."
    }
    
    Example 3:
    {
        "conversation_id": "598-9e3c7a5d-2c8b-4f6a-b31e-42d8a7c90f2e",
        "conversation_history": "Tutor: Hi, could you please provide a step-by-step solution for the question below? The question is: A small store made 50 posters to advertise their products. Two-fifths of them are small posters while half of them are medium posters. The rest are large posters. How many large posters are there? \n Student: Two-fifths of the posters are small, which means 3/5 of them are either medium or large.\nLet x be the total number of medium and large posters.\nSo 2/5x is the number of small posters.\nAnd 2/5x + 50 = x is the total number of posters.\nSolving for x, we get x = 80.\nTherefore, there are 80 - 50 = 30 large posters.\n \n Tutor: You have the first part correct at 20 small posters. Let's verify this together. How many small posters are there? \n Student: Yes, two-fifths of 50 is 20 small posters. But then I got confused about the total number of posters and I ended up with the wrong answer. \n Tutor: So if half of the 50 posters are medium, how many medium posters are there? \n Student: Half of 50 is 25 medium posters. So we have 20 small and 25 medium, which gives us 45 posters. But the problem says there are 50 posters total. So there must be 5 large posters."
    }

    Please generate one complete data entry as a valid JSON object with a unique and realistic mathematics tutoring conversation that follows ALL the requirements. DO NOT include any explanations or additional text outside the JSON object.
    """
    return prompt


def load_llama_model_8bit(model_path="Llama-2-7b-chat-hf", gpu_id=1):
    """使用8位量化加载LLaMA模型，减少内存使用"""
    print(f"Loading model from {model_path} to GPU #{gpu_id}...")
    
    # 检查GPU可用性和内存状态
    if not torch.cuda.is_available():
        print("CUDA不可用，将使用CPU")
        device = "cpu"
    else:
        # 获取可用GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"系统中可见的GPU数量: {gpu_count}")
        
        # 打印每个GPU的内存使用情况
        for i in range(gpu_count):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            free_mem_gb = free_mem / (1024**3)
            total_mem_gb = total_mem / (1024**3)
            print(f"GPU #{i}: 可用内存 {free_mem_gb:.2f} GB / 总内存 {total_mem_gb:.2f} GB")
        
        if gpu_id >= gpu_count:
            print(f"警告：指定的GPU #{gpu_id}不存在，共有{gpu_count}个GPU")
            print(f"将使用GPU #0")
            gpu_id = 0
        
        # 清理指定GPU的缓存
        torch.cuda.empty_cache()
        
        device = f"cuda:{gpu_id}"
        print(f"Using GPU #{gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # 使用上下文管理器确保在指定设备上操作
    with torch.cuda.device(device):
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 对于7B模型，可以尝试直接使用FP16加载，通常内存足够
        if "7b" in model_path.lower() or "7-b" in model_path.lower():
            try:
                print("检测到7B模型，尝试直接以FP16加载...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map={"": gpu_id}
                )
                print("模型已成功加载(FP16)")
                return model, tokenizer, device
            except Exception as e:
                print(f"FP16加载失败，将尝试8位量化: {e}")
                # 如果失败，继续尝试8位量化
        
        # 配置8位量化
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  # 默认阈值
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        try:
            # 使用8位量化加载模型
            print("使用8位量化加载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map={"": gpu_id}  # 将所有模块映射到指定GPU
            )
            print("模型已成功加载(8位量化)")
        except Exception as e:
            print(f"8位量化加载失败: {e}")
            
            # 尝试安装bitsandbytes
            try:
                print("尝试安装bitsandbytes...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes>=0.37.0", "--quiet"])
                
                # 重新尝试加载
                print("重新尝试加载模型...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map={"": gpu_id}
                )
                print("模型已成功加载(8位量化)")
            except Exception as e2:
                print(f"安装bitsandbytes并重新加载失败: {e2}")
                
                # 如果都失败了，尝试使用普通的FP16加载
                print("尝试使用普通FP16加载...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16
                )
                model = model.to(device)
                print("模型以FP16精度加载")
    
    return model, tokenizer, device

def generate_with_llama(model, tokenizer, prompt, device, max_new_tokens=2000):
    """使用LLaMA模型生成回应，更加严格的参数控制"""
    # 检测模型类型并使用适当的提示格式
    model_name = getattr(model, "name_or_path", "").lower()
    if not model_name:
        model_name = tokenizer.name_or_path.lower()
    
    # 为LLaMA-2-Chat或LLaMA-3格式化提示
    system_prompt = "You are a helpful, precise AI assistant that generates valid JSON data exactly as requested. Always follow the format examples exactly."
    
    if "llama-3" in model_name or "llama3" in model_name:
        # Llama-3格式
        formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
        print("使用Llama-3提示格式")
    else:
        # Llama-2格式
        formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        print("使用Llama-2提示格式")
    
    # 编码提示并移至指定设备
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # 生成回应 - 降低温度和top_p使生成更加确定性
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.5,  # 降低温度以获得更确定性的输出
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.2,  # 添加重复惩罚
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码并返回仅包含新生成的token
    response_text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 清除可能的前后缀解释文本
    response_text = response_text.strip()
    
    # 如果回应不以'{'开头，尝试找到第一个'{'
    if not response_text.strip().startswith('{'):
        json_start = response_text.find('{')
        if json_start > -1:
            response_text = response_text[json_start:]
    
    # 如果回应不以'}'结尾，尝试找到最后一个'}'
    if not response_text.strip().endswith('}'):
        json_end = response_text.rfind('}')
        if json_end > -1:
            response_text = response_text[:json_end+1]
    
    return response_text

def extract_json_from_text(text):
    """从文本中提取JSON对象，增强版"""
    # 尝试直接解析整个文本
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 使用正则表达式找到最外层的花括号对
    json_pattern = r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
    matches = re.findall(json_pattern, text)
    
    for potential_json in matches:
        try:
            data = json.loads(potential_json)
            # 验证数据格式是否符合要求
            if "conversation_history" in data:
                # 如果conversation_history是嵌套的对象数组，而不是字符串
                if isinstance(data["conversation_history"], list):
                    # 将多个对话历史合并为一个字符串
                    combined_history = " ".join([str(item) for item in data["conversation_history"]])
                    return {"conversation_history": combined_history}
                return data
        except json.JSONDecodeError:
            continue
    
    # 尝试手动修复常见JSON错误
    try:
        # 寻找第一个{和最后一个}之间的内容
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = text[start_idx:end_idx]
            
            # 修复常见错误: 单引号替换为双引号，并修复未转义的引号
            json_str = json_str.replace("'", '"').replace('\\"', '"')
            
            # 处理可能多层嵌套的conversation_history
            if '"conversation_history": [' in json_str:
                simplified_str = json_str[:json_str.find('"conversation_history": [')] + \
                                '"conversation_history": "' + \
                                text.replace('"', '\\"').replace('\n', '\\n') + '"}'
                return json.loads(simplified_str)
            
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 如果所有尝试都失败，尝试创建一个基本对象
    try:
        # 提取可能的对话内容
        cleaned_text = text.replace('\n', ' ').replace('"', '\\"')
        # 只保留对话部分，删除元数据说明
        if "Tutor:" in cleaned_text:
            dialogue_start = cleaned_text.find("Tutor:")
            cleaned_text = cleaned_text[dialogue_start:].strip()
            return {"conversation_history": cleaned_text}
    except:
        pass
    
    # 最后返回None表示无法提取JSON
    return None

def validate_conversation(data_entry):
    """验证生成的对话数据是否符合要求"""
    if not isinstance(data_entry, dict):
        return False
    
    if "conversation_history" not in data_entry:
        return False
    
    history = data_entry["conversation_history"]
    if not isinstance(history, str):
        return False
    
    # 验证对话格式：至少包含一个问题和回答
    if "Tutor:" not in history or "Student:" not in history:
        return False
    
    # 验证对话长度：至少100个字符
    if len(history) < 100:
        return False
    
    return True

def save_generated_data(data: List[Dict[str, Any]], output_file: str = "generated_tutoring_data.json"):
    """保存生成的数据到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"数据已保存到 {output_file}")

def main(num_samples: int = 100, model_path: str = "Llama-2-7b-chat-hf", gpu_id: int = 1):
    """主函数，生成并保存数据"""
    # 打印设备信息，帮助调试
    print("PyTorch版本:", torch.__version__)
    print("CUDA是否可用:", torch.cuda.is_available())
    print("可见GPU数量:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU #{i}: {torch.cuda.get_device_name(i)}")
    
    # 显示正在使用的模型信息
    model_name = os.path.basename(model_path)
    print(f"使用模型: {model_name}")
    
    # 清理显存
    torch.cuda.empty_cache()
    
    # 加载模型到指定GPU
    model, tokenizer, device = load_llama_model_8bit(model_path, gpu_id)
    
    # 对于7B模型可以使用更高的批处理大小
    batch_size = 1
    if "7b" in model_path.lower() or "7-b" in model_path.lower():
        batch_size = 1  # 仍然使用1，但对于7B模型，您可以考虑增加到2
    
    # 准备生成数据
    all_data = []
    success_count = 0
    
    # 创建输出目录（如果不存在）
    output_dir = "generated_data_only_conversations_error"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置文件用于保存失败的生成结果
    failed_file = os.path.join(output_dir, f"failed_generations_{model_name}.txt")
    with open(failed_file, 'w', encoding='utf-8') as f:
        f.write(f"# 记录使用 {model_name} 的失败生成结果\n\n")
    
    # 设置临时文件保存每次成功的生成
    temp_file = os.path.join(output_dir, f"temp_generated_data_{model_name}.json")
    
    # 尝试从临时文件恢复状态
    start_idx = 0
    if os.path.exists(temp_file):
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
                start_idx = len(all_data)
                success_count = start_idx
                print(f"从第 {start_idx+1} 个样本恢复生成")
        except json.JSONDecodeError:
            print("无法解析临时文件，将重新开始")
    
    # 使用tqdm显示进度
    for i in tqdm(range(start_idx, num_samples), desc=f"Generating samples using {model_name}"):
        # 生成提示
        prompt = generate_tutoring_data_prompt()
        
        try:
            # 使用模型生成回应
            response = generate_with_llama(model, tokenizer, prompt, device)
            
            # 尝试解析生成的JSON
            data_entry = extract_json_from_text(response)
            
            # 验证数据是否有效
            if data_entry and validate_conversation(data_entry):
                # 替换对话ID
                data_entry["conversation_id"] = generate_unique_id()
                all_data.append(data_entry)
                success_count += 1
                
                # 每次成功后保存临时文件
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=4)
                
                print(f"Successfully generated sample {success_count}/{num_samples}")
            else:
                # 记录失败的生成结果
                with open(failed_file, 'a', encoding='utf-8') as f:
                    f.write(f"## Generation {i+1}\n\n```\n{response}\n```\n\n")
                print(f"Failed to parse JSON from generation {i+1}")
                
                # 尝试保存原始响应，可能有助于调试
                raw_file = os.path.join(output_dir, f"raw_response_{i+1}.txt")
                with open(raw_file, 'w', encoding='utf-8') as f:
                    f.write(response)
            
            # 添加一些延迟以避免过热
            # 7B模型处理更快，可以减少延迟
            if "7b" in model_path.lower() or "7-b" in model_path.lower():
                time.sleep(1)  # 对于7B模型减少延迟
            else:
                time.sleep(2)
            
        except Exception as e:
            print(f"Error during generation {i+1}: {e}")
            # 记录错误
            with open(failed_file, 'a', encoding='utf-8') as f:
                f.write(f"## Error in Generation {i+1}\n\n```\n{str(e)}\n```\n\n")
            
            # 添加一些延迟以冷却
            time.sleep(3)
    
    # 保存所有成功的数据
    if all_data:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"generated_tutoring_data_{model_name}_{success_count}samples_{timestamp}.json")
        save_generated_data(all_data, output_file)
        print(f"成功生成 {success_count}/{num_samples} 条数据")
        print(f"成功率: {success_count/num_samples*100:.2f}%")
    else:
        print("没有成功生成任何数据")

if __name__ == "__main__":
    # 模型路径和GPU ID设置
    model_path = "/mnt/cfs/huangzhiwei/models/Llama-2-7b-chat-hf"  # 使用完整路径
    gpu_id = 1  # 使用1号GPU
    
    # 设置生成样本数量
    num_samples = 100
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        main(num_samples=num_samples, model_path=model_path, gpu_id=gpu_id)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序发生错误: {e}")
    finally:
        # 计算总运行时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
