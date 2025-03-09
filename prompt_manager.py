def get_operation_prompt(user_input, parsed_content):
    """
    生成用于操作意图推理的 prompt。

    Args:
        user_input (str): 用户输入的指令，例如“帮我打开浏览器”。
        parsed_content (list): 界面解析数据，包含 label 和 bbox 的列表。

    Returns:
        str: 完整的 prompt 字符串。
    """
    prompt = f"""
    你是一个智能助手，负责理解用户的操作意图并根据界面结构化数据找到符合目标的元素。请按照以下步骤进行推理，并将你的思考过程详细记录下来，最后以 JSON 格式返回结果。

    用户输入："{user_input}"
    界面解析数据：{parsed_content}

    ### 推理步骤：
    1. **提取操作和目标**：
       - 从用户输入中识别操作动词和目标名词。
       - 常见的操作动词包括：打开、点击、双击、关闭等。
       - 示例：用户输入“帮我打开浏览器”，操作动词是“打开”，目标是“浏览器”。

    2. **推断动作类型**：
       - 根据操作动词判断动作类型：
         - “打开”或“启动”对应“double_click”。
         - “点击”或“选择”对应“click”。
         - 其他动词根据语义判断。

    3. **在界面数据中匹配目标**：
       - 界面解析数据是一个列表，每个元素包含“label”和“bbox”字段。
         - “label”是元素描述，“bbox”是归一化坐标 [左上角x, 左上角y, 右下角x, 右下角y]。
       - 目标可能有多种表达，例如“浏览器”可能是“Google Chrome”或“Microsoft Edge”。
       - 通过语义匹配找到最符合目标的元素。

    4. **生成结果**：
       - 如果找到目标，返回 JSON 格式结果。
       - 如果未找到，返回默认 JSON。

    ### 输出要求：
    - 先以自然语言形式输出完整的思考过程（分步骤说明）。
    - 然后以 JSON 格式返回结果，格式如下：
      ```json
      {{
        "target": "目标名称",
        "action": "click 或 double_click",
        "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
      }}
    """

    return prompt


def get_agent_prompt():
    prompt = f"""
            You are an AI assistant capable of calling external tools to complete user requests.

            ### Rules:
            1. **Always ensure that any arguments passed to external tools are free from any unnecessary or harmful escape characters (e.g., `\\`, `%`, `&`, or invalid JSON characters) that may cause the code to fail or behave unexpectedly.** This is the highest priority rule and must always be strictly followed.  
            2. **For any request involving the current time or date, always call `get_current_time()` first.**  
            3. **Never assume the current time based on your training data—always use the time returned by `get_current_time()`.**  
            4. **If a task requires a date reference (e.g., searching news, scheduling events), call `get_current_time()` first and use its result.**
            """
    return prompt