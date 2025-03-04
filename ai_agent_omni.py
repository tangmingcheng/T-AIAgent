import time
from PIL import Image
import pyautogui
import tkinter as tk
import threading
from omniparser_function import click_bbox, process_image
from query_llm import query_deepseek


def get_dpi():
    """获取系统 DPI"""
    root = tk.Tk()
    dpi = root.winfo_fpixels('1i')
    root.destroy()
    print(f"系统 DPI: {dpi}")
    return dpi


def capture_screen(image_path):
    """截取当前屏幕并保存为图像文件"""
    dpi = get_dpi()
    screenshot = pyautogui.screenshot()

    if dpi != 96:
        scale_factor = dpi / 96
        new_width = int(screenshot.width * scale_factor)
        new_height = int(screenshot.height * scale_factor)
        screenshot = screenshot.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # 保存截图
        threading.Thread(target=save_screenshot, args=(screenshot, image_path)).start()


def save_screenshot(screenshot, image_path):
    screenshot.save(image_path)
    print(f"屏幕截图已保存至: {image_path}")


def execute_ui(user_input: str, image_path: str = "D:\\WallPaper\\test.png"):
    # 1. 截屏
    #capture_screen(image_path)

    # 2. 调用 omniparser 处理截图
    print("等待 omniparser 解析...")
    time.sleep(5)
    parsed_result = process_image(
        image_path=image_path,
        box_threshold=0.05,
        iou_threshold=0.1,
        use_paddleocr=True,
        imgsz=640
    )

    print(f"omniparser result:", parsed_result)

    if parsed_result['status'] != 'success':
        print("Omniparser 解析失败:", parsed_result['message'])
        return 'Failed to execute the operation.'

    # 3. 结合用户输入，调用 deepseek-r1 推理
    print("调用 deepseek-r1 获取目标信息...")
    print(user_input)
    target_data = query_deepseek(user_input, parsed_result['parsed_content'])

    if target_data:
        print("AI 推理目标:", target_data)
        # 提取 bbox 字段
        bbox = target_data['bbox']
        click_bbox(bbox)
        return 'Success to execute the operation'
    else:
        print("未能解析到有效的目标数据")
        return 'Failed to execute the operation due to the target was not found.'


if __name__ == "__main__":
    user_input = input("请输入操作指令（例如，点击回收站图标）：")
    result = execute_ui(user_input)
    print(result)
