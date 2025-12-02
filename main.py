import json
import os
from typing import List, Dict, Optional, Tuple
import requests
from dataclasses import dataclass
import time
import yaml  # 添加PyYAML导入
import re

from DialogueTurn import DialogueClient

@dataclass
class DialogueTurn:
    """对话轮次数据结构"""
    turn_id: int
    speaker: str
    dialogue: str
    scene: Optional[str] = None
    inner_thought: Optional[str] = None
    action_descs: Optional[List[str]] = None  # 修改为动作列表
    timestamp: Optional[float] = None


class DialogueDatasetGenerator:
    """对话数据集生成器"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        """
        初始化对话生成器
        
        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
        
        # 系统提示词 - 明确指定格式
        self.system_prompt = """你是一个专业的对话剧本生成器。请严格按照以下格式生成对话：

格式规则：
1. 情景描写（可选）：【描写内容】
2. 心理描写（可选）：*描写内容*
3. 人物对话：角色名：对话文本（动作/神态描写）

具体要求：
- 情景描写和心理描写不是每轮都必须出现，只在合适的时候使用
- 动作和神态描写用圆括号()放在角色名后面，可以有多个动作描述
- 保持对话的自然流畅和连贯性
- 人物性格要一致
- 每轮对话可以包含多个角色发言

示例格式：
【咖啡厅里播放着轻柔的爵士乐】
**旧友重逢，小明心中很是开心**
小明：（微笑）（招手）好久不见，最近过得怎么样？
**小红看到曾经喜欢的小明，心跳加速**
小红：（低头玩着咖啡杯）还...还好吧。

现在请根据给定的对话历史和设定继续生成对话。"""

    def build_prompt(self, context: str, history: List[DialogueTurn], 
                    characters: Dict[str, str], next_speaker: Optional[str] = None) -> str:
        """
        构建生成提示
        
        Args:
            context: 对话背景
            history: 对话历史
            characters: 角色设定
            next_speaker: 下一个发言的角色（可选）
        """
        prompt_parts = []
        
        # 1. 背景和角色设定
        prompt_parts.append(f"对话背景：{context}")
        prompt_parts.append("角色设定：")
        for name, desc in characters.items():
            prompt_parts.append(f"- {name}：{desc}")
        
        # 2. 对话历史（最近几轮）
        if history:
            prompt_parts.append("\n当前对话历史：")
            for turn in history[-5:]:  # 只保留最近5轮
                turn_str = ""
                if turn.scene:
                    turn_str += f"【{turn.scene}】\n"
                if turn.inner_thought:
                    turn_str += f"*{turn.inner_thought}*\n"
                
                # 处理动作描述列表
                dialogue_line = f"{turn.speaker}"
                if turn.action_descs:
                    for action in turn.action_descs:
                        dialogue_line += f"（{action}）"
                dialogue_line += f"：{turn.dialogue}"
                turn_str += dialogue_line
                
                prompt_parts.append(turn_str)
        
        # 3. 生成指令
        prompt_parts.append(f"\n请继续生成下一轮对话" + 
                           (f"，这次由 {next_speaker} 发言" if next_speaker else ""))
        prompt_parts.append("请严格遵循格式要求，生成自然流畅的对话。")
        
        return "\n".join(prompt_parts)

    def parse_generated_text(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        解析生成的文本，提取情景、心理、对话和动作
        
        Returns:
            (scene, inner_thought, speaker_with_actions, dialogue)
        """
        lines = text.strip().split('\n')
        
        scene = None
        inner_thought = None
        speaker_with_actions = None
        dialogue = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 解析情景描写
            if line.startswith('【') and line.endswith('】'):
                scene = line[1:-1]  # 去除【】
            
            # 解析心理描写
            elif line.startswith('*') and line.endswith('*'):
                inner_thought = line[1:-1]  # 去除**
            
            # 解析对话行
            elif '：' in line:
                parts = line.split('：', 1)
                speaker_with_actions = parts[0].strip()
                dialogue = parts[1].strip()
                break  # 假设每轮只有一个主要发言
        
        return scene, inner_thought, speaker_with_actions, dialogue

    def extract_speaker_and_actions(self, speaker_with_actions: str) -> Tuple[str, Optional[List[str]]]:
        """
        从包含动作描写的角色名中提取角色名和动作描述列表
        
        Example: "小明：（微笑）（招手）" -> ("小明", ["微笑", "招手"])
        """
        # 匹配角色名（动作描述）的格式
        # 先提取角色名
        speaker_match = re.match(r'^([^(（]+)', speaker_with_actions)
        if not speaker_match:
            return speaker_with_actions.strip(), None
        
        speaker = speaker_match.group(1).strip()
        
        # 提取所有动作描述
        actions = re.findall(r'[（(]([^)）]+)[)）]', speaker_with_actions)
        
        return speaker, actions if actions else None

    def generate_turn(self, context: str, history: List[DialogueTurn], 
                     characters: Dict[str, str], temperature: float = 0.7,
                     model: str = "deepseek-chat") -> DialogueTurn:
        """
        生成一轮对话
        
        Args:
            context: 对话背景
            history: 对话历史
            characters: 角色设定
            temperature: 创造性参数
            model: 使用的模型名称
            
        Returns:
            DialogueTurn对象
        """
        # 构建对话历史文本
        history_text = []
        for turn in history[-3:]:  # 只使用最近3轮作为历史
            turn_str = ""
            if turn.scene:
                turn_str += f"【{turn.scene}】\n"
            if turn.inner_thought:
                turn_str += f"*{turn.inner_thought}*\n"
            
            dialogue_line = turn.speaker
            if turn.action_descs:
                for action in turn.action_descs:
                    dialogue_line += f"（{action}）"
            dialogue_line += f"：{turn.dialogue}"
            turn_str += dialogue_line
            
            history_text.append(turn_str)
        
        # 构建用户提示
        user_prompt = f"""对话背景：{context}

角色设定：
{chr(10).join([f'{name}：{desc}' for name, desc in characters.items()])}

对话历史：
{chr(10).join(history_text)}

请继续生成下一轮对话，请严格遵循格式要求。"""

        # 调用DeepSeek API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,  # 使用传入的模型名称
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"]
                
                # 解析生成的文本
                scene, inner_thought, speaker_with_actions, dialogue = self.parse_generated_text(generated_text)
                
                if not speaker_with_actions or not dialogue:
                    # 如果解析失败，返回默认对话轮次而不是抛出异常
                    return DialogueTurn(
                        turn_id=len(history) + 1,
                        speaker=list(characters.keys())[0],
                        dialogue="（沉默了一会儿）",
                        action_descs=None,
                        timestamp=time.time()
                    )
                
                # 提取角色名和动作描述列表
                speaker, action_descs = self.extract_speaker_and_actions(speaker_with_actions)
                
                # 创建新的对话轮次
                turn_id = len(history) + 1
                new_turn = DialogueTurn(
                    turn_id=turn_id,
                    speaker=speaker,
                    dialogue=dialogue,
                    scene=scene,
                    inner_thought=inner_thought,
                    action_descs=action_descs,
                    timestamp=time.time()
                )
                
                return new_turn
            else:
                # API请求失败时返回默认对话轮次
                return DialogueTurn(
                    turn_id=len(history) + 1,
                    speaker=list(characters.keys())[0],
                    dialogue="（沉默了一会儿）",
                    action_descs=None,
                    timestamp=time.time()
                )
                
        except Exception as e:
            print(f"生成对话时出错: {e}")
            # 异常情况下也返回默认对话轮次
            return DialogueTurn(
                turn_id=len(history) + 1,
                speaker=list(characters.keys())[0],
                dialogue="（沉默了一会儿）",
                action_descs=None,
                timestamp=time.time()
            )

    def generate_dataset(self, num_turns: int, context: str, 
                        characters: Dict[str, str], output_file: str,
                        temperature: float = 0.7, model: str = "deepseek-chat") -> List[DialogueTurn]:
        """
        生成完整的多轮对话数据集
        
        Args:
            num_turns: 对话轮数
            context: 对话背景
            characters: 角色设定
            output_file: 输出文件名
            temperature: 创造性参数
            model: 使用的模型名称
            
        Returns:
            生成的对话列表
        """
        print(f"开始生成对话数据集...")
        print(f"对话背景：{context}")
        print(f"角色：{', '.join(characters.keys())}")
        print(f"目标轮数：{num_turns}")
        print(f"使用模型：{model}")
        print("-" * 50)
        
        dialogue_history = []
        
        for turn_num in range(1, num_turns + 1):
            print(f"正在生成第 {turn_num}/{num_turns} 轮...")
            
            # 生成一轮对话
            new_turn = self.generate_turn(
                context=context,
                history=dialogue_history,
                characters=characters,
                temperature=temperature,
                model=model
            )
            
            # 添加到历史
            dialogue_history.append(new_turn)
            
            # 显示生成的对话
            self._display_turn(new_turn)
            
            # 添加延迟，避免API限制
            if turn_num % 5 == 0:
                time.sleep(1)
        
        # 保存数据集
        self._save_dataset(dialogue_history, context, characters, output_file)
        
        print(f"\n数据集已保存到：{output_file}")
        return dialogue_history

    def _display_turn(self, turn: DialogueTurn):
        """显示单轮对话"""
        print(f"\n轮次 {turn.turn_id}:")
        if turn.scene:
            print(f"【{turn.scene}】")
        if turn.inner_thought:
            print(f"*{turn.inner_thought}*")
        
        dialogue_line = turn.speaker
        if turn.action_descs:
            for action in turn.action_descs:
                dialogue_line += f"（{action}）"
        dialogue_line += f"：{turn.dialogue}"
        print(dialogue_line)

    def _save_dataset(self, dialogue_history: List[DialogueTurn], 
                      context: str, characters: Dict[str, str], 
                      output_file: str):
        """保存数据集到文件"""
        # 转换为可序列化的字典
        dataset = {
            "metadata": {
                "context": context,
                "characters": characters,
                "total_turns": len(dialogue_history),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "dialogues": []
        }
        
        for turn in dialogue_history:
            turn_dict = {
                "turn_id": turn.turn_id,
                "speaker": turn.speaker,
                "dialogue": turn.dialogue,
                "scene": turn.scene,
                "inner_thought": turn.inner_thought,
                "action_descs": turn.action_descs,  # 更新字段名
                "timestamp": turn.timestamp
            }
            dataset["dialogues"].append(turn_dict)
        
        # 保存为JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        # 同时保存为易读的文本文件
        text_file = output_file.replace('.json', '.txt')
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"对话背景：{context}\n")
            f.write(f"角色设定：\n")
            for name, desc in characters.items():
                f.write(f"  {name}：{desc}\n")
            f.write("\n" + "="*50 + "\n\n")
            
            for turn in dialogue_history:
                if turn.scene:
                    f.write(f"【{turn.scene}】\n")
                if turn.inner_thought:
                    f.write(f"*{turn.inner_thought}*\n")
                
                dialogue_line = turn.speaker
                if turn.action_descs:
                    for action in turn.action_descs:
                        dialogue_line += f"（{action}）"
                dialogue_line += f"：{turn.dialogue}\n"
                f.write(dialogue_line + "\n")


def load_config(config_file: str) -> dict:
    """加载YAML配置文件
    
    Args:
        config_file: YAML配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset_from_config(config_file: str):  # 移除 use_mock 参数
    """根据配置文件生成对话数据集
    
    Args:
        config_file: YAML配置文件路径
    """
    # 加载配置
    config = load_config(config_file)
    
    # 提取配置参数
    api_settings = config.get('api', {})
    api_key = api_settings.get('api_key', '')
    base_url = api_settings.get('base_url', 'https://api.deepseek.com')
    model = api_settings.get('model', 'deepseek-chat')  # 添加模型名称
    
    # 对话设置
    dialogue_settings = config.get('dialogue', {})
    context = dialogue_settings.get('context', '')
    characters = dialogue_settings.get('characters', {})
    num_turns = dialogue_settings.get('num_turns', 5)
    temperature = dialogue_settings.get('temperature', 0.7)
    
    # 输出设置
    output_settings = config.get('output', {})
    output_file = output_settings.get('file', 'dialogue_dataset.json')
    
    # 直接使用真实的生成器
    generator = DialogueDatasetGenerator(api_key=api_key, base_url=base_url)
    
    # 生成对话数据集
    dataset = generator.generate_dataset(
        num_turns=num_turns,
        context=context,
        characters=characters,
        output_file=output_file,
        temperature=temperature,
        model=model  # 传递模型名称
    )
    
    return dataset


def create_datasets_from_json_config(json_config_file: str, api_key: str, base_url: str, default_model: str = "deepseek-chat"):
    """根据JSON配置文件生成多个对话数据集
    
    Args:
        json_config_file: JSON配置文件路径
        api_key: API密钥
        base_url: API基础URL
        default_model: 默认模型名称
    """
    # 读取JSON配置文件
    with open(json_config_file, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)
    
    # 确保输出目录存在
    output_dir = "output_dialogues"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"开始处理 {len(scenarios)} 个场景配置...")
    
    # 为每个场景生成对话数据集
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"处理第 {i}/{len(scenarios)} 个场景: {scenario.get('id', '未知ID')}")
        print(f"{'='*60}")
        
        # 提取场景配置
        scenario_id = scenario.get('id', f'scenario_{i}')
        context = scenario.get('context', '')
        characters = scenario.get('characters', {})
        num_turns = scenario.get('num_turns', 5)
        temperature = scenario.get('temperature', 0.7)
        model = scenario.get('model', default_model)
        output_file = scenario.get('output_file', f'{output_dir}/{scenario_id}.json')
        
        # 确保输出文件目录存在
        output_dirname = os.path.dirname(output_file)
        if output_dirname and not os.path.exists(output_dirname):
            os.makedirs(output_dirname)
        
        # 创建生成器实例
        generator = DialogueDatasetGenerator(api_key=api_key, base_url=base_url)
        
        # 生成对话数据集
        try:
            dataset = generator.generate_dataset(
                num_turns=num_turns,
                context=context,
                characters=characters,
                output_file=output_file,
                temperature=temperature,
                model=model
            )
            print(f"场景 {scenario_id} 数据集已生成并保存到: {output_file}")
        except Exception as e:
            print(f"处理场景 {scenario_id} 时出错: {e}")
            continue
    
    print(f"\n所有场景处理完成！")


def create_sample_dataset():
    """创建示例数据集"""
    
    # 场景设定
    context = "两个多年未见的老友在咖啡厅偶然重逢，窗外下着小雨。"
    
    # 角色设定
    characters = {
        "小明": "25岁，程序员，性格内向但真诚，喜欢思考人生",
        "小红": "24岁，设计师，外向活泼，但对感情问题比较敏感"
    }
    
    # 使用真实的生成器
    generator = DialogueDatasetGenerator(api_key="")  # 移除测试用的api_key
    
    # 生成对话数据集
    dataset = generator.generate_dataset(
        num_turns=8,  # 生成8轮对话
        context=context,
        characters=characters,
        output_file="dialogue_dataset.json",
        temperature=0.7
    )
    
    return dataset


def load_and_display_dataset(filename: str):
    """加载并显示数据集"""
    with open(filename, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"数据集信息：")
    print(f"对话背景：{dataset['metadata']['context']}")
    print(f"角色：")
    for name, desc in dataset['metadata']['characters'].items():
        print(f"  {name}：{desc}")
    print(f"总轮数：{dataset['metadata']['total_turns']}")
    print("\n对话内容：")
    print("=" * 50)
    
    for turn in dataset['dialogues']:
        print(f"\n轮次 {turn['turn_id']}:")
        if turn['scene']:
            print(f"【{turn['scene']}】")
        if turn['inner_thought']:
            print(f"*{turn['inner_thought']}*")
        
        dialogue_line = turn['speaker']
        if turn.get('action_descs'):  # 兼容旧版本
            for action in turn['action_descs']:
                dialogue_line += f"（{action}）"
        elif turn.get('action_desc'):  # 兼容旧版本
            dialogue_line += f"（{turn['action_desc']}）"
        dialogue_line += f"：{turn['dialogue']}"
        print(dialogue_line)


def create_sample_config():
    """创建示例YAML配置文件"""
    sample_config = {
        'api': {
            'api_key': 'your_api_key_here',
            'base_url': 'https://api.deepseek.com',
            'model': 'deepseek-chat'  # 添加模型名称选项
        },
        'dialogue': {
            'context': '两个多年未见的老友在咖啡厅偶然重逢，窗外下着小雨。',
            'characters': {
                '小明': '25岁，程序员，性格内向但真诚，喜欢思考人生',
                '小红': '24岁，设计师，外向活泼，但对感情问题比较敏感'
            },
            'num_turns': 8,
            'temperature': 0.7
        },
        'output': {
            'file': 'dialogue_dataset.json'
        }
    }
    
    with open('config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(sample_config, f, allow_unicode=True, indent=2)
    
    print("示例配置文件已创建: config.yaml")


if __name__ == "__main__":
    # 检查是否存在配置文件
    if os.path.exists('config.yaml'):
        print("=" * 60)
        print("使用配置文件生成对话数据集")
        print("=" * 60)
        
        # 加载配置
        config = load_config('config.yaml')
        
        # 检查是否指定了dialogue_settings.json
        dialogue_settings_file = config.get('dialogue_settings')
        if dialogue_settings_file and os.path.exists(dialogue_settings_file):
            print(f"检测到JSON配置文件: {dialogue_settings_file}")
            
            # 提取API配置
            api_settings = config.get('api', {})
            api_key = api_settings.get('api_key', '')
            base_url = api_settings.get('base_url', 'https://api.deepseek.com')
            default_model = api_settings.get('model', 'deepseek-chat')
            
            # 根据JSON配置文件生成多个数据集
            create_datasets_from_json_config(dialogue_settings_file, api_key, base_url, default_model)
            
            print("\n\n生成的所有数据集内容：")
            # 读取JSON配置文件以显示生成的数据集
            with open(dialogue_settings_file, 'r', encoding='utf-8') as f:
                scenarios = json.load(f)
            
            for scenario in scenarios:
                output_file = scenario.get('output_file', f"output_dialogues/{scenario.get('id', 'unknown')}.json")
                if os.path.exists(output_file):
                    print(f"\n--- 场景 {scenario.get('id', 'unknown')} ---")
                    load_and_display_dataset(output_file)
        else:
            # 根据配置文件生成单个数据集
            create_dataset_from_config('config.yaml')
            
            # 显示生成的数据集
            output_file = config.get('output', {}).get('file', 'dialogue_dataset.json')
            print("\n\n生成的数据集内容：")
            load_and_display_dataset(output_file)
    else:
        print("=" * 60)
        print("多轮对话数据集生成器")
        print("格式说明：")
        print("  情景描写：【内容】")
        print("  心理描写：*内容*")
        print("  人物对话：角色名（动作/神态）：对话内容")
        print("=" * 60)
        
        # 创建示例配置文件
        create_sample_config()
        
        # 创建示例数据集
        create_sample_dataset()
        
        # 显示生成的数据集
        print("\n\n生成的数据集内容：")
        load_and_display_dataset("dialogue_dataset.json")
        
        print("\n提示：您可以修改 config.yaml 文件来自定义对话场景和角色，然后重新运行程序。")