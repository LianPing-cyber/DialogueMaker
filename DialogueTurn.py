import json
import requests
import time
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


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


class DialogueClient:
    """对话生成客户端"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
        """
        初始化对话生成客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
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
            for turn in history[-10:]:  # 只保留最近10轮
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
                     characters: Dict[str, str], temperature: float = 0.7) -> DialogueTurn:
        """
        生成一轮对话
        
        Args:
            context: 对话背景
            history: 对话历史
            characters: 角色设定
            temperature: 创造性参数
            
        Returns:
            DialogueTurn对象
        """
        # 构建对话历史文本
        history_text = []
        for turn in history[-10:]:  # 只使用最近10轮作为历史
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

请生成下一轮对话，严格遵循以下格式：
1. 情景描写（可选）：【内容】
2. 心理描写（可选）：*内容*
3. 人物对话：角色名：（动作/神态）对话内容

请确保对话自然连贯，符合角色性格。"""

        # API请求
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_new_tokens": 512,
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