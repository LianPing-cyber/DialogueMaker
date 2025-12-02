# DialogueMaker
Generating dialogues using api. Support customizing characters, dialogue background, multiple characters, etc. parameters. Can generate action descriptions, psychological descriptions and scene descriptions.
使用api生成多轮角色对话，支持自定义角色、对话背景、多个角色等参数。能够生成动作描写、心理描写和场景描写。

## 特点
除了传统的对话内容以外，还会生成动作描写、心理描写和场景描写。
模板：
```
【场景描写】
*心理描写*
人物：xxx（动作）xxxx（动作）
```
也就是说，场景描写和心理描写是在对话开始前生成的，而人物描写是穿插在说话过程中的。

程序会自动帮你处理生成内容，整理成json格式，方便后续使用，你可以在yaml中指定输出位置。

## 使用方法
Set your api key and base url in config.yaml
Set each dialogue settings in dialogue_settings.json
