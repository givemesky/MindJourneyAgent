# -*- coding: utf-8 -*-
import os
import sys

# 自然语言到符号逻辑的翻译（NL2SL）：这是该文件的核心任务。它将模糊的文本（如“预算5000元”、“不去爬山”）转化为结构化的 Python 逻辑表达式或 JSON 约束。
#project_root_path = os.path.dirname(
#    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#)
#if project_root_path not in sys.path:
#    sys.path.insert(0, project_root_path)
#if os.path.join(project_root_path, "chinatravel") not in sys.path:
#    sys.path.insert(0, os.path.join(project_root_path, "chinatravel"))
# print(project_root_path)
project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)
import json
from tqdm import tqdm
# from chinatravel.agent.llms import Deepseek
# from chinatravel.data.load_datasets import save_json_file, load_json_file
# from chinatravel.agent.nesy_agent.prompts import NL2SL_INSTRUCTION_V2

from perception.llms import Deepseek
from infra.data.load_datasets import save_json_file, load_json_file
from perception.nesy_agent.prompts.prompts import NL2SL_INSTRUCTION_V2

# 上下文提示词工程（Prompt Engineering）：代码内置了详细的函数文档（func_docs），告知 LLM 哪些符号函数（如 day_count, budget_constraint）是可用的，以及它们的返回类型和用法。
func_docs = """
(1) day_count(plan)
Docs: Get the number of days in the plan.
Return: int
(2) people_count(plan)
Docs: Get the number of people in the plan.
Return: int
(3) target_city(plan)
Docs: Get the target city of the plan.
Return: str
(4) allactivities(plan)
Docs: Get all the activities in the plan.
Return: list of activities
(5) allactivities_count(plan)
Docs: Get the number of activities in the plan.
Return: int
(6) dayactivities(plan, day)
Docs: Get all the activities in the specific day [1, 2, 3, ...].
Return: list of activities
(7) activity_cost(activity)
Docs: Get the cost of specific activity without transport cost.
Return: float
(8) activity_position(activity)
Docs: Get the position name of specific activity.
Return: str
(9) activity_price(activity)
Docs: Get the price of specific activity. The price is price per person.
Return: float
(10) activity_type(activity)
Docs: Get the type of specific activity. ['breakfast', 'lunch', 'dinner', 'attraction', 'accommodation', 'train', 'airplane']
Return: str
(11) activity_tickets(activity)
Docs: Get the number of tickets needed for specific activity. ['attraction', 'train', 'airplane']
Return: int
(12) activity_transports(activity)
Docs: Get the transport information of specific activity.
Return: list of dict
(13) activity_start_time(activity)
Docs: Get the start time of specific activity.
Return: str
(14) activity_end_time(activity)
Docs: Get the end time of specific activity.
Return: str
(15) activity_time(activity)
Docs: Get the duration of specific activity.
Return: int (minutes)
(16) innercity_transport_cost(transports)
Docs: Get the total cost of innercity transport.
Return: float
(17) poi_recommend_time(city, poi):
Docs: Get the recommend time of specific poi in the city. Only support attractions now.
Return: int (minutes)
(18) poi_distance(city, poi1, poi2):
Docs: Get the distance between two pois in the city.
Return: float (km)
(19) innercity_transport_price(transports)
Docs: Get the price of innercity transport. The price is price per person.
Return: float
(20) innercity_transport_distance(transports)
Docs: Get the distance of innercity transport.
Return: float (km)
(21) metro_tickets(transports)
Docs: Get the number of metro tickets if the type of transport is metro.
Return: int
(22) taxi_cars(transports)
Docs: Get the number of taxi cars if the type of transport is taxi. The number of taxi cars is `(people_count(plan) + 3) // 4`.
Return: int
(23) room_count(activity)
Docs: Get the number of rooms of accommodation activity.
Return: int
(24) room_type(activity)
Docs: Get the type of room of accommodation activity.
1 for single room, 2 for double room. Must be 1 or 2. Never use "大床房" or "双床房" or other words but 1 or 2.
Return: int
(25) restaurant_type(activity, target_city)
Docs: Get the type of restaurant's cuisine in the target city. The return value must be in ['云南菜', '西藏菜', '东北菜', '烧烤', '亚洲菜', '粤菜', '西北菜', '闽菜', '客家菜', '快餐简餐', '川菜', '台湾菜', '其他', '清真菜', '小吃', '西餐', '素食', '日本料理', '江浙菜', '湖北菜', '东南亚菜', '湘菜', '北京菜', '韩国料理', '海鲜', '中东料理', '融合菜', '茶馆/茶室', '酒吧/酒馆', '创意菜', '自助餐', '咖啡店', '本帮菜', '徽菜', '拉美料理', '鲁菜', '新疆菜', '农家菜', '海南菜', '火锅', '面包甜点', '其他中餐'].
Return: str
(26) attraction_type(activity, target_city)
Docs: Get the type of attraction in the target city. The return value must be in ['博物馆/纪念馆', '美术馆/艺术馆', '红色景点', '自然风光', '人文景观', '大学校园', '历史古迹', '游乐园/体育娱乐', '图书馆', '园林', '其它', '文化旅游区', '公园', '商业街区'].
Return: str
(27) accommodation_type(activity, target_city)
Docs: Get the feature of accommodation in the target city to judge whether it's feature meets the user's requirement. The return value must be in ['儿童俱乐部', '空气净化器', '山景房', '私汤房', '四合院', '温泉', '湖畔美居', '电竞酒店', '温泉泡汤', '行政酒廊', '充电桩', '设计师酒店', '民宿', '湖景房', '动人夜景', '行李寄存', '中式庭院', '桌球室', '私人泳池', '钓鱼', '迷人海景', '园林建筑', '老洋房', '儿童泳池', '历史名宅', '棋牌室', '智能客控', '情侣房', '小而美', '特色 住宿', '茶室', '亲子主题房', '多功能厅', '洗衣房', '客栈', '自营亲子房', '停车场', 'Boss推荐', '江河景房', '日光浴场', '自营影音房', '厨房', '空调', '网红泳池', '别墅', '免费停车', '洗衣服务', '窗外好景', '酒店公寓', '会议厅', '家庭房', '24小时前台', '商务中心', '提前入园', '农家乐', '智能马桶', '美食酒店', 'SPA', '拍照出片', '海景房', '泳池', '影音房', '管家服务', '穿梭机场班车', '桑拿', '机器人服务', '儿童乐园', '健身室', '洗衣机', '自营舒睡房', '宠物友好', '电竞房', '位置超好', '套房'].
Return: str
(28) innercity_transport_type(transports)
Docs: Get the type of innercity transport. The return value must be in ['metro', 'taxi', 'walk'].
Return: str
(29) innercity_transport_start_time(transports)
Docs: Get the start time of innercity transport.
Return: str
(30) innercity_transport_end_time(transports)
Docs: Get the end time of innercity transport.
Return: str
"""

nl_trans_prompt = (
    """
You are a travel agent. A customer is asking you to help them book a trip. Try to understand the customer's request and extract the nature language logic constraints in the request.
Besides `Start City`, `Target City`, `People Count`, `Day Count`. There are some other constraints about `Innercity Transport Type`, `InterCity Transport Type`, `Attraction`, `Accommodation`, `Restaurant`, `Time`(start time and end time or other time constraints), `Budget`, 
`Cost` on specific activity, `Position` of specific activity, `Type` of specific activity, `Num`of something. You can create some constraints that are not listed here as long as they are reasonable and mentioned in user's request. The are also some Implicit constraints that are not mentioned in user's request but can be inferred from the request. For example, if there are many people in the request, it is reasonable to infer that the user may want to book multiple rooms.
Combine constraints is also allowed. For example, "The cost of xxx activity  should be less than 1000 and the position of xxx activity should be in the city center" as long as they are mentioned in user's request.

### MANDATORY RULES FOR ATTRACTIONS (景点强制规则) ###
1. If the user specifies a particular attraction or a clear type of interest:
   - Output: "Exist Attraction which Position == 'NAME' and Type == 'CATEGORY'"
2. If the user mentions "attractions" or "spots" generally WITHOUT specifying names or types:
   - Output: "Exist Attraction which Type == '其它'" 
3. The 'Type' MUST be selected from this specific list: 
   ['游乐园/体育娱乐', '历史古迹', '博物馆/纪念馆', '文化旅游区', '公园', '人文景观', '商业街区', '红色景点', '美术馆/艺术馆', '图书馆', '大学校园', '自然风光','其它'].
4. If the user mentions "attractions" but doesn't specify a type, you MUST use '其它' as the Type instead of listing all categories.
5. DO NOT output multiple categories in a list like ["Type A", "Type B"] unless the user explicitly requested multiple types. 
6. DO NOT dump the entire candidate list. If you are uncertain, use '其它'.
####################################################

we offer some functions below, which can help you extract the nature language logic constraints in the user's request.
variables:
(1) plan: a dict of the generated plan with information of the specific plan.

functions:"""
    + func_docs
    + """!!!Implicit Constraints should also be extracted!!! Try use `All`, `For each`, `Exist`, `For some`... to express the constraints, use "==", "<=", "<", ">=", ">" to express the relationship between variables.
Output it in the following format:

<constraint>
1. [constraint 1]
2. [constraint 2]
3. ...
<end>

!!!You must obey all the rules we offer above, return type, the list of choices, the relationship between variables, etc.!!!

for example:
Query 1: 当前位置广州。我们三个人想坐飞机去成都玩3天，只打车或地铁或步行，住成都明悦大酒店，都开双人间，请给我们一个旅行规划，想去都江堰，还想看看自然风景，希望酒店带泳池。
Nature Language Logic Constraints:
<constraint>
1. Start City == "广州"
2. Target City == "成都"
3. People Count == 3
4. Day Count == 3
5. All Accommodation Position == "成都明悦大酒店"
6. Number of beds per room/Room Type == 2
7. Tickets == People Count = 3
8. Metro Tickets == People Count = 3
9. Cars == (People Count + 3) // 4 = 1
10. Exist "都江堰" in Attraction Position and Type == "自然风光"
11. Exist Attraction which Type == "自然风光"
12. All InterCity Transport Type == "airplane"
13. All Innercity Transport set <= {"taxi", "metro", "walk"}
14. Exist Accommodation which Type == "泳池"
<end>

Query 2: 我想多去几个景点看看，行程排满一点，行程不要太赶。
Nature Language Logic Constraints:
<constraint>
1. Exist Attraction which Type == "其它"
2. All Activities count >= 5
3. All Activities' activity_time() <= 90
<end>

DO NOT MAKE ANY ASSUMPTIONS. Follow the user's request strictly!!!
DO NOT FORGET Tickets and Metro Tickets
Please extract the nature language logic constraints in the user's request:
"""
)

sl_trans_prompt = (
    """
We offer some functions below, try to translate the constraints in nature language into python code and output them in json format.
variables:
(1) plan: a dict of the generated plan with information of the specific plan.

functions:"""
    + func_docs
    + """
You need to response in the following format:
{
    "start_city": str,
    "target_city": str,
    "days": int,
    "people_number": int,
    "hard_logic_py":[
        [python code 1],
        [python code 2],
        ...
    ]
}
Not all the constraints need to be translated into python code. Ignore them if they can not be translated into legal python code.
!!! Only `plan` variable can be used directly in the python code. Others must be defined in the python code use the functions we offer above. !!! Pay attention to the return TYPE of functions!!!
For most case, for exist constraints, you can set `result=False` at the beginning of the code, and then set `result=True` if the condition is satisfied. For all constraints, you can set `result=True` at the beginning of the code, and then set `result=False` if the condition is not satisfied.

### Attention!!!
If you find some pesucode in the nature language constraints is not defined in the functions we offer above, you must translate them into python block code with the functions we offer above.
###

Example:
nature_language:
当前位置上海。我和朋友计划去北京玩三天，预算6000元，市内交通只使用地铁，开一间单床房，吃南门涮肉(天坛店)。请给我一个旅行规划，希望酒店带泳池。
1. Start City == "上海"
2. Target City == "北京"
3. Day Count == 3
4. People Count == 2
5. Total Cost <= 6000
6. Tickets == 2
7. Taxi Cars == 1
8. Room Count == 1
9. Room Type == 1
10. Innercity Transport Type == "metro"
11. Metro Tickets == 2
12. Innercity Transport Set <= {"metro"}
13. Exist "南门涮肉(天坛店)" in Restaurant Position
answer:
{
"start_city": "上海",
"target_city": "北京",
"days": 3,
"people_number": 2,
"hard_logic_py": ["result=(day_count(plan)==3)","result=(people_count(plan)==2)","total_cost=0 \nfor activity in allactivities(plan): total_cost+=activity_cost(activity)+innercity_transport_cost(activity_transports(activity))\nresult=(total_cost<=6000)","result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!=2: result=False\n  if innercity_transport_type(activity_transports(activity))=='metro'and metro_tickets(activity_transports(activity))!=2: result=False","result=True\nfor activity in allactivities(plan):\n  if innercity_transport_type(activity_transports(activity))=='taxi'and taxi_cars(activity_transports(activity))!=1: result=False","result=True\nfor activity in allactivities(plan):\n  if activity_type(activity)=='accommodation' and room_count(activity)!=1: result=False\n  if activity_type(activity)=='accommodation' and room_type(activity)!=1: result=False\n  if activity_type(activity)=='accommodation' and accommodation_type(activity, target_city(plan))!='泳池': result=False","innercity_transport_set=set()\nfor activity in allactivities(plan):\n  if activity_transports(activity)!=[]: innercity_transport_set.add(innercity_transport_type(activity_transports(activity)))\nresult=(innercity_transport_set<={'metro'})","result=False\nfor activity in allactivities(plan):\n  if activity_position(activity)=='南门涮肉(天坛店)': result=True"],
}
"""
)

# get_json_str: 一个稳健的解析函数，用于从 LLM 的返回中截取标准的 JSON 字符串。
# def get_json_str(json_str):
#     l_ptr = json_str.find("{")
#     r_ptr = json_str.rfind("}")
#     if l_ptr != -1 and r_ptr != -1:
#         return json_str[l_ptr : r_ptr + 1]
#     else:
#         return {"error": "Translation failed, please try again."}

def get_json_str(json_str):
    if not isinstance(json_str, str):
        return "{}"
    
    # 移除 Markdown 代码块标记
    json_str = json_str.replace("```json", "").replace("```", "").strip()
    
    # 找到第一个 { 的位置
    start_idx = json_str.find("{")
    if start_idx == -1:
        return "{}"
    
    # 截取从第一个 { 开始的内容
    potential_json = json_str[start_idx:]
    
    try:
        # 使用 JSONDecoder 的 raw_decode 寻找第一个合法的 JSON 对象
        # 它会自动处理嵌套括号并停止在对象结束处
        import json
        decoder = json.JSONDecoder()
        obj, end_idx = decoder.raw_decode(potential_json)
        # 将对象转回字符串返回（为了保持你原有流程的兼容性）
        return json.dumps(obj)
    except Exception as e:
        # 如果 raw_decode 失败，使用最基础的正则尝试提取
        print(f"⚠️ [JSON Decoder 失败] 尝试正则提取: {e}")
        import re
        match = re.search(r"(\{.*\})", potential_json, re.DOTALL)
        if match:
            # 这里的正则仍然可能是贪婪的，但在 decoder 失败时是最后的补救
            return match.group(1).strip().replace("\n", " ")
        return "{}"

# get_step_1_prompt / get_step_2_prompt: 组装 LLM 输入。
def get_step_1_prompt(query):
    return nl_trans_prompt + "\n" + query["nature_language"]


def get_step_2_prompt(query, addtion_text):
    return sl_trans_prompt + "\n" + query["nature_language"] + "\n" + addtion_text


def nl2sl_2step(query, backbone_llm):

    addtion_text = backbone_llm(
        messages=[
            {
                "role": "user",
                "content": get_step_1_prompt(query),
            }
        ],
        one_line=False,
        json_mode=False,
    )
    addtion_text = addtion_text.split("<end>")[0].split("<constraint>")[1].strip()
    prompt = get_step_2_prompt(query, addtion_text)
    res = backbone_llm(
        messages=[{"role": "user", "content": prompt}],
        one_line=False,
        json_mode=True,
    )
    res = get_json_str(res)
    res = json.loads(res)
    for key in res:
        query[key] = res[key]
    return query


def nl2sl_fol_enhanced(query, backbone_llm):
    pass


def nl2sl_dsl_prompt_deepmind(query, backbone_llm):
    pass

# nl2sl_v2 / _nl2sl_v2: 核心工作流。
# 调用 LLM 获取自然语言约束。再次调用 LLM 将约束转为 Python 代码。解析返回的 JSON 并持久化到缓存。
# def nl2sl_v2(query, backbone_llm):
#     addtion_text = backbone_llm(
#         messages=[
#             {
#                 "role": "user",
#                 "content": get_step_1_prompt(query),
#             }
#         ],
#         one_line=False,
#         json_mode=False,
#     )
#     addtion_text = addtion_text.split("<end>")[0].split("<constraint>")[1].strip()
#     prompt = get_step_2_prompt(query, addtion_text)
#     res = backbone_llm(
#         messages=[{"role": "user", "content": prompt}],
#         one_line=False,
#         json_mode=True,
#     )
#     res = get_json_str(res)
#     res = json.loads(res)
#     for key in res:
#         query[key] = res[key]
#     query["nature_language_constraints"] = addtion_text.split("\n")
#     return query

def nl2sl_v2(query, backbone_llm):
    # 第一步：提取自然语言约束
    addtion_text = backbone_llm(
        messages=[{"role": "user", "content": get_step_1_prompt(query)}],
        one_line=False,
        json_mode=False,
    )
    
    try:
        # 增加切分安全性检查
        if "<constraint>" in addtion_text:
            addtion_text = addtion_text.split("<end>")[0].split("<constraint>")[1].strip()
        else:
            # 兜底逻辑：如果没按格式出，尝试按行匹配
            addtion_text = addtion_text.strip()
    except Exception:
        addtion_text = ""

    # 第二步：将约束转为 Python/JSON
    prompt = get_step_2_prompt(query, addtion_text)
    res_raw = backbone_llm(
        messages=[{"role": "user", "content": prompt}],
        one_line=False,
        json_mode=True,
    )
    
    # 鲁棒性解析
    res_str = get_json_str(res_raw)

    try:
        # 看报错的字符串
        # print(f"DEBUG: 待解析的JSON字符串为: |{res_str}|")
        res = json.loads(res_str)
    except json.JSONDecodeError:
        print(f"⚠️ [JSON解析失败] 原始输出: {res_raw[:100]}...")
        res = {} # 兜底返回空字典

    # try:
    #     # --- 修改部分：解析并以中文格式打印 ---
    #     res = json.loads(res_str)
        
    #     # 使用 ensure_ascii=False 来显示中文，indent=2 让格式更整齐
    #     readable_res = json.dumps(res, ensure_ascii=False, indent=2)
    #     print(f"DEBUG: 成功解析 JSON 对象:\n{readable_res}")
        
    # except json.JSONDecodeError:
    #     # --- 修改部分：解析失败时，尝试强制解码查看中文内容 ---
    #     try:
    #         # 这种方法可以将字符串中的 \u4e0a 这种编码强制转为中文
    #         decoded_error_str = res_str.encode('utf-8').decode('unicode_escape')
    #         print(f"DEBUG: [解析失败] 待解析的字符串内容为: |{decoded_error_str}|")
    #     except Exception:
    #         print(f"DEBUG: [解析失败] 待解析的原始字符串为: |{res_str}|")
            
    #     print(f"⚠️ [JSON解析失败] 原始输出前100字: {res_raw[:100]}...")
    #     res = {} # 兜底返回空字典

    # 更新 query
    for key in res:
        query[key] = res[key]
    
    # 保存约束列表供 EvidenceMapper 使用
    query["nature_language_constraints"] = [line.strip() for line in addtion_text.split("\n") if line.strip()]
    return query



def _nl2sl_v2(query, backbone_llm):

    llm_method = "translation_{}".format(backbone_llm.name)
    # cache_dir = os.path.join(project_root_path, "cache")
    # 向上跳三级到达 dpi_me，再跳一级到达 SynPT-main 根目录
    main_root = os.path.dirname(os.path.dirname(project_root_path))
    cache_dir = os.path.join(main_root, "cache")

    if not os.path.exists(os.path.join(cache_dir, llm_method)):
        os.makedirs(os.path.join(cache_dir, llm_method))
    file_path = os.path.join(cache_dir, llm_method, "{}.json".format(query["uid"]))
    if os.path.exists(file_path):  # and False:
        return load_json_file(file_path)
    else:
        # print(query["uid"])
        addtion_text = backbone_llm(
            messages=[
                {
                    "role": "user",
                    "content": get_step_1_prompt(query),
                }
            ],
            one_line=False,
            json_mode=False,
        )
        addtion_text = addtion_text.split("<end>")[0].split("<constraint>")[1].strip()
        prompt = get_step_2_prompt(query, addtion_text)
        res = backbone_llm(
            messages=[{"role": "user", "content": prompt}],
            one_line=False,
            json_mode=True,
        )
        l_ptr = res.find("{")
        r_ptr = res.rfind("}")
        if l_ptr != -1 and r_ptr != -1:
            res = res[l_ptr : r_ptr + 1]
        else:
            res = '{"error": "Translation failed, please try again."}'
            res["uid"] = query["uid"]
            return res
        res = json.loads(res)
        for key in res:
            query[key] = res[key]
        query["nature_language_constraints"] = addtion_text.split("\n")
        save_json_file(query, file_path)

    return query


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", "-d", type=str, default="easy")

#     args = parser.parse_args()
#     #data_dir = os.path.join(
#     #    project_root_path, "chinatravel/data/{}".format(args.data_dir)
#     #)
#     #deepseek = Deepseek()
#     # data_dir = os.path.join(project_root_path, "chinatravel/data/generated_easy")
#     file_list = os.listdir(data_dir)
#     # query = {
#     #     "nature_language": "当前位置苏州。我和女朋友打算去上海玩两天，开一间单床房。请给我一个旅行规划。"
#     # }
#     # cnt = 150
#     for file_name in tqdm(file_list):
#         # if cnt == 0:
#         #     break
#         if not file_name.startswith("t"):
#             continue
#         file_path = os.path.join(data_dir, file_name)
#         query = load_json_file(file_path)
#         res = _nl2sl_v2(query, deepseek)
#         # cnt -= 1
#         # print(res)
#         # break
