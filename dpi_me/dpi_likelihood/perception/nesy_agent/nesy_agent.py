import sys
import os
import time
import argparse
import pandas as pd
import json
import numpy as np

# 其核心逻辑是将自然语言查询（NL）转化为符号表示（SL），随后通过深度优先搜索（DFS）结合约束校验，在复杂的旅行规划环境（如城市、景点、酒店、交通）中寻找最优解。
sys.path.append("./../../../")
project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)


from agent.base import AbstractAgent, BaseAgent
from agent.nesy_agent.utils import (
    time_compare_if_earlier_equal,
    calc_cost_from_itinerary_wo_intercity,
    add_time_delta,
    TimeOutError,
)

# from chinatravel.eval.utils import load_json_file, validate_json, save_json_file
from chinatravel.data.load_datasets import load_json_file, save_json_file
from chinatravel.agent.utils import Logger
from chinatravel.symbol_verification.commonsense_constraint import (
    func_commonsense_constraints,
)
from chinatravel.symbol_verification.hard_constraint import (
    get_symbolic_concepts,
    evaluate_constraints,
    evaluate_constraints_py,
)
from chinatravel.symbol_verification.preference import evaluate_preference_py

from chinatravel.symbol_verification.concept_func import *
from chinatravel.agent.nesy_agent.nl2sl_hybrid import nl2sl_reflect
from copy import deepcopy


class NesyAgent(BaseAgent):
    # def __init__(
    #     self,
    #     env,
    #     backbone_llm,
    #     method="NeSy",
    #     cache_dir="cache/",
    #     max_time=None,
    #     debug=True,
    #     search_width=None,
    # ):

    # __init__: 初始化缓存目录、搜索宽度、时间阈值（5分钟超时）以及各类计数器（搜索节点数、回溯次数等）。
    def __init__(self, **kwargs):
        super().__init__(name="LLMNeSy", **kwargs)

        self.max_steps = kwargs.get('max_steps', 0)

        self.debug = kwargs.get("debug", False)

        self.memory = {}

        self.TIME_CUT = 60 * 5

        cache_dir = kwargs.get("cache_dir", "cache/")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

        # self.least_plan_schema, self.least_plan_comm = None, None
        # --- 修改 1: 补全回退计划变量 ---
        self.least_plan_schema, self.least_plan_comm, self.least_plan_logic = None, None, None
        self.least_plan_logical_pass = -1
        self.least_plan_logic_pvalue = 0

        self.time_before_search = time.time()
        self.llm_inference_time_count = 0 
        self.llm_rec_count = 0
        self.llm_rec_format_error = 0
        self.search_nodes = 0
        self.backtrack_count = 0
        self.constraints_validation_count = 0 
        self.commonsense_pass_count = 0
        self.logical_pass_count = 0
        self.all_constraints_pass = 0

        self.method = kwargs["method"]

        print("cache dir:", self.cache_dir)
        if not os.path.exists(
            os.path.join(self.cache_dir, self.method + "_" + self.backbone_llm.name)
        ):
            os.makedirs(
                os.path.join(self.cache_dir, self.method + "_" + self.backbone_llm.name)
            )
        self.search_width = kwargs.get("search_width", None)

        self.preference_search = False
        self.prompt_upd = True

        # --- 修改 2: 预先初始化所有统计属性，彻底解决 AttributeError ---
        self.time_before_search = time.time()
        self.llm_inference_time_count = 0  # 对应 run 函数中的耗时统计
        self.llm_rec_count = 0
        self.llm_rec_format_error = 0     # 对应 run 函数中的格式错误统计
        self.search_nodes = 0
        self.backtrack_count = 0
        self.constraints_validation_count = 0
        self.commonsense_pass_count = 0
        self.logical_pass_count = 0
        self.all_constraints_pass = 0

    def reset(self):
        pass
 
    # translate_nl2sl: 调用 nl2sl_reflect 模块，将模糊的语言描述转化为可执行的符号逻辑。
    def translate_nl2sl(self, query, load_cache=False):

        llm_method = "translation_{}_reflect".format(self.backbone_llm.name)
        if not os.path.exists(os.path.join(self.cache_dir, llm_method)):
            os.makedirs(os.path.join(self.cache_dir, llm_method))

        file_path = os.path.join(
            self.cache_dir, llm_method, "{}.json".format(query["uid"])
        )

        print(file_path)

        if load_cache and os.path.exists(file_path):
            query = load_json_file(file_path)

        else:
            query = nl2sl_reflect(query, self.backbone_llm)
            # 打印翻译后结果
            print("\n" + "="*50)
            print("【DEBUG】LLM 翻译生成的 Symbolic Query 如下：")
            print(json.dumps(query, indent=4, ensure_ascii=False))
            print("="*50 + "\n")
            if "error" in query:
                query["hard_logic_py"] = {}
            # 确保翻译后的 query 依然使用原始指定的城市
            # 使用 get 方法或者 self.query 确保安全访问
            if hasattr(self, 'query') and isinstance(self.query, dict):
                query["start_city"] = self.query.get("start_city", query["start_city"])
                query["target_city"] = self.query.get("target_city", query["target_city"])
            # query["start_city"] = self.query["start_city"] if hasattr(self, 'query') else query["start_city"]
            # query["target_city"] = self.query["target_city"] if hasattr(self, 'query') else query["target_city"]
            save_json_file(query, file_path)

        return query

    # run: 整个 Agent 的主入口，协调翻译逻辑和搜索逻辑，并最终记录 LLM Token 消耗和搜索效率指标。
    def run(self, query, load_cache=False, oralce_translation=False, preference_search=False):
        original_start_city = query.get("start_city")
        original_target_city = query.get("target_city")

        self.preference_search = preference_search
        method_name = self.method + "_" + self.backbone_llm.name
        if oralce_translation:
            method_name = method_name + "_oracletranslation"
        if preference_search:
            method_name = method_name + "_preferencesearch"
        self.log_dir = os.path.join(self.cache_dir, method_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        sys.stdout = Logger(
            "{}/{}.log".format(
                self.log_dir, query["uid"]
            ),
            sys.stdout,
            self.debug,
        )
        sys.stderr = Logger(
            "{}/{}.error".format(
                self.log_dir, query["uid"]
            ),
            sys.stderr,
            self.debug,
        )


        self.backbone_llm.input_token_count = 0
        self.backbone_llm.output_token_count = 0
        self.backbone_llm.input_token_maxx = 0


        # natural language -> symoblic language -> plan

        if not oralce_translation:
            query = self.translate_nl2sl(query, load_cache=load_cache)
        
        # --- 关键加固代码：在这里强行改回来 ---
        if original_start_city:
            query["start_city"] = original_start_city
        if original_target_city:
            query["target_city"] = original_target_city
        # -----------------------------------

        succ, plan = self.symbolic_search(query)

        if succ:
            plan_out = plan
        else:
            # if self.least_plan_logic is not None:
            # if getattr(self, 'least_plan_logic', None) is not None:
            #     plan_out = self.least_plan_logic

            #     if preference_search:
            #         plan_out["preference_value"] = self.least_plan_logic_pvalue

            #     print("The least plan with logic constraints: ", plan_out)
            #     succ = True

            # elif self.least_plan_comm is not None:
            #     plan_out = self.least_plan_comm
            # elif self.least_plan_schema is not None:
            #     plan_out = self.least_plan_schema
            # else:
            #     plan_out = {}

            # # plan_out["search_time_sec"] = time.time() - self.time_before_search
            # # 使用 getattr 安全获取起始时间，如果不存在则设为当前时间（即耗时为0）
            # plan_out["search_time_sec"] = time.time() - getattr(self, 'time_before_search', time.time())
            # plan_out["llm_inference_time_sec"] = self.llm_inference_time_count
            # if plan_out["search_time_sec"] > self.TIME_CUT:
            #     plan_out["time_out_flag"] = True
            
            # 修改 1: 使用 getattr 安全获取 least_plan_logic
            if getattr(self, 'least_plan_logic', None) is not None:
                plan_out = self.least_plan_logic
                if preference_search:
                    plan_out["preference_value"] = getattr(self, 'least_plan_logic_pvalue', 0)
                print("The least plan with logic constraints: ", plan_out)
                succ = True
            elif getattr(self, 'least_plan_comm', None) is not None:
                plan_out = self.least_plan_comm
            elif getattr(self, 'least_plan_schema', None) is not None:
                plan_out = self.least_plan_schema
            else:
                plan_out = {}

            # 修改 2: 使用 getattr 安全计算搜索时间
            plan_out["search_time_sec"] = time.time() - getattr(self, 'time_before_search', time.time())
            
            # 修改 3: 安全获取 LLM 推理时间
            plan_out["llm_inference_time_sec"] = getattr(self, 'llm_inference_time_count', 0)
            
            if plan_out.get("search_time_sec", 0) > getattr(self, 'TIME_CUT', 3600):
                plan_out["time_out_flag"] = True


        # plan_out["input_token_count"] = self.backbone_llm.input_token_count
        # plan_out["output_token_count"] = self.backbone_llm.output_token_count
        # plan_out["input_token_maxx"] = self.backbone_llm.input_token_maxx

        # plan_out["llm_rec_count"] = self.llm_rec_count
        # plan_out["llm_rec_format_error_count"] = self.llm_rec_format_error

        # plan_out["search_nodes"] = self.search_nodes
        # plan_out["backtrack_count"] = self.backtrack_count
        # plan_out["constraints_validation_count"] = self.constraints_validation_count
        # plan_out["commonsense_pass_count"] = self.commonsense_pass_count
        # plan_out["logical_pass_count"] = self.logical_pass_count
        # plan_out["all_constraints_pass"] = self.all_constraints_pass

        # 修改 4: 所有的统计字段全部改为安全访问 (getattr)
        # 这样即便跳过搜索，API 也不会报 500 错误
        plan_out["input_token_count"] = self.backbone_llm.input_token_count
        plan_out["output_token_count"] = self.backbone_llm.output_token_count
        plan_out["input_token_maxx"] = self.backbone_llm.input_token_maxx

        plan_out["llm_rec_count"] = getattr(self, 'llm_rec_count', 0)
        plan_out["llm_rec_format_error_count"] = getattr(self, 'llm_rec_format_error', 0)

        plan_out["search_nodes"] = getattr(self, 'search_nodes', 0)
        plan_out["backtrack_count"] = getattr(self, 'backtrack_count', 0)
        plan_out["constraints_validation_count"] = getattr(self, 'constraints_validation_count', 0)
        plan_out["commonsense_pass_count"] = getattr(self, 'commonsense_pass_count', 0)
        plan_out["logical_pass_count"] = getattr(self, 'logical_pass_count', 0)
        plan_out["all_constraints_pass"] = getattr(self, 'all_constraints_pass', 0)

        return succ, plan_out

    # constraints_validation: 调用外部工具校验当前的行程是否违反逻辑规则，并记录“最优部分解”（Least Plan）。
    def constraints_validation(self, query, plan, poi_plan):

        self.constraints_validation_count += 1

        res_plan = {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": plan,
        }
        print("validate the plan [for query {}]: ".format(query["uid"]))
        print(res_plan)

        self.least_plan_schema = deepcopy(res_plan)

        bool_result = func_commonsense_constraints(query, res_plan, verbose=True)

        # if not bool_result:
        #     exit(0)

        if bool_result:
            self.commonsense_pass_count += 1

        try:
            extracted_vars = get_symbolic_concepts(query, res_plan, need_ood=False)

        except:
            extracted_vars = None

        print(extracted_vars)

        logical_result = evaluate_constraints_py(query["hard_logic_py"], res_plan, verbose=True)

        print(logical_result)

        logical_pass = True
        for idx, item in enumerate(logical_result):
            logical_pass = logical_pass and item

            if item:
                print(query["hard_logic_py"][idx], "passed!")
            else:

                print(query["hard_logic_py"][idx], "failed...")
        if bool_result and np.sum(logical_result) > self.least_plan_logical_pass:
            self.least_plan_comm = deepcopy(res_plan)
            self.least_plan_logical_pass = np.sum(logical_result)
        # if logical_result:
        #     print("Logical passed!")

        if logical_pass:
            self.logical_pass_count += 1

        bool_result = bool_result and logical_pass

        if bool_result:
            print("\n Pass! \n")
            self.all_constraints_pass += 1

            if self.least_plan_logic is None:
                self.least_plan_logic = res_plan

            if self.preference_search:
                # self.least_plan_logic = res_plan
                try:
                    if self.query["preference_opt"] == "maximize":
                        
                        res = evaluate_preference_py([(self.query["preference_opt"], self.query["preference_concept"], self.query["preference_code"])], res_plan)[0]
                        print(self.query["preference_concept"], res)

                        # print(res, self.least_plan_logic_pvalue)
                        if res != -1 and res > self.least_plan_logic_pvalue:
                            print("preference value [{}]: {} -> {} \n update plan".format(self.query["preference_concept"], self.least_plan_logic_pvalue, res))
                            self.least_plan_logic_pvalue = res
                            self.least_plan_logic = deepcopy(res_plan)


                    elif self.query["preference_opt"] == "minimize":
                        res = evaluate_preference_py([(self.query["preference_opt"], self.query["preference_concept"], self.query["preference_code"])] , res_plan)[0]
                        print(self.query["preference_concept"], res)

                        # print(res, self.least_plan_logic_pvalue)
                        if res != -1 and res < self.least_plan_logic_pvalue:
                            print("preference value [{}]: {} -> {} \n update plan".format(self.query["preference_concept"], self.least_plan_logic_pvalue, res))
                            self.least_plan_logic_pvalue = res
                            self.least_plan_logic = deepcopy(res_plan)

                    else:
                        raise ValueError("Invalid preference_opt")
                    print(self.least_plan_logic)
                except Exception as e:
                    print(e)
                    print(self.query["preference_code"])
        else:
            print("\n Failed \n")

        # plan = res_plan

        # print(result)
        # exit(0)

        if self.preference_search:
            return False, plan

        if bool_result:
            res_plan["search_time_sec"] = time.time() - self.time_before_search
            res_plan["llm_inference_time_sec"] = self.llm_inference_time_count
            return True, res_plan
        else:
            return False, plan

    def add_intercity_transport(
        self, activities, intercity_info, innercity_transports=[], tickets=1
    ):
        activity_i = {
            "start_time": intercity_info["BeginTime"],
            "end_time": intercity_info["EndTime"],
            "start": intercity_info["From"],
            "end": intercity_info["To"],
            "price": intercity_info["Cost"],
            "cost": intercity_info["Cost"] * tickets,
            "tickets": tickets,
            "transports": innercity_transports,
        }
        if not pd.isna(intercity_info["TrainID"]):
            activity_i["TrainID"] = intercity_info["TrainID"]
            activity_i["type"] = "train"
        elif not pd.isna(intercity_info["FlightID"]):
            activity_i["FlightID"] = intercity_info["FlightID"]
            activity_i["type"] = "airplane"

        activities.append(activity_i)
        return activities
    

    # def add_intercity_transport(
    #     self, activities, intercity_info, innercity_transports=[], tickets=1
    # ):
    #     print(f"\n[DEBUG LOG] 正在尝试添加交通信息...")
    #     print(f"[DEBUG LOG] intercity_info 类型: {type(intercity_info)}")
    #     print(f"[DEBUG LOG] intercity_info 内容: {intercity_info}")
    #     # 1. 安全提取数据，防止 KeyError
    #     if isinstance(intercity_info, pd.Series) or isinstance(intercity_info, dict):
    #         begin_time = intercity_info.get("BeginTime", "08:00")
    #         end_time = intercity_info.get("EndTime", "10:00")
    #         from_loc = intercity_info.get("From", "Unknown")
    #         to_loc = intercity_info.get("To", "Unknown")
    #         cost = intercity_info.get("Cost", 0)
    #         train_id = intercity_info.get("TrainID", None)
    #         flight_id = intercity_info.get("FlightID", None)
    #     else:
    #         begin_time, end_time, from_loc, to_loc, cost = "08:00", "10:00", "Unknown", "Unknown", 0
    #         train_id, flight_id = None, None

    #     # 2. 构造活动字典
    #     activity_i = {
    #         "start_time": begin_time,
    #         "end_time": end_time,
    #         "start": from_loc,
    #         "end": to_loc,
    #         "price": float(cost),
    #         "cost": float(cost) * tickets,
    #         "tickets": tickets,
    #         "transports": innercity_transports,
    #     }

    #     # 3. 补充交通类型（保留原项目逻辑）
    #     if train_id and not pd.isna(train_id):
    #         activity_i["TrainID"] = train_id
    #         activity_i["type"] = "train"
    #     elif flight_id and not pd.isna(flight_id):
    #         activity_i["FlightID"] = flight_id
    #         activity_i["type"] = "airplane"
    #     else:
    #         activity_i["type"] = "intercity_transport"

    #     # 4. 关键：必须添加到列表并返回！
    #     activities.append(activity_i)
    #     return activities

    # add_xxx (add_restaurant, add_attraction, etc.): 具体的动作函数，负责计算每个活动的起始/结束时间、门票费用以及所需的市内交通（Taxi/Walk）。
    def add_poi(
        self,
        activities,
        position,
        poi_type,
        price,
        cost,
        start_time,
        end_time,
        innercity_transports,
    ):
        activity_i = {
            "position": position,
            "type": poi_type,
            "price": price,
            "cost": cost,
            "start_time": start_time,
            "end_time": end_time,
            "transports": innercity_transports,
        }

        activities.append(activity_i)
        return activities

    def add_accommodation(
        self,
        current_plan,
        hotel_sel,
        current_day,
        arrived_time,
        required_rooms,
        transports_sel,
    ):

        current_plan[current_day]["activities"] = self.add_poi(
            activities=current_plan[current_day]["activities"],
            position=hotel_sel["name"],
            poi_type="accommodation",
            price=int(hotel_sel["price"]),
            cost=int(hotel_sel["price"]) * required_rooms,
            start_time=arrived_time,
            end_time="24:00",
            innercity_transports=transports_sel,
        )
        current_plan[current_day]["activities"][-1]["room_type"] = hotel_sel["numbed"]
        current_plan[current_day]["activities"][-1]["rooms"] = required_rooms

        return current_plan

    def add_restaurant(
        self, current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel
    ):

        # 开放时间
        opentime, endtime = (
            poi_sel["opentime"],
            poi_sel["endtime"],
        )

        # it is closed ...
        if time_compare_if_earlier_equal(endtime, arrived_time):
            raise Exception("Add POI error")
        if time_compare_if_earlier_equal(arrived_time, opentime):
            act_start_time = opentime
        else:
            act_start_time = arrived_time

        if poi_type == "lunch" and time_compare_if_earlier_equal(
            act_start_time, "11:00"
        ):
            act_start_time = "11:00"
        if poi_type == "lunch" and time_compare_if_earlier_equal(endtime, "11:00"):
            raise Exception("Add POI error")

        if poi_type == "dinner" and time_compare_if_earlier_equal(
            act_start_time, "17:00"
        ):
            act_start_time = "17:00"
        if poi_type == "dinner" and time_compare_if_earlier_equal(endtime, "17:00"):
            raise Exception("Add POI error")

        if poi_type == "lunch" and time_compare_if_earlier_equal(
            "13:00", act_start_time
        ):
            raise Exception("Add POI error")
        if poi_type == "dinner" and time_compare_if_earlier_equal(
            "20:00", act_start_time
        ):
            raise Exception("Add POI error")

        poi_time = 60
        act_end_time = add_time_delta(act_start_time, poi_time)
        if time_compare_if_earlier_equal(endtime, act_end_time):
            act_end_time = endtime

        tmp_plan = deepcopy(current_plan)
        tmp_plan[current_day]["activities"] = self.add_poi(
            activities=tmp_plan[current_day]["activities"],
            position=poi_sel["name"],
            poi_type=poi_type,
            price=int(poi_sel["price"]),
            cost=int(poi_sel["price"]) * self.query["people_number"],
            start_time=act_start_time,
            end_time=act_end_time,
            innercity_transports=transports_sel,
        )
        return tmp_plan

    def add_attraction(
        self, current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel
    ):

        # 开放时间
        opentime, endtime = (
            poi_sel["opentime"],
            poi_sel["endtime"],
        )

        # it is closed ...

        opentime, endtime = poi_sel["opentime"], poi_sel["endtime"]
        # it is closed ...
        if time_compare_if_earlier_equal(endtime, arrived_time):
            raise Exception("Add POI error")

        if time_compare_if_earlier_equal(arrived_time, opentime):
            act_start_time = opentime
        else:
            act_start_time = arrived_time

        poi_time = 90
        act_end_time = add_time_delta(act_start_time, poi_time)
        if time_compare_if_earlier_equal(endtime, act_end_time):
            act_end_time = endtime

        tmp_plan = deepcopy(current_plan)
        tmp_plan[current_day]["activities"] = self.add_poi(
            activities=tmp_plan[current_day]["activities"],
            position=poi_sel["name"],
            poi_type=poi_type,
            price=int(poi_sel["price"]),
            cost=int(poi_sel["price"]) * self.query["people_number"],
            start_time=act_start_time,
            end_time=act_end_time,
            innercity_transports=transports_sel,
        )
        tmp_plan[current_day]["activities"][-1]["tickets"] = self.query["people_number"]

        return tmp_plan

    def check_if_too_late(
        self, query, current_day, current_time, current_position, poi_plan
    ):

        if current_time != "" and time_compare_if_earlier_equal("23:00", current_time):
            print("too late, after 23:00")
            return True

        if current_time != "" and current_day == query["days"] - 1:
            # We should go back in time ...
            transports_ranking = self.innercity_transports_ranking_from_query

            for transport_type_sel in transports_ranking:
                
                self.search_nodes += 1

                flag = True
                if "back_transport" in poi_plan:
                    transports_sel = self.collect_innercity_transport(
                        query["target_city"],
                        current_position,
                        poi_plan["back_transport"]["From"],
                        current_time,
                        transport_type_sel,
                    )
                    if not isinstance(transports_sel, list):
                        self.backtrack_count += 1
                        print("inner-city transport error, backtrack...")
                        continue

                    if len(transports_sel) > 0:
                        arrived_time = transports_sel[-1]["end_time"]
                    else:
                        arrived_time = current_time

                    if not time_compare_if_earlier_equal(
                        poi_plan["back_transport"]["BeginTime"], arrived_time
                    ):
                        flag = False
                if flag:
                    print(
                        "Can not go back source-city in time, current POI {}, station arrived time: {}".format(
                            current_position, arrived_time
                        )
                    )
                    return True

        elif current_time != "":
            if "accommodation" in poi_plan:
                hotel_sel = poi_plan["accommodation"]
                transports_ranking = self.innercity_transports_ranking_from_query

                for transport_type_sel in transports_ranking:
                    self.search_nodes += 1
                    flag = True
                    if "back_transport" in poi_plan:
                        transports_sel = self.collect_innercity_transport(
                            query["target_city"],
                            current_position,
                            hotel_sel["name"],
                            current_time,
                            transport_type_sel,
                        )
                        if not isinstance(transports_sel, list):
                            self.backtrack_count += 1
                            print("inner-city transport error, backtrack...")
                            continue

                        flag = True
                        # print(transports_sel)
                        # print(transports_sel[-1])

                        if len(transports_sel) > 0:
                            arrived_time = transports_sel[-1]["end_time"]
                        else:
                            arrived_time = current_time
                        if not time_compare_if_earlier_equal("24:00", arrived_time):
                            flag = False
                    if flag:
                        print(
                            "Can not go back to hotel, current POI {}, hotel arrived time: {}".format(
                                current_position, arrived_time
                            )
                        )
                        return True

        return False

    # reranking_xxx_with_constraints: 这是一系列函数（针对交通、酒店、餐厅、景点），在搜索之前先根据符号约束对候选项目进行重排序，优先尝试符合约束的项目以提高搜索效率。
    def reranking_intercity_transport_go_with_constraints(
        self, ranking_go, go_info, query
    ):

        ### check constraints
        pass_num_list = np.zeros(len(go_info))

        for go_i in ranking_go:

            go_sel = go_info.iloc[go_i]
            tmp_plan = [{"day": 1, "activities": []}]
            tmp_plan[0]["activities"] = self.add_intercity_transport(
                tmp_plan[0]["activities"],
                go_sel,
                innercity_transports=[],
                tickets=self.query["people_number"],
            )

            res_plan = {
                "people_number": query["people_number"],
                "start_city": query["start_city"],
                "target_city": query["target_city"],
                "itinerary": tmp_plan,
            }
            # print("validate the plan [for query {}]: ".format(query["uid"]))
            # print(res_plan)

            logical_result = evaluate_constraints_py(query["hard_logic_py"], res_plan)

            # print(logical_result)

            pass_num_list[go_i] = np.sum(logical_result)

        # pass_maxx = int(np.max(pass_num_list))

        # 修改为防御性写法：
        if len(pass_num_list) > 0:
            pass_maxx = int(np.max(pass_num_list))
        else:
            pass_maxx = 1 # 设置为1防止下方除法报错，且反正列表为空不影响结果

        print(pass_num_list)
        print(pass_maxx)

        reranking_list = []
        if pass_maxx > 0:
            for p_i in range(pass_maxx, -1, -1):
                for idx in ranking_go:
                    if pass_num_list[idx] == p_i:
                        reranking_list.append(idx)
        else:
            reranking_list = ranking_go

        # print(reranking_list)
        # exit(0)
        return reranking_list

    def reranking_intercity_transport_back_with_constraints(
        self, ranking_back, back_info, query, go_sel
    ):

        ### check constraints
        pass_num_list = np.zeros(len(back_info))

        for back_i in ranking_back:

            back_sel = back_info.iloc[back_i]
            tmp_plan = [{"day": 1, "activities": []}]
            tmp_plan[0]["activities"] = self.add_intercity_transport(
                tmp_plan[0]["activities"],
                go_sel,
                innercity_transports=[],
                tickets=self.query["people_number"],
            )
            if query["days"] > 1:
                for dayy in range(1, query["days"]):
                    tmp_plan.append({"day": dayy + 1, "activities": []})
            tmp_plan[-1]["activities"] = self.add_intercity_transport(
                tmp_plan[-1]["activities"],
                back_sel,
                innercity_transports=[],
                tickets=self.query["people_number"],
            )

            res_plan = {
                "people_number": query["people_number"],
                "start_city": query["start_city"],
                "target_city": query["target_city"],
                "itinerary": tmp_plan,
            }
            # print("validate the plan [for query {}]: ".format(query["uid"]))
            # print(res_plan)

            logical_result = evaluate_constraints_py(query["hard_logic_py"], res_plan)

            # print(logical_result)

            pass_num_list[back_i] = np.sum(logical_result)

        pass_maxx = int(np.max(pass_num_list))

        # print(pass_num_list)
        # print(pass_maxx)

        reranking_list = []
        if pass_maxx > 0:
            for p_i in range(pass_maxx, -1, -1):
                for idx in ranking_back:
                    if pass_num_list[idx] == p_i:
                        reranking_list.append(idx)
        else:
            reranking_list = ranking_back

        # print(reranking_list)
        # exit(0)
        return reranking_list

    def reranking_hotel_with_constraints(
        self, ranking_hotel, hotel_info, query, query_room_number
    ):

        pass_num_list = np.zeros(len(hotel_info))
        ### check constraints

        for idx in range(len(hotel_info)):
            hotel_sel = hotel_info.iloc[idx]

            if query_room_number == None:
                room_type = hotel_sel["numbed"]
                required_rooms = int((query["people_number"] - 1) / room_type) + 1
            else:
                required_rooms = query_room_number

            plan = []
            for dayy in range(query["days"] - 1):
                plan.append({"day": dayy + 1, "activities": []})
                plan = self.add_accommodation(
                    current_plan=plan,
                    hotel_sel=hotel_sel,
                    current_day=dayy,
                    arrived_time="20:00",
                    required_rooms=required_rooms,
                    transports_sel=[],
                )

            res_plan = {
                "people_number": query["people_number"],
                "start_city": query["start_city"],
                "target_city": query["target_city"],
                "itinerary": plan,
            }
            # print("validate the plan [for query {}]: ".format(query["uid"]))
            # print(res_plan)

            logical_result = evaluate_constraints_py(query["hard_logic_py"], res_plan)

            # print(logical_result)

            pass_num_list[idx] = np.sum(logical_result)

        pass_maxx = int(np.max(pass_num_list))

        # print(pass_num_list)
        # print(pass_maxx)

        reranking_list = []
        if pass_maxx > 0:
            for p_i in range(pass_maxx, -1, -1):
                for idx in ranking_hotel:
                    if pass_num_list[idx] == p_i:
                        reranking_list.append(idx)
        else:
            reranking_list = ranking_hotel

        # for r_i in reranking_list[:10]:
        #     print(hotel_info.iloc[r_i])
        #     print(pass_num_list[r_i])

        # print(reranking_list)
        # exit(0)
        return reranking_list

    def reranking_restaurants_with_constraints(
        self,
        plan,
        poi_type,
        current_day,
        current_time,
        current_position,
        rest_info,
        query,
        ranking_restaurants,
    ):

        pass_num_list = []
        ### check constraints

        for idx in range(len(rest_info)):
            poi_sel = rest_info.iloc[idx]
            self.search_nodes += 1
            if current_position == poi_sel["name"]:
                transports_sel = []
                arrived_time = current_time
            else:

                transports_sel = self.collect_innercity_transport(
                    query["target_city"],
                    current_position,
                    poi_sel["name"],
                    current_time,
                    "taxi",
                )
                if not isinstance(transports_sel, list):
                    self.backtrack_count += 1
                    print("inner-city transport error, backtrack...")
                    continue

                if len(transports_sel) == 0:
                    arrived_time = current_time
                else:
                    arrived_time = transports_sel[-1]["end_time"]


            try:
                tmp_plan = self.add_restaurant(
                    plan, poi_type, poi_sel, current_day, arrived_time, transports_sel
                )
                res_plan = {
                    "people_number": query["people_number"],
                    "start_city": query["start_city"],
                    "target_city": query["target_city"],
                    "itinerary": tmp_plan,
                }
                logical_result = evaluate_constraints_py(
                    query["hard_logic_py"], res_plan
                )
                pass_num_list.append(np.sum(logical_result))
            except:
                pass_num_list.append(0)

            # print(logical_result)
            # pass_num_list.append(np.sum(logical_result))

        pass_maxx = np.max(pass_num_list)

        # print(pass_num_list)
        # print(pass_maxx)

        reranking_list = []
        if pass_maxx > 0:
            for p_i in range(pass_maxx, -1, -1):
                for idx in ranking_restaurants:
                    if pass_num_list[idx] == p_i:
                        reranking_list.append(idx)
        else:
            reranking_list = ranking_restaurants

        # print(reranking_list)
        # exit(0)

        # for r_i in ranking_restaurants[:10]:
        #     print(rest_info.iloc[r_i])

        # print("re-ranking ---")
        # for r_i in reranking_list[:10]:
        #     print(rest_info.iloc[r_i])

        return reranking_list

    def reranking_attractions_with_constraints(
        self,
        plan,
        poi_type,
        current_day,
        current_time,
        current_position,
        attr_info,
        query,
        ranking_attractions,
    ):

        pass_num_list = []
        ### check constraints

        for idx in range(len(attr_info)):
            poi_sel = attr_info.iloc[idx]
            self.search_nodes += 1
            if poi_sel["name"] == current_position:
                transports_sel = []
                arrived_time = current_time
            else:
                transports_sel = self.collect_innercity_transport(
                    query["target_city"],
                    current_position,
                    poi_sel["name"],
                    current_time,
                    "taxi",
                )
                if not isinstance(transports_sel, list):
                    self.backtrack_count += 1
                    print("inner-city transport error, backtrack...")
                    continue

                if len(transports_sel) == 0:
                    arrived_time = current_time
                else:
                    arrived_time = transports_sel[-1]["end_time"]


            try:
                tmp_plan = self.add_attraction(
                    plan, poi_type, poi_sel, current_day, arrived_time, transports_sel
                )
                res_plan = {
                    "people_number": query["people_number"],
                    "start_city": query["start_city"],
                    "target_city": query["target_city"],
                    "itinerary": tmp_plan,
                }
                logical_result = evaluate_constraints_py(
                    query["hard_logic_py"], res_plan
                )
                pass_num_list.append(np.sum(logical_result))
            except:
                pass_num_list.append(0)
        pass_maxx = np.max(pass_num_list)

        # print(pass_num_list)
        # print(pass_maxx)

        reranking_list = []
        if pass_maxx > 0:
            for p_i in range(pass_maxx, -1, -1):
                for idx in ranking_attractions:
                    if pass_num_list[idx] == p_i:
                        reranking_list.append(idx)
        else:
            reranking_list = ranking_attractions

        # print(reranking_list)
        # exit(0)
        return reranking_list

    # dfs_poi (深度优先搜索)：代码的核心算法。它递归地在每一天内填充早餐、景点、午餐、晚餐和酒店，确保每一步都符合时间限制和预算限制。
    def dfs_poi(
        self, query, poi_plan, plan, current_time, current_position, current_day=0
    ):

        self.search_nodes += 1
        if (
            time.time() - self.time_before_search
            > self.TIME_CUT + self.llm_inference_time_count
        ):

            raise TimeOutError

        if self.check_if_too_late(
            query, current_day, current_time, current_position, poi_plan
        ):
            self.backtrack_count += 1
            print("The current time is too late to go hotel or back-transport, backtrack...")
            return False, plan

        if self.required_budget != None:
            total_cost = 0
            for day_activities in plan:
                for activity in day_activities["activities"]:
                    if activity["type"] in [
                        "breakfast",
                        "lunch",
                        "dinner",
                        "attraction",
                    ]:
                        total_cost += activity["cost"]

            if total_cost + self.intercity_with_hotel_cost > self.required_budget:
                self.backtrack_count += 1
                print("budget exceeded, backtrack...")
                return False, plan

        # intercity_transport - go
        if current_day == 0 and current_time == "":
            plan = [{"day": current_day + 1, "activities": []}]
            plan[current_day]["activities"] = self.add_intercity_transport(
                plan[current_day]["activities"],
                poi_plan["go_transport"],
                innercity_transports=[],
                tickets=self.query["people_number"],
            )
            new_time = poi_plan["go_transport"]["EndTime"]
            new_position = poi_plan["go_transport"]["To"]
            success, plan = self.dfs_poi(
                query, poi_plan, plan, new_time, new_position, current_day
            )
            if success:
                return True, plan
            else:
                self.backtrack_count += 1
                print("No solution for the given Go Transport, backtrack...")
                return False, plan

        # breakfast
        if current_time == "00:00":

            if len(plan) < current_day + 1:
                plan.append({"day": current_day + 1, "activities": []})

            self.search_nodes += 1
            plan = self.select_and_add_breakfast(
                plan, poi_plan, current_day, current_time, current_position
            )

            new_time = plan[current_day]["activities"][-1]["end_time"]
            new_position = current_position
            success, plan = self.dfs_poi(
                query, poi_plan, plan, new_time, new_position, current_day
            )
            if success:
                return True, plan
            
            plan[current_day]["activities"].pop()

            candidates_type = []
            if current_day == query["days"] - 1 and current_time != "":
                candidates_type.append("back-intercity-transport")
            else:

                self.backtrack_count += 1
                print("No solution for the given Breakfast, backtrack...")

                return False, plan

        else:
            haved_lunch_today, haved_dinner_today = False, False

            for act_i in plan[current_day]["activities"]:
                if act_i["type"] == "lunch":
                    haved_lunch_today = True
                if act_i["type"] == "dinner":
                    haved_dinner_today = True

            candidates_type = ["attraction"]
            if not haved_lunch_today:
                candidates_type.append("lunch")
            if not haved_dinner_today:
                candidates_type.append("dinner")
            if ("accommodation" in poi_plan) and (current_day < query["days"] - 1):
                candidates_type.append("hotel")
            if current_day == query["days"] - 1 and current_time != "":
                candidates_type.append("back-intercity-transport")

        print("candidates_type: ", candidates_type)

        while len(candidates_type) > 0:

            poi_type, candidates_type = self.select_next_poi_type(
                candidates_type,
                plan,
                poi_plan,
                current_day,
                current_time,
                current_position,
            )

            print(
                "POI planning, day {} {}, {}, next-poi type: {}".format(
                    current_day, current_time, current_position, poi_type
                )
            )

            if poi_type == "back-intercity-transport":

                if len(plan) < current_day + 1:
                    plan.append({"day": current_day + 1, "activities": []})

                # transports_ranking = self.ranking_innercity_transport(current_position, poi_plan["back_transport"]["From"], current_day, current_time)
                transports_ranking = self.innercity_transports_ranking_from_query
                for trans_type_sel in transports_ranking:
                    self.search_nodes += 1
                    transports_sel = self.collect_innercity_transport(
                        query["target_city"],
                        current_position,
                        poi_plan["back_transport"]["From"],
                        current_time,
                        trans_type_sel,
                    )
                    if not isinstance(transports_sel, list):
                        self.backtrack_count += 1
                        print("inner-city transport error, backtrack...")
                        continue

                    plan[current_day]["activities"] = self.add_intercity_transport(
                        plan[current_day]["activities"],
                        poi_plan["back_transport"],
                        innercity_transports=transports_sel,
                        tickets=self.query["people_number"],
                    )

                    res_bool, res_plan = self.constraints_validation(
                        query, plan, poi_plan
                    )

                    if res_bool:
                        return True, res_plan
                    else:
                        plan[current_day]["activities"].pop()
                        self.backtrack_count += 1

                        print(
                            "Back-transport, but constraints_validation failed, backtrack..."
                        )
                        return False, plan
            elif poi_type == "hotel":

                hotel_sel = poi_plan["accommodation"]

                # transports_ranking = self.ranking_innercity_transport(current_position, hotel_sel["name"], current_day, current_time)
                transports_ranking = self.innercity_transports_ranking_from_query
                
                for trans_type_sel in transports_ranking:
                    self.search_nodes += 1
                    if hotel_sel["name"] == current_position:
                        transports_sel = []
                        arrived_time = current_time
                    else:
                        transports_sel = self.collect_innercity_transport(
                            query["target_city"],
                            current_position,
                            hotel_sel["name"],
                            current_time,
                            trans_type_sel,
                        )
                        if not isinstance(transports_sel, list):
                            self.backtrack_count += 1
                            print("inner-city transport error, backtrack...")
                            continue

                        if len(transports_sel) == 0:
                            arrived_time = current_time
                        else:
                            arrived_time = transports_sel[-1]["end_time"]


                    plan = self.add_accommodation(
                        current_plan=plan,
                        hotel_sel=hotel_sel,
                        current_day=current_day,
                        arrived_time=arrived_time,
                        required_rooms=self.required_rooms,
                        transports_sel=transports_sel,
                    )

                    new_time = "00:00"
                    new_position = hotel_sel["name"]

                    success, plan = self.dfs_poi(
                        query, poi_plan, plan, new_time, new_position, current_day + 1
                    )

                    if success:
                        return True, plan

                    self.backtrack_count += 1
                    print("Fail with the given accommodation activity, backtrack...")

                    plan[current_day]["activities"].pop()
            elif poi_type in ["lunch", "dinner", "attraction"]:

                if poi_type in ["lunch", "dinner"]:

                    # print(poi_info["restaurants"])
                    ranking_idx = self.ranking_restaurants(
                        plan,
                        poi_plan,
                        current_day,
                        current_time,
                        current_position,
                        self.intercity_with_hotel_cost,
                    )
                    ranking_idx = self.reranking_restaurants_with_constraints(
                        plan,
                        poi_type,
                        current_day,
                        current_time,
                        current_position,
                        self.memory["restaurants"],
                        query,
                        ranking_idx,
                    )

                    for sea_i, r_i in enumerate(ranking_idx):

                        if self.search_width != None and sea_i >= self.search_width:
                            print(
                                "Out of search_width [{}], break".format(
                                    self.search_width
                                )
                            )
                            break

                        res_idx = r_i

                        if not (res_idx in self.restaurants_visiting):

                            if res_idx < 0 or res_idx >= len(
                                self.memory["restaurants"]
                            ):
                                print("index error: ", res_idx, len(self.memory["restaurants"]))

                            poi_sel = self.memory["restaurants"].iloc[res_idx]

                            # transports_ranking = self.ranking_innercity_transport(current_position, poi_sel["name"], current_day, current_time)
                            transports_ranking = (
                                self.innercity_transports_ranking_from_query
                            )

                            for trans_type_sel in transports_ranking:
                                self.search_nodes += 1
                                transports_sel = self.collect_innercity_transport(
                                    query["target_city"],
                                    current_position,
                                    poi_sel["name"],
                                    current_time,
                                    trans_type_sel,
                                )
                                if not isinstance(transports_sel, list):
                                    self.backtrack_count += 1
                                    print("inner-city transport error, backtrack...")
                                    continue

                                if len(transports_sel) == 0:
                                    arrived_time = current_time
                                else:
                                    arrived_time = transports_sel[-1]["end_time"]


                                try:
                                    plan = self.add_restaurant(
                                        plan,
                                        poi_type,
                                        poi_sel,
                                        current_day,
                                        arrived_time,
                                        transports_sel,
                                    )
                                except:
                                    self.backtrack_count += 1
                                    print("add_restaurant failed, backtrack...")
                                    continue

                                new_time = plan[current_day]["activities"][-1][
                                    "end_time"
                                ]
                                new_position = poi_sel["name"]
                                self.restaurants_visiting.append(res_idx)
                                self.food_type_visiting.append(poi_sel["cuisine"])
                                success, plan = self.dfs_poi(
                                    query,
                                    poi_plan,
                                    plan,
                                    new_time,
                                    new_position,
                                    current_day,
                                )
                                if success:
                                    return True, plan

                                self.backtrack_count += 1
                                print("add_restaurant failed, backtrack...")

                                plan[current_day]["activities"].pop()
                                self.restaurants_visiting.pop()
                                self.food_type_visiting.pop()

                                # print("res {} fail...".format(poi_sel["name"]))

                elif poi_type == "attraction":
                    ranking_idx = self.ranking_attractions(
                        plan,
                        poi_plan,
                        current_day,
                        current_time,
                        current_position,
                        self.intercity_with_hotel_cost,
                    )

                    ranking_idx = self.reranking_attractions_with_constraints(
                        plan,
                        poi_type,
                        current_day,
                        current_time,
                        current_position,
                        self.memory["attractions"],
                        query,
                        ranking_idx,
                    )

                    for sea_i, r_i in enumerate(ranking_idx):

                        if self.search_width != None and sea_i >= self.search_width:
                            print(
                                "Out of search_width [{}], break".format(
                                    self.search_width
                                )
                            )
                            break
                        self.search_nodes += 1
                        attr_idx = r_i
                        if not (attr_idx in self.attractions_visiting):

                            if attr_idx < 0 or attr_idx >= len(
                                self.memory["attractions"]
                            ):
                                print(attr_idx, len(self.memory["attractions"]))

                            poi_sel = self.memory["attractions"].iloc[attr_idx]
                            # print(current_position, poi_sel["name"])

                            # transports_ranking = self.ranking_innercity_transport(current_position, poi_sel["name"], current_day, current_time)
                            transports_ranking = (
                                self.innercity_transports_ranking_from_query
                            )
                            for trans_type_sel in transports_ranking:
                                self.search_nodes += 1
                                transports_sel = self.collect_innercity_transport(
                                    query["target_city"],
                                    current_position,
                                    poi_sel["name"],
                                    current_time,
                                    trans_type_sel,
                                )
                                if not isinstance(transports_sel, list):
                                    self.backtrack_count += 1
                                    print("inner-city transport error, backtrack...")
                                    continue
                                if len(transports_sel) == 0:
                                    arrived_time = current_time
                                else:
                                    arrived_time = transports_sel[-1]["end_time"]

                                opentime, endtime = (
                                    poi_sel["opentime"],
                                    poi_sel["endtime"],
                                )
                                # too late
                                if time_compare_if_earlier_equal("21:00", arrived_time):
                                    self.backtrack_count += 1
                                    print("The current time is too late...")
                                    continue

                                # it is closed ...
                                if time_compare_if_earlier_equal(endtime, arrived_time):
                                    self.backtrack_count += 1
                                    print("The attraction is closed now...")
                                    continue

                                if time_compare_if_earlier_equal(
                                    arrived_time, opentime
                                ):
                                    act_start_time = opentime
                                else:
                                    act_start_time = arrived_time

                                poi_time = self.select_poi_time(
                                    plan,
                                    poi_plan,
                                    current_day,
                                    act_start_time,
                                    poi_sel["name"],
                                    poi_type,
                                    recommended_visit_time=poi_sel["recommendmintime"]
                                    * 60,
                                )
                                act_end_time = add_time_delta(act_start_time, poi_time)
                                if time_compare_if_earlier_equal(endtime, act_end_time):
                                    act_end_time = endtime

                                plan[current_day]["activities"] = self.add_poi(
                                    activities=plan[current_day]["activities"],
                                    position=poi_sel["name"],
                                    poi_type=poi_type,
                                    price=int(poi_sel["price"]),
                                    cost=int(poi_sel["price"])
                                    * self.query["people_number"],
                                    start_time=act_start_time,
                                    end_time=act_end_time,
                                    innercity_transports=transports_sel,
                                )
                                plan[current_day]["activities"][-1]["tickets"] = (
                                    self.query["people_number"]
                                )

                                new_time = act_end_time
                                new_position = poi_sel["name"]

                                self.attractions_visiting.append(attr_idx)
                                self.spot_type_visiting.append(poi_sel["type"])
                                self.attraction_names_visiting.append(poi_sel["name"])

                                success, plan = self.dfs_poi(
                                    query,
                                    poi_plan,
                                    plan,
                                    new_time,
                                    new_position,
                                    current_day,
                                )

                                if success:
                                    return True, plan

                                self.backtrack_count += 1
                                print("add_attraction failed, backtrack...")

                                plan[current_day]["activities"].pop()
                                self.attractions_visiting.pop()
                                self.spot_type_visiting.pop()
                                self.attraction_names_visiting.pop()

                # The last event in a day: hotel or go-back

                if current_day == query["days"] - 1:

                    # go back

                    if len(plan) < current_day + 1:
                        plan.append({"day": current_day + 1, "activities": []})
                    self.search_nodes += 1
                    # transports_ranking = self.ranking_innercity_transport(current_position, poi_plan["back_transport"]["From"], current_day, current_time)
                    transports_ranking = self.innercity_transports_ranking_from_query
                    for trans_type_sel in transports_ranking:
                        self.search_nodes += 1
                        transports_sel = self.collect_innercity_transport(
                            query["target_city"],
                            current_position,
                            poi_plan["back_transport"]["From"],
                            current_time,
                            trans_type_sel,
                        )
                        if not isinstance(transports_sel, list):
                            self.backtrack_count += 1
                            print("inner-city transport error, backtrack...")
                            continue

                        plan[current_day]["activities"] = self.add_intercity_transport(
                            plan[current_day]["activities"],
                            poi_plan["back_transport"],
                            innercity_transports=transports_sel,
                            tickets=self.query["people_number"],
                        )

                        res_bool, res_plan = self.constraints_validation(
                            query, plan, poi_plan
                        )

                        if res_bool:
                            return True, res_plan
                        else:
                            plan[current_day]["activities"].pop()
                            
                            self.backtrack_count += 1

                            print(
                                "Back-transport, but constraints_validation failed, backtrack..."
                            )
                            # return False, plan

                elif self.query["days"] > 1:
                    # go to hotel
                    hotel_sel = poi_plan["accommodation"]
                    self.search_nodes += 1
                    # transports_ranking = self.ranking_innercity_transport(current_position, hotel_sel["name"], current_day, current_time)
                    transports_ranking = self.innercity_transports_ranking_from_query
                    for trans_type_sel in transports_ranking:
                        self.search_nodes += 1
                        transports_sel = self.collect_innercity_transport(
                            query["target_city"],
                            current_position,
                            hotel_sel["name"],
                            current_time,
                            trans_type_sel,
                        )
                        if not isinstance(transports_sel, list):
                            self.backtrack_count += 1
                            print("inner-city transport error, backtrack...")
                            continue

                        if len(transports_sel) == 0:
                            arrived_time = current_time
                        else:
                            arrived_time = transports_sel[-1]["end_time"]


                        plan = self.add_accommodation(
                            current_plan=plan,
                            hotel_sel=hotel_sel,
                            current_day=current_day,
                            arrived_time=arrived_time,
                            required_rooms=self.required_rooms,
                            transports_sel=transports_sel,
                        )

                        new_time = "00:00"
                        new_position = hotel_sel["name"]

                        success, plan = self.dfs_poi(
                            query,
                            poi_plan,
                            plan,
                            new_time,
                            new_position,
                            current_day + 1,
                        )

                        if success:
                            return True, plan
                        else:
                            self.backtrack_count += 1
                            print("Try the go back hotel, failed, backtrack...")

                            plan[current_day]["activities"].pop()

                            # return False, plan
            else:
                # raise Exception("Not Implemented.")
                print("incorrect poi type: {}".format(poi_type))
                continue

            candidates_type.remove(poi_type)
            print("try another poi type, backtrack...")

        return False, plan

    # generate_plan_with_search: 顶层搜索逻辑。它首先确定城市间的往返交通和酒店，为后续的每日行程打下框架。
    def generate_plan_with_search(self, query):
        # 确保变量在这里被显式清空
        self.least_plan_schema, self.least_plan_comm, self.least_plan_logic = None, None, None
        self.least_plan_logical_pass = -1
        self.least_plan_logic_pvalue = 0
        self.search_nodes = 0
        self.backtrack_count = 0
        self.llm_rec_count = 0
        self.llm_rec_format_error = 0

        source_city = query["start_city"]
        target_city = query["target_city"]

        print(source_city, "->", target_city)

        train_go = self.collect_intercity_transport(source_city, target_city, "train")
        train_back = self.collect_intercity_transport(target_city, source_city, "train")

        flight_go = self.collect_intercity_transport(
            source_city, target_city, "airplane"
        )
        flight_back = self.collect_intercity_transport(
            target_city, source_city, "airplane"
        )

        # print(train_go)
        # print(train_back)
        # print(flight_go)
        # print(flight_back)

        flight_go_num = 0 if flight_go is None else flight_go.shape[0]
        train_go_num = 0 if train_go is None else train_go.shape[0]
        flight_back_num = 0 if flight_back is None else flight_back.shape[0]
        train_back_num = 0 if train_back is None else train_back.shape[0]

        go_info = pd.concat([train_go, flight_go], axis=0)
        back_info = pd.concat([train_back, flight_back], axis=0)

        if self.debug:
            print(
                "from {} to {}: {} flights, {} trains".format(
                    source_city, target_city, flight_go_num, train_go_num
                )
            )
            print(
                "from {} to {}: {} flights, {} trains".format(
                    target_city, source_city, flight_back_num, train_back_num
                )
            )

            print(go_info.head())
            print(back_info.head())

        self.time_before_search = time.time()
        self.llm_inference_time_count = 0

        # reset the cache before searching
        poi_plan = {}
        self.restaurants_visiting = []
        self.attractions_visiting = []
        self.food_type_visiting = []
        self.spot_type_visiting = []
        self.attraction_names_visiting = []
        self.restaurant_names_visiting = []
        self.ranking_attractions_flag = False
        self.ranking_restaurants_flag = False

        self.llm_rec_format_error = 0
        self.llm_rec_count = 0
        self.search_nodes = 0
        self.backtrack_count = 0

        self.constraints_validation_count = 0 
        self.commonsense_pass_count = 0
        self.logical_pass_count = 0
        self.all_constraints_pass = 0

        self.least_plan_schema, self.least_plan_comm, self.least_plan_logic = None, None, None
        self.least_plan_logical_pass = -1
        self.least_plan_logic_pvalue = 0   # 建议初始化这个

        ranking_go = self.ranking_intercity_transport_go(go_info, query)
        ranking_go = self.reranking_intercity_transport_go_with_constraints(
            ranking_go, go_info, query
        )

        ranking_hotel = self.ranking_hotel(self.memory["accommodations"], query)
        query_room_number, query_room_type = self.decide_rooms(query)
        self.required_budget = self.extract_budget(query)

        ranking_hotel = self.reranking_hotel_with_constraints(
            ranking_hotel, self.memory["accommodations"], query, query_room_number
        )

        self.innercity_transports_ranking_from_query = self.ranking_innercity_transport_from_query(query)

        for go_i in ranking_go:
            go_info_i = go_info.iloc[go_i]
            poi_plan["go_transport"] = go_info_i
            self.search_nodes += 1

            ranking_back = self.ranking_intercity_transport_back(
                back_info, query, go_info_i
            )

            ranking_back = self.reranking_intercity_transport_back_with_constraints(
                ranking_back, back_info, query, go_info_i
            )

            for back_i in ranking_back:
                back_info_i = back_info.iloc[back_i]
                poi_plan["back_transport"] = back_info_i
                self.search_nodes += 1

                if query["days"] > 1:
                    for hotel_i in ranking_hotel:
                        poi_plan["accommodation"] = self.memory["accommodations"].iloc[hotel_i]
                        room_type = poi_plan["accommodation"]["numbed"]
                        self.search_nodes += 1

                        required_rooms = (int((query["people_number"] - 1) / room_type) + 1)

                        if query_room_type != None and query_room_type != room_type:
                            self.backtrack_count += 1
                            print("room_type not match, backtrack...")
                            continue

                        if query_room_number != None:
                            required_rooms = query_room_number

                        if query_room_number != None and query_room_type != None:
                            pass
                        else:
                            if (
                                room_type * required_rooms >= query["people_number"]
                            ) and (
                                room_type * required_rooms < query["people_number"] + room_type
                            ):
                                pass
                            else:
                                if query_room_number != None and room_type == 2:
                                    pass
                                else:
                                    self.backtrack_count += 1
                                    print("room_number * room_type not match, backtrack...")
                                continue
                        self.required_rooms = required_rooms

                        self.intercity_with_hotel_cost = (
                            poi_plan["go_transport"]["Cost"]
                            + poi_plan["back_transport"]["Cost"]
                        ) * query["people_number"] + poi_plan["accommodation"][
                            "price"
                        ] * required_rooms * (
                            query["days"] - 1
                        )
                        if (
                            self.required_budget != None
                            and self.required_budget - self.intercity_with_hotel_cost
                            <= self.query["people_number"]
                            * (self.query["days"] - 1)
                            * 100
                        ):
                            self.backtrack_count += 1
                            print("required_budget - intercity_with_hotel_cost <= 100 * people_number * (days-1), backtrack...")
                            continue

                        print("search: ...")
                        try:
                            success, plan = self.dfs_poi(
                                query,
                                poi_plan,
                                plan=[],
                                current_time="",
                                current_position="",
                            )
                        except TimeOutError as e:
                            print("TimeOutError")
                            return False, {"error_info": "TimeOutError"}
                        # exit(0)

                        print(success, plan)
                        if success:
                            return True, plan
                        else:
                            if time.time() > self.time_before_search + self.TIME_CUT:
                                print("Searching TIME OUT !!!")
                                return False, {"error_info": "TimeOutError"}

                            self.backtrack_count += 1
                            print("search failed given the intercity-transport and hotels, backtrack...")

                else:
                    if time_compare_if_earlier_equal(
                        poi_plan["back_transport"]["BeginTime"],
                        poi_plan["go_transport"]["EndTime"],
                    ):
                        self.backtrack_count += 1
                        print("back_transport BeginTime earlier than go_transport EndTime, backtrack...")
                        continue

                    self.intercity_with_hotel_cost = (
                        poi_plan["go_transport"]["Cost"]
                        + poi_plan["back_transport"]["Cost"]
                    ) * query["people_number"]
                    print("search: ...")
                    try:
                        success, plan = self.dfs_poi(
                            query,
                            poi_plan,
                            plan=[],
                            current_time="",
                            current_position="",
                        )
                    except TimeOutError as e:
                        print("TimeOutError")
                        return False, {"error_info": "TimeOutError"}

                    print(success, plan)
                    if success:
                        return True, plan
                    else:
                        if time.time() > self.time_before_search + self.TIME_CUT:
                            print("Searching TIME OUT !!!")
                            return False, {"error_info": "TimeOutError"}
                        
                        self.backtrack_count += 1
                        print("search failed given the intercity-transport and hotels, backtrack...")


        return False, {"error_info": "No solution found."}

    # symbolic_search: 搜索任务的触发器，负责预加载目标城市的酒店、景点、餐厅数据到内存（Memory）中。
    def symbolic_search(self, query):

        # print(query)

        if (query["target_city"] in self.env.support_cities) and (
            query["start_city"] in self.env.support_cities
        ):
            pass
        else:
            return False, {"error_info": f"Unsupported cities {query['start_city']} -> {query['target_city']}."}

        if self.preference_search:
            # print(query["preference_py"])

            preference_py = query["preference_py"][0]
            index = preference_py.find("\n")

            concept = preference_py[:index]
            code = preference_py[index + 1 :]

            # print(concept, code)

            query["preference_opt"] = concept.split(" ")[0]
            query["preference_concept"] = concept.split(" ")[1]
            query["preference_code"] = code
            print(query["preference_opt"], "\n", query["preference_concept"], "\n", query["preference_code"])

            if query["preference_opt"] == "maximize":
                self.least_plan_logic_pvalue = -19260817
            elif query["preference_opt"] == "minimize":
                self.least_plan_logic_pvalue = 19260817
            else:
                raise ValueError("preference_opt must be maximize or minimize")


        
        self.memory["accommodations"] = self.collect_poi_info_all(
            query["target_city"], "accommodation"
        )
        self.memory["attractions"] = self.collect_poi_info_all(
            query["target_city"], "attraction"
        )
        self.memory["restaurants"] = self.collect_poi_info_all(
            query["target_city"], "restaurant"
        )

        # print(query)



        self.query = query

        success, plan = self.generate_plan_with_search(query)

        print(success, plan)

        return success, plan

    def collect_innercity_transport(self, city, start, end, start_time, trans_type):

        call_str = (
            'goto("{city}", "{start}", "{end}", "{start_time}", "{trans_type}")'.format(
                city=city,
                start=start,
                end=end,
                start_time=start_time,
                trans_type=trans_type,
            )
        )

        # print(call_str)
        if start == end:
            return []
        info = self.env(call_str)["data"]

        # print(info)

        if not isinstance(info, list):
            return "No solution"

        if len(info) == 3:
            info[1]["price"] = info[1]["cost"]
            info[1]["tickets"] = self.query["people_number"]
            info[1]["cost"] = info[1]["price"] * info[1]["tickets"]

            info[0]["price"] = info[0]["cost"]
            info[2]["price"] = info[2]["cost"]
        elif info[0]["mode"] == "taxi":
            info[0]["price"] = info[0]["cost"]
            info[0]["cars"] = int((self.query["people_number"] - 1) / 4) + 1
            info[0]["cost"] = info[0]["price"] * info[0]["cars"]
        elif info[0]["mode"] == "walk":
            info[0]["price"] = info[0]["cost"]

        return info

    #collect_xxx: 与 WorldEnv 环境交互，获取真实的航班、高铁时刻表和 POI（兴趣点）详细信息。
    def collect_intercity_transport(self, source_city, target_city, trans_type):

        info_return = self.env(
            "intercity_transport_select('{source_city}', '{target_city}', '{trans_type}')".format(
                source_city=source_city, target_city=target_city, trans_type=trans_type
            )
        )
        if not info_return["success"]:
            return pd.DataFrame([])
        trans_info = info_return["data"]
        # print(poi_info)
        while True:
            info_i = self.env("next_page()")["data"]
            if len(info_i) == 0:
                break
            else:
                trans_info = pd.concat([trans_info, info_i], axis=0, ignore_index=True)
        # print(poi_info)
        return trans_info

    def collect_poi_info_all(self, city, poi_type):

        if poi_type == "accommodation":
            func_name = "accommodations_select"
        elif poi_type == "attraction":
            func_name = "attractions_select"
        elif poi_type == "restaurant":
            func_name = "restaurants_select"
        else:
            raise NotImplementedError

        poi_info = self.env(
            "{func}('{city}', 'name', lambda x: True)".format(func=func_name, city=city)
        )["data"]
        # print(poi_info)
        while True:
            info_i = self.env("next_page()")["data"]
            if len(info_i) == 0:
                break
            else:
                poi_info = pd.concat([poi_info, info_i], axis=0, ignore_index=True)

        # print(poi_info)
        return poi_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="argparse testing")
    parser.add_argument(
        "--splits",
        "-l",
        type=str,
        default="easy",
        choices=["easy", "medium", "human"],
        help="query subset",
    )
    parser.add_argument("--index", "-i", type=int, default=None, help="query index")
    parser.add_argument(
        "--start", "-s", type=int, default=None, help="start query index"
    )
    parser.add_argument(
        "--oracle_translation",
        action="store_true",
        help="Set this flag to enable oracle translation.",
    )
    args = parser.parse_args()

    from evaluation.test import load_query
    from agent.llms import Deepseek
    from environment.world_env import WorldEnv

    env = WorldEnv()

    query_index, query_data = load_query(args)

    # print(query_index, query_data)
    print(len(query_index), "samples")

    agent = NesyAgent(env=env, backbone_llm=Deepseek())

    if args.index is not None:
        query_index = [query_index[args.index]]

    for i, data_idx in enumerate(query_index):

        symbolic_input = query_data[data_idx]
        print(symbolic_input)

        agent.symbolic_search(symbolic_input)
