"""大模型请求、有限重试、请求账本和 Token 预算。"""

import json
import time


def request_usage_known(request):
    if request.get("status") != "received":
        return False
    return request.get("usage_known", all(key in request for key in ("prompt_tokens", "completion_tokens")))


class Planner:
    def __init__(self, config, ledger, transport=None):
        self.config, self.ledger = config, ledger
        self.transport = transport


    def _request_proposal(self, key, system, context, validate, *, output_budget=None):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
        error = ""
        for attempt in range(2):
            request_id = f"{key}_request_{attempt}"
            request = self.ledger.request(request_id)
            if request is None:
                reservation = {"id": request_id, "trial_id": key, "status": "reserved", "context": context,
                               "model": self.config.model}
                if output_budget is not None:
                    remaining_tokens = output_budget - self.accounted_output_tokens()
                    limit = self.config.max_output_tokens or remaining_tokens
                    if limit <= 0 or limit > remaining_tokens:
                        return None, "API 输出 Token 预算不足"
                    reservation["reserved_output_tokens"] = limit
                if not self.ledger.reserve_request(request_id, self.config.max_api_calls, reservation):
                    return None, "API 请求预算已用尽"
                started = time.perf_counter()
                try:
                    response = self.transport(messages)
                    request = {**reservation, "status": "received", **response}
                except Exception as exc:
                    request = {**reservation, "status": "error", "error_type": type(exc).__name__}
                request["elapsed_seconds"] = time.perf_counter() - started
                self.ledger.finish_request(request_id, request)
            if request["status"] == "reserved":
                error = "上次请求结果未知，仍计入预算"
            elif request["status"] == "error":
                error = "API 请求失败：" + request["error_type"]
            else:
                try:
                    return validate(json.loads(request["content"])), ""
                except (ValueError, TypeError, KeyError) as exc:
                    error = str(exc)
            messages = messages[:2] + [{"role": "user", "content": "请修正方案并返回 JSON：" + error}]
        return None, error

    def accounted_output_tokens(self):
        return sum(row.get("completion_tokens") or row.get("reserved_output_tokens", 0)
                   for row in self.ledger.requests())
