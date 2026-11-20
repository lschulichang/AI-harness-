"""大模型请求、有限重试、请求账本和 Token 预算。"""

import json
import time

from .credentials import get_secret


def request_usage_known(request):
    if request.get("status") != "received":
        return False
    return request.get("usage_known", all(key in request for key in ("prompt_tokens", "completion_tokens")))


class OpenAITransport:
    def __init__(self, config, *, temperature=0.2, default_max_tokens=None):
        from openai import OpenAI
        key = get_secret(config.api_key_env)
        if not key:
            raise ValueError(f"请在环境变量或 local_secrets.py 中配置 {config.api_key_env}")
        self.config = config
        self.temperature = temperature
        self.default_max_tokens = default_max_tokens
        # 禁用 SDK 隐式重试，使每个请求都能计入我们自己的预算。
        self.client = OpenAI(api_key=key, base_url=config.base_url, timeout=config.api_timeout, max_retries=0)

    def __call__(self, messages, *, max_tokens=None):
        kwargs = {}
        limit = self.config.max_output_tokens if max_tokens is None else max_tokens
        if limit is None:
            limit = self.default_max_tokens
        if limit is not None:
            kwargs["max_tokens"] = limit
        if self.config.provider == "deepseek":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = self.client.chat.completions.create(
            model=self.config.model, messages=messages, response_format={"type": "json_object"},
            temperature=self.temperature, **kwargs,
        )
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        return {
            "content": response.choices[0].message.content or "",
            "prompt_tokens": prompt_tokens or 0, "completion_tokens": completion_tokens or 0,
            "usage_known": prompt_tokens is not None and completion_tokens is not None,
        }


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
                    if self.transport is None:
                        self.transport = OpenAITransport(self.config)
                    if isinstance(self.transport, OpenAITransport):
                        response = self.transport(messages, max_tokens=reservation.get("reserved_output_tokens"))
                    else:
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
