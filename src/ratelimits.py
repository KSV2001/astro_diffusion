# src/ratelimits.py
import time
import threading
import os
from typing import Tuple, Optional, Dict, Any


class RateLimiter:
    """
    In-memory, multi-scope limiter.

    Enforced (defaults can be overridden by env):
      - per-session: max 5 requests, max 15 minutes
      - per-IP: max 10 req/hour, max 20 req/day, max 3600s/day active time
      - global: max 100 req/hour, max 200 req/day, max 21600s/day active time
      - cost: max 5 USD/day, max 25 USD/month (computed as duration * COST_PER_SEC)

    All counters reset on container restart.
    """

    def __init__(self):
        # load limits from env so you can tune on serverless
        self.per_session_max_req = int(os.getenv("AD_SESSION_MAX_REQ", "5"))
        self.per_session_max_age = int(os.getenv("AD_SESSION_MAX_AGE_SEC", str(15 * 60)))

        self.per_ip_max_req_hour = int(os.getenv("AD_IP_MAX_REQ_HOUR", "10"))
        self.per_ip_max_req_day = int(os.getenv("AD_IP_MAX_REQ_DAY", "20"))
        self.per_ip_max_active_sec_day = int(os.getenv("AD_IP_MAX_ACTIVE_SEC_DAY", str(60 * 60)))  # 1h

        self.global_max_req_hour = int(os.getenv("AD_GLOBAL_MAX_REQ_HOUR", "100"))
        self.global_max_req_day = int(os.getenv("AD_GLOBAL_MAX_REQ_DAY", "200"))
        self.global_max_active_sec_day = int(os.getenv("AD_GLOBAL_MAX_ACTIVE_SEC_DAY", str(6 * 60 * 60)))  # 6h

        self.cost_per_sec = float(os.getenv("AD_COST_PER_SEC", "0.0005"))  # tune to your runpod price
        self.daily_cost_limit = float(os.getenv("AD_DAILY_COST_LIMIT", "5.0"))
        self.monthly_cost_limit = float(os.getenv("AD_MONTHLY_COST_LIMIT", "25.0"))

        self._lock = threading.Lock()

        # per-ip buckets
        self._ip_hour: Dict[str, Dict[str, float]] = {}
        self._ip_day: Dict[str, Dict[str, float]] = {}

        # global buckets
        self._global_hour = {"count": 0, "reset_at": time.time() + 3600}
        self._global_day = {"count": 0, "active_sec": 0.0, "cost": 0.0, "reset_at": time.time() + 86400}
        # month is coarse
        self._global_month = {"cost": 0.0, "reset_at": time.time() + 30 * 86400}

    # ------------- internal bucket helpers -------------

    def _get_ip_hour_bucket(self, ip: str):
        now = time.time()
        b = self._ip_hour.get(ip)
        if b is None or now >= b["reset_at"]:
            b = {"count": 0, "reset_at": now + 3600}
            self._ip_hour[ip] = b
        return b

    def _get_ip_day_bucket(self, ip: str):
        now = time.time()
        b = self._ip_day.get(ip)
        if b is None or now >= b["reset_at"]:
            b = {"count": 0, "active_sec": 0.0, "reset_at": now + 86400}
            self._ip_day[ip] = b
        return b

    def _get_global_hour(self):
        now = time.time()
        g = self._global_hour
        if now >= g["reset_at"]:
            g["count"] = 0
            g["reset_at"] = now + 3600
        return g

    def _get_global_day(self):
        now = time.time()
        g = self._global_day
        if now >= g["reset_at"]:
            g["count"] = 0
            g["active_sec"] = 0.0
            g["cost"] = 0.0
            g["reset_at"] = now + 86400
        return g

    def _get_global_month(self):
        now = time.time()
        g = self._global_month
        if now >= g["reset_at"]:
            g["cost"] = 0.0
            g["reset_at"] = now + 30 * 86400
        return g

    # ------------- public API -------------

    def pre_check(
        self,
        ip: str,
        session_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check count-based and time-window limits BEFORE doing work.
        session_state must contain:
            {
              "count": int,
              "started_at": float
            }
        """
        now = time.time()
        with self._lock:
            # session checks
            sess_count = int(session_state.get("count", 0))
            sess_started = float(session_state.get("started_at", now))
            if sess_count >= self.per_session_max_req:
                return False, f"session request cap {self.per_session_max_req} reached"
            if now - sess_started > self.per_session_max_age:
                return False, "session time cap reached (15 min)"

            # ip checks
            ip_h = self._get_ip_hour_bucket(ip)
            if ip_h["count"] >= self.per_ip_max_req_hour:
                return False, f"ip hourly cap {self.per_ip_max_req_hour} reached"

            ip_d = self._get_ip_day_bucket(ip)
            if ip_d["count"] >= self.per_ip_max_req_day:
                return False, f"ip daily cap {self.per_ip_max_req_day} reached"
            if ip_d["active_sec"] >= self.per_ip_max_active_sec_day:
                return False, "ip daily active time cap reached"

            # global checks
            g_h = self._get_global_hour()
            if g_h["count"] >= self.global_max_req_hour:
                return False, "global hourly cap reached"

            g_d = self._get_global_day()
            if g_d["count"] >= self.global_max_req_day:
                return False, "global daily cap reached"
            if g_d["active_sec"] >= self.global_max_active_sec_day:
                return False, "global daily active time cap reached"
            if g_d["cost"] >= self.daily_cost_limit:
                return False, "global daily cost cap reached"

            g_m = self._get_global_month()
            if g_m["cost"] >= self.monthly_cost_limit:
                return False, "global monthly cost cap reached"

            # if all clear, tentatively increment counts that don't need duration
            ip_h["count"] += 1
            ip_d["count"] += 1
            g_h["count"] += 1
            g_d["count"] += 1

            # session count increment here
            session_state["count"] = sess_count + 1

            # return ok
            return True, None

    def post_consume(
        self,
        ip: str,
        duration_sec: float,
    ) -> None:
        """
        Update time-based and cost-based buckets AFTER doing work.
        """
        cost = duration_sec * self.cost_per_sec
        with self._lock:
            ip_d = self._get_ip_day_bucket(ip)
            ip_d["active_sec"] += duration_sec

            g_d = self._get_global_day()
            g_d["active_sec"] += duration_sec
            g_d["cost"] += cost

            g_m = self._get_global_month()
            g_m["cost"] += cost
