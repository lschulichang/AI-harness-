"""原子文件、SQLite 实验账本及单运行锁。密钥不写入账本。"""

import json
import os
import sqlite3
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Ledger:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS records(kind TEXT, id TEXT, value TEXT, PRIMARY KEY(kind,id))")
        self.db.execute("CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, value TEXT)")
        self.db.commit()

    def get(self, kind, key):
        row = self.db.execute("SELECT value FROM records WHERE kind=? AND id=?", (kind, key)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind, key, value):
        self.db.execute("INSERT OR REPLACE INTO records VALUES(?,?,?)", (kind, key, json.dumps(value, ensure_ascii=False)))
        self.db.commit()

    def put_trial_decision(self, trial, decision):
        with self.db:
            for kind, value in (("trial", trial), ("decision", decision)):
                self.db.execute("INSERT OR REPLACE INTO records VALUES(?,?,?)",
                                (kind, value["id"], json.dumps(value, ensure_ascii=False)))

    def items(self, kind):
        return [json.loads(r[0]) for r in self.db.execute("SELECT value FROM records WHERE kind=? ORDER BY id", (kind,))]

    def count(self, kind):
        return self.db.execute("SELECT COUNT(*) FROM records WHERE kind=?", (kind,)).fetchone()[0]

    def request(self, key):
        row = self.db.execute("SELECT value FROM requests WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def reserve_request(self, key, limit, value):
        # 先落盘扣一次预算，再发网络请求；进程崩溃也不会忘记已发出的请求。
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.db.execute("SELECT 1 FROM requests WHERE id=?", (key,)).fetchone():
                return False
            if limit is not None and self.api_calls() >= limit:
                return False
            self.db.execute("INSERT INTO requests VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))
        return True

    def finish_request(self, key, value):
        self.db.execute("UPDATE requests SET value=? WHERE id=?", (json.dumps(value, ensure_ascii=False), key))
        self.db.commit()

    def requests(self):
        return [json.loads(r[0]) for r in self.db.execute("SELECT value FROM requests ORDER BY id")]

    def api_calls(self):
        return self.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

    def close(self):
        self.db.close()


class RunLock:
    """防止两个进程同时恢复同一次运行；进程退出后操作系统自动释放。"""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.file = open(self.path, "a+b")
        self.file.seek(0)
        if os.name == "nt":
            import msvcrt
            if self.file.read(1) == b"":
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            try:
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                self.file.close()
                raise RuntimeError("该运行正在被另一个进程使用") from None
        else:
            import fcntl
            try:
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self.file.close()
                raise RuntimeError("该运行正在被另一个进程使用") from None
        return self

    def __exit__(self, *args):
        self.file.close()
