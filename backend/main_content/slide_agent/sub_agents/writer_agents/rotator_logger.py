import logging
import os
import time
import gzip
import shutil
import glob
from datetime import datetime, timedelta

class DailyCompressingHandler(logging.Handler):
    """
    每天写到 writer_agent_YYYYMMDD.log。
    - 启动时会压缩历史未压缩日志（非当天）。
    - 每次跨日时会压缩前一天的日志。
    - 可调用 cleanup_old_logs 删除超过 keep_days 的历史文件（.log/.log.gz）。
    适合单进程/单实例程序；多进程请使用集中式或并发安全的 handler。
    """
    def __init__(self, log_dir='.', prefix='writer_agent', encoding='utf-8', keep_days=7):
        super().__init__()
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.prefix = prefix
        self.encoding = encoding
        self.current_date = None   # YYYYMMDD
        self._file_handler = None
        self.keep_days = int(keep_days)

        # 启动时压缩历史未压缩文件（但不压缩当天的文件）
        self.compress_uncompressed_historical()

    def _filename_for_date(self, date_str):
        return os.path.join(self.log_dir, f"{self.prefix}_{date_str}.log")

    def _open_for_today(self):
        today = datetime.now().strftime("%Y%m%d")
        if self.current_date == today and self._file_handler:
            return

        # 记录之前的日期（用于压缩）
        prev_date = self.current_date

        # 关闭旧 handler（如果有）
        if self._file_handler:
            try:
                self._file_handler.close()
            except Exception:
                pass
            self._file_handler = None

        # 打开当日文件（追加模式）
        filename = self._filename_for_date(today)
        fh = logging.FileHandler(filename, encoding=self.encoding, mode='a')
        if self.formatter:
            fh.setFormatter(self.formatter)
        fh.setLevel(self.level)
        self._file_handler = fh
        self.current_date = today

        # 如果有前一日（prev_date），尝试压缩它（如果未压缩）
        if prev_date:
            prev_path = self._filename_for_date(prev_date)
            try:
                self._safe_compress(prev_path)
            except Exception:
                logging.getLogger(__name__).exception("compress prev day failed for %s", prev_path)

        # 每次切换也顺便清理过期文件
        try:
            self.cleanup_old_logs(self.keep_days)
        except Exception:
            logging.getLogger(__name__).exception("cleanup failed")

    def emit(self, record):
        try:
            self._open_for_today()
            self._file_handler.emit(record)
        except Exception:
            self.handleError(record)

    def setFormatter(self, fmt):
        super().setFormatter(fmt)
        if self._file_handler:
            self._file_handler.setFormatter(fmt)

    def setLevel(self, level):
        super().setLevel(level)
        if self._file_handler:
            self._file_handler.setLevel(level)

    def close(self):
        try:
            if self._file_handler:
                self._file_handler.close()
        finally:
            super().close()

    # --------- 辅助方法 ----------
    def _safe_compress(self, path):
        """
        将指定 path（.log）压缩为 path + '.gz'（原子替换）。
        不会压缩当日日志（调用者需确保）。
        """
        if not os.path.exists(path):
            return
        if path.endswith(".gz"):
            return
        gz_path = path + ".gz"
        # 如果已存在 .gz，则删除原文件（避免重复占用）
        if os.path.exists(gz_path):
            try:
                os.remove(path)
            except OSError:
                pass
            return

        tmp_path = gz_path + ".tmp"
        with open(path, "rb") as f_in, gzip.open(tmp_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        # 原子替换：先移除同名（若存在），再重命名
        try:
            os.replace(tmp_path, gz_path)
            os.remove(path)
        except Exception:
            # 若 rename 失败，清理 tmp 并抛出
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise

    def compress_uncompressed_historical(self):
        """
        压缩所有历史 .log（非当天）为 .gz（如果尚未压缩）。
        在启动时调用一次，避免遗留未压缩文件堆积。
        """
        today = datetime.now().strftime("%Y%m%d")
        pattern = os.path.join(self.log_dir, f"{self.prefix}_*.log")
        for path in glob.glob(pattern):
            # 跳过当日日志
            if path.endswith("_" + today + ".log"):
                continue
            # 跳过已经压缩（应该没后缀 .log，但保险起见）
            if path.endswith(".gz"):
                continue
            try:
                self._safe_compress(path)
            except Exception:
                logging.getLogger(__name__).exception("startup compress failed for %s", path)

    def cleanup_old_logs(self, keep_days):
        """
        删除超过 keep_days 的历史日志（按 mtime 判断），包括 .log 和 .log.gz。
        keep_days: 保留的天数（≥0）。0 表示删除所有早于今天的文件。
        """
        keep_days = int(keep_days)
        cutoff = time.time() - (keep_days * 86400)
        # 匹配 .log 与 .log.gz
        patterns = [
            os.path.join(self.log_dir, f"{self.prefix}_*.log"),
            os.path.join(self.log_dir, f"{self.prefix}_*.log.gz"),
        ]
        for pat in patterns:
            for path in glob.glob(pat):
                try:
                    # 不删除当天日志（根据文件名）
                    fname = os.path.basename(path)
                    if fname.startswith(self.prefix + "_" + datetime.now().strftime("%Y%m%d")):
                        continue
                    mtime = os.path.getmtime(path)
                    if mtime < cutoff:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                except Exception:
                    logging.getLogger(__name__).exception("failed to evaluate/remove %s", path)

# ---------- 帮助函数：设置 logger ----------
def setup_daily_logger(log_dir="logs", prefix="writer_agent", keep_days=7):
    logger = logging.getLogger("writer_agent")
    logger.setLevel(logging.INFO)
    # 清理老 handler（便于交互式/重复运行）
    if logger.handlers:
        logger.handlers.clear()

    handler = DailyCompressingHandler(log_dir=log_dir, prefix=prefix, keep_days=keep_days)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(fmt)
    handler.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.propagate = False
    return logger

# ---------- main 示例 ----------
def main():
    # 配置：日志目录、前缀、保留天数
    LOG_DIR = "logs"
    PREFIX = "writer_agent"
    KEEP_DAYS = 7  # 保留最近 7 天；改成 0 测试会删除今天之前的所有历史文件

    logger = setup_daily_logger(log_dir=LOG_DIR, prefix=PREFIX, keep_days=KEEP_DAYS)

    print(f"开始写日志。日志目录: {os.path.abspath(LOG_DIR)} 。保留最近 {KEEP_DAYS} 天的日志（包含 .gz）。")
    try:
        i = 0
        while True:
            logger.info("示例日志 %05d - 按天保存，历史自动压缩并清理（keep_days=%d）", i, KEEP_DAYS)
            i += 1
            time.sleep(0.5)  # 写频率：示例使用 0.5s；实际可改成 2s 或更慢
    except KeyboardInterrupt:
        print("已停止（Ctrl+C）。")

if __name__ == "__main__":
    main()
