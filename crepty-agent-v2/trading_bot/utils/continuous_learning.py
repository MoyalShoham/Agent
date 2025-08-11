"""
Continuous Learning Module
Retrains ML/meta-learner models on new data and deploys updates.
"""
import subprocess, logging, os, time

def retrain_meta_learner(trade_log_path, model_path):
    """Invoke training script; if succeeds, log mtime & size for monitoring.
    Non-blocking quick return; training runs synchronously here (could be async later).
    """
    start = time.time()
    try:
        result = subprocess.run(["python", "train_meta_learner.py"], capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logging.warning(f"[CONT_LEARN] Meta learner retrain failed rc={result.returncode} stderr={result.stderr[-400:]}" )
            return False
        if os.path.exists(model_path):
            stat = os.stat(model_path)
            logging.info(f"[CONT_LEARN] Meta model updated size={stat.st_size} mtime={stat.st_mtime} elapsed={time.time()-start:.1f}s")
            return True
        logging.warning("[CONT_LEARN] Training completed but model file missing")
        return False
    except subprocess.TimeoutExpired:
        logging.error("[CONT_LEARN] Meta learner training timed out")
    except Exception as e:
        logging.exception(f"[CONT_LEARN] retrain_meta_learner error: {e}")
    return False
