from .register import Registry

# Đăng ký các mô hình
META_TASK = Registry("TASK")

def build_task(config):
    task = META_TASK.get(config.task)(config)
    return task
