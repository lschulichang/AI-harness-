"""读取平台提供的气体分类检查点。"""

def load_gas_model(checkpoint):
    from baseline.gas.baselines.model import Model

    model = Model.load(checkpoint)
    if model.meta["task"] != "classification":
        raise ValueError("气体任务需要分类模型检查点")
    return model
