"""Optional model diagnostics on completed public validation batches."""

from .gas_data import evaluate_samples, gas_samples

HISTORY_WINDOW = 6


def history_batches(completed, *, enabled):
    """Return the validation window without reading samples when disabled."""
    return completed[-HISTORY_WINDOW:] if enabled else []


def gas_references(data, batches, entry_path):
    references = []
    for batch in batches:
        samples = gas_samples(data, batch, "validation")
        entry_f1 = evaluate_samples(entry_path, samples)["macro_f1"]
        references.append((data.batch_names[batch], samples, entry_f1))
    return references


def gas_scores(path, references):
    scores = []
    for name, samples, entry_f1 in references:
        candidate_f1 = evaluate_samples(path, samples)["macro_f1"]
        scores.append({
            "source": name,
            "entry_macro_f1": entry_f1,
            "candidate_macro_f1": candidate_f1,
            "drop": entry_f1 - candidate_f1,
        })
    return scores


def rul_references(adapter, batches, entry_model):
    references = []
    for batch in batches:
        samples = adapter.samples(batch["batch"])
        entry_rmse = adapter.metrics(entry_model, samples)["rmse"]
        references.append((batch["source_file"], samples, entry_rmse))
    return references


def rul_scores(adapter, model, references):
    scores = []
    for group, samples, entry_rmse in references:
        candidate = adapter.metrics(model, samples)["rmse"]
        scores.append({
            "group": group,
            "entry_rmse": entry_rmse,
            "candidate_rmse": candidate,
            "relative_increase": (candidate - entry_rmse) / max(entry_rmse, 1e-12),
        })
    return scores
