import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def load():
    import sys as _sys

    _sys.path.insert(0, "src")
    from basic import experiment as _experiment

    results = _experiment.results()
    return (results,)


@app.cell
def summary(results):
    import marimo as mo
    _best = max(results, key=lambda exp: exp.result["accuracy"])

    mo.md(
        f"""
        # Basic Experiment Results

        Loaded **{len(list(results))} runs** from the latest experiment.

        Best: **{_best.name}** with accuracy **{_best.result['accuracy']:.4f}**
        """
    )
    return


@app.cell
def plot(results):
    import matplotlib.pyplot as _plt

    _names = [exp.name for exp in results]
    _accuracies = [exp.result["accuracy"] for exp in results]

    _fig, _ax = _plt.subplots(figsize=(8, 5))
    _bars = _ax.bar(_names, _accuracies, color=["#4e79a7", "#f28e2b", "#e15759"])
    _ax.set_ylabel("Accuracy")
    _ax.set_title("Accuracy by Configuration")
    _ax.set_ylim(0, max(_accuracies) * 1.15)

    for _bar, _acc in zip(_bars, _accuracies):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _bar.get_height() + 0.002,
            f"{_acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    _fig.tight_layout()

    _fig.gca()
    return


if __name__ == "__main__":
    app.run()
