import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    plt.ioff()
    return


@app.cell
def _():
    import sys as _sys

    _sys.path.insert(0, "src")
    from demo_logs import train as _train
    from pyexp import LogReader

    results = _train.results()
    readers = {_exp.name: LogReader(_exp.out) for _exp in results}
    return (results, readers)


@app.cell
def _(mo, results):
    _lines = []
    for _exp in results:
        if _exp.error:
            _lines.append(f"- **{_exp.name}**: ERROR - {_exp.error}")
        else:
            _r = _exp.result
            _lines.append(
                f"- **{_exp.name}**: loss={_r['final_loss']:.4f}, accuracy={_r['final_accuracy']:.4f}"
            )

    mo.md(f"""
# Demo Logs Report

Loaded **{len(results)} runs** from the latest experiment.

{chr(10).join(_lines)}
""")
    return


@app.cell
def _(mo, plt, readers):
    loss_fig, loss_ax = plt.subplots(figsize=(10, 5))
    for _name, _reader in readers.items():
        _its, _vals = _reader.scalars("loss")
        loss_ax.plot(_its, _vals, label=_name)
    loss_ax.set_xlabel("Iteration")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Loss Curves")
    loss_ax.legend()
    loss_fig.tight_layout()
    mo.mpl.interactive(loss_fig)
    return


@app.cell
def _(mo, plt, readers):
    acc_fig, acc_ax = plt.subplots(figsize=(10, 5))
    for _name, _reader in readers.items():
        _its, _vals = _reader.scalars("accuracy")
        acc_ax.plot(_its, _vals, label=_name)
    acc_ax.set_xlabel("Iteration")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title("Accuracy Curves")
    acc_ax.legend()
    acc_fig.tight_layout()
    mo.mpl.interactive(acc_fig)
    return


@app.cell
def _(mo, plt, readers):
    lr_fig, lr_ax = plt.subplots(figsize=(10, 5))
    for _name, _reader in readers.items():
        _its, _vals = _reader.scalars("learning_rate")
        lr_ax.plot(_its, _vals, label=_name)
    lr_ax.set_xlabel("Iteration")
    lr_ax.set_ylabel("Learning Rate")
    lr_ax.set_title("Learning Rate Schedule")
    lr_ax.legend()
    lr_fig.tight_layout()
    mo.mpl.interactive(lr_fig)
    return


@app.cell
def _(mo, plt, readers):
    _items = []
    for _name, _reader in readers.items():
        for _tag in sorted(_reader.figure_tags):
            _its, _figs = _reader.figures(_tag)
            _items.append(mo.md(f"**{_name}** / {_tag} (iteration {_its[-1]})"))
            _items.append(mo.mpl.interactive(_figs[-1]))

    mo.vstack([mo.md("## Logged Figures (last iteration)"), *_items])
    return


if __name__ == "__main__":
    app.run()
