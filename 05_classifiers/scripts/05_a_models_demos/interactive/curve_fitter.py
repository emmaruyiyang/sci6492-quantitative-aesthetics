"""Curve fitter: draw a curve, then build a small network that traces it.

The drawn stroke is resampled into N points p_i = [x_i, y_i]. Each point gets a
parameter value t_i = i / (N-1), so the target is a parametric curve

    t in [0, 1]  ->  [x, y]

The model is a stack of layers that maps a tensor of shape [N, 1] (the t
values) to a tensor of shape [N, 2] (the predicted points). The loss compares
the predicted and the drawn points, either as the mean squared error (MSE) or
the mean absolute error (L1, less pulled around by a few far off points).

Layer types (built in the browser, sent here as JSON):

  linear    torch.nn.Linear(in, out). "in" is whatever the previous layer
            produced, "out" is chosen by the user.
  relu      torch.relu, keeps the shape.
  function  one or more formulas separated by ";" or new lines, e.g.

                a1*cos(x*f1) + a2*sin(x*f2)

            x is the whole input of the layer [N, D] and x0, x1, ... are its
            columns [N, 1]. Known functions (sin, cos, exp, ...) and the
            constants pi and e are looked up, every other name becomes a
            trainable scalar parameter. Each formula gives one or more output
            columns, all formulas are concatenated side by side:

                a*cos(2*pi*x) ; b*sin(2*pi*x)      [N, 1] -> [N, 2]

Run it and open http://127.0.0.1:5000
"""

import ast
import math
import re
import threading
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"

CURVE_SAMPLES = 400     # how finely the model curve is drawn
REPORT_EVERY = 10       # optimization steps between snapshots sent to the browser

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Formula parsing
# ---------------------------------------------------------------------------

# The only functions a formula may call. Anything else is rejected, so a formula
# can't reach torch.save, open, ... (see check_node below).
FUNCTIONS = {
    name: getattr(torch, name) for name in [
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "exp", "log", "log2", "log10", "sqrt",
        "abs", "sign", "floor", "ceil", "round", "sigmoid", "relu",
        "minimum", "maximum", "erf", "sinc", "frac",
    ]
}
FUNCTIONS["softplus"] = torch.nn.functional.softplus
FUNCTIONS["sq"] = torch.square
FUNCTIONS["pow"] = torch.pow
CONSTANTS = {"pi": math.pi, "e": math.e}
INPUT_NAME = re.compile(r"^x\d*$")

ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
)


class FormulaError(ValueError):
    pass


def split_formulas(text):
    """'a ; b \\n c' -> ['a', 'b', 'c']"""
    return [part.strip() for part in re.split(r"[;\n]", text) if part.strip()]


def parse_formula(text):
    """Compile every formula in text and find the names of its parameters.

    Returns (list of code objects, list of parameter names in order of appearance).
    """
    formulas = split_formulas(text)
    if not formulas:
        raise FormulaError("empty formula")

    codes, params = [], []
    for formula in formulas:
        # Allow ^ for powers, as people tend to write it.
        source = formula.replace("^", "**")
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as err:
            raise FormulaError(f"syntax error in '{formula}': {err.msg}") from None

        for node in ast.walk(tree):
            check_node(node, formula)
        # Parameters in the order they are written.
        names = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Name)), key=lambda n: n.col_offset)
        for node in names:
            name = node.id
            if name in FUNCTIONS or name in CONSTANTS or INPUT_NAME.match(name):
                continue
            if name not in params:
                params.append(name)

        codes.append(compile(tree, "<formula>", "eval"))
    return codes, params


def check_node(node, formula):
    if not isinstance(node, ALLOWED_NODES):
        raise FormulaError(f"'{type(node).__name__}' is not allowed in '{formula}'")
    if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
        raise FormulaError(f"only numbers are allowed as constants in '{formula}'")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = getattr(node.func, "id", "?")
            raise FormulaError(f"unknown function '{name}'")
        if node.keywords:
            raise FormulaError("keyword arguments are not allowed")


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class FormulaLayer(torch.nn.Module):
    def __init__(self, text, values):
        super().__init__()
        self.codes, self.names = parse_formula(text)
        self.params = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.tensor(float(values.get(name, 1.0))))
            for name in self.names
        })

    def forward(self, x):
        env = dict(FUNCTIONS)
        env.update(CONSTANTS)
        env["x"] = x
        for i in range(x.shape[1]):
            env[f"x{i}"] = x[:, i:i + 1]
        env.update(self.params)

        zeros = torch.zeros(x.shape[0], 1)
        outputs = []
        for code in self.codes:
            y = eval(code, {"__builtins__": {}}, env)
            # Constants and scalar parameters become a column, [N, 1] stays, [N, D] stays.
            outputs.append(torch.as_tensor(y, dtype=torch.float32) + zeros)
        return torch.cat(outputs, dim=1)


class Relu(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)


def build_model(layers):
    """Turn the JSON layer list into torch modules.

    Returns (modules, info) where info has one dict per layer for the browser:
    its output width, an error message or its current parameter values.
    Stops at the first broken layer (modules is then None).
    """
    modules, info = [], []
    width = 1
    for spec in layers:
        kind = spec.get("type")
        entry = {"id": spec.get("id")}
        info.append(entry)
        try:
            if kind == "linear":
                out = int(spec.get("out", 8))
                if out < 1:
                    raise FormulaError("out features must be at least 1")
                module = torch.nn.Linear(width, out)
                params = spec.get("params") or {}
                weight, bias = params.get("weight"), params.get("bias")
                # Keep the stored weights unless the shape changed (then start fresh).
                if weight is not None and bias is not None:
                    weight, bias = torch.tensor(weight), torch.tensor(bias)
                    if weight.shape == module.weight.shape and bias.shape == module.bias.shape:
                        with torch.no_grad():
                            module.weight.copy_(weight)
                            module.bias.copy_(bias)
            elif kind == "relu":
                module = Relu()
            elif kind == "function":
                values = {k: v.get("value", 1.0) for k, v in (spec.get("params") or {}).items()}
                module = FormulaLayer(spec.get("expr", ""), values)
            else:
                raise FormulaError(f"unknown layer type '{kind}'")

            # A dry run tells us the output width and catches shape errors early.
            with torch.no_grad():
                out = module(torch.zeros(3, width))
            if out.dim() != 2 or out.shape[0] != 3:
                raise FormulaError(f"output has shape {list(out.shape)}")
        except Exception as err:            # anything the formula does wrong ends up here
            entry["error"] = str(err)
            return None, info

        width = out.shape[1]
        entry["in"] = int(module.weight.shape[1]) if kind == "linear" else None
        entry["out"] = int(width)
        modules.append(module)
    return modules, info


def run(modules, t):
    for module in modules:
        t = module(t)
    return t


def export_params(modules, info):
    """Write the current parameter values into the per-layer info dicts."""
    for module, entry in zip(modules, info):
        if isinstance(module, torch.nn.Linear):
            entry["weight"] = module.weight.detach().tolist()
            entry["bias"] = module.bias.detach().tolist()
        elif isinstance(module, FormulaLayer):
            entry["names"] = module.names
            entry["values"] = {k: float(p.detach()) for k, p in module.params.items()}


def t_values(n):
    return torch.linspace(0, 1, n).unsqueeze(1)


def predictions(modules, n):
    """Model curve (finely sampled) and the predictions at the N target t values."""
    with torch.no_grad():
        curve = run(modules, t_values(CURVE_SAMPLES))
        pred = run(modules, t_values(n)) if n > 1 else torch.zeros(0, 2)
    return curve, pred


# ---------------------------------------------------------------------------
# Optimization job (one at a time, in a background thread)
# ---------------------------------------------------------------------------

# name shown in the browser -> loss between prediction and target
LOSSES = {
    "MSE": lambda pred, target: torch.mean((pred - target) ** 2),
    "L1": lambda pred, target: torch.mean(torch.abs(pred - target)),
}

# name shown in the browser -> (how to make it, its momentum-like settings with defaults)
# beta1 is Adam's momentum (how much of the past gradient direction is kept),
# beta2 how long it remembers the size of the gradients.
ADAM_BETAS = {"beta1": 0.9, "beta2": 0.999}

OPTIMIZERS = {
    "Adam": (lambda params, lr, h: torch.optim.Adam(params, lr=lr, betas=(h["beta1"], h["beta2"])), ADAM_BETAS),
    "AdamW": (lambda params, lr, h: torch.optim.AdamW(params, lr=lr, betas=(h["beta1"], h["beta2"])), ADAM_BETAS),
    "NAdam": (lambda params, lr, h: torch.optim.NAdam(params, lr=lr, betas=(h["beta1"], h["beta2"])), ADAM_BETAS),
    "Adamax": (lambda params, lr, h: torch.optim.Adamax(params, lr=lr, betas=(h["beta1"], h["beta2"])), ADAM_BETAS),
    "RMSprop": (lambda params, lr, h: torch.optim.RMSprop(params, lr=lr, momentum=h["momentum"], alpha=h["alpha"]),
                {"momentum": 0.0, "alpha": 0.99}),
    "SGD": (lambda params, lr, h: torch.optim.SGD(params, lr=lr, momentum=h["momentum"]), {"momentum": 0.9}),
    "Adagrad": (lambda params, lr, h: torch.optim.Adagrad(params, lr=lr), {}),
    # LBFGS does several evaluations per step, so each step is slower but bigger.
    "LBFGS": (lambda params, lr, h: torch.optim.LBFGS(params, lr=lr, max_iter=5, line_search_fn="strong_wolfe"), {}),
}


def make_optimizer(name, params, lr, settings):
    factory, defaults = OPTIMIZERS[name]
    h = {key: float(settings.get(key, value)) for key, value in defaults.items()}
    return factory(params, lr, h)


job_lock = threading.Lock()
job = {"running": False, "stop": False, "losses": [], "snapshot": None, "error": None, "step": 0}


def optimize(modules, info, layers, points, steps, lr, clamp, optimizer_name, settings, loss_name):
    target = torch.tensor(points, dtype=torch.float32)
    t = t_values(len(points))

    params = [p for m in modules for p in m.parameters()]
    optimizer = make_optimizer(optimizer_name, params, lr, settings)

    # Formula parameters can be kept inside their slider ranges.
    bounds = []
    if clamp:
        for module, spec in zip(modules, layers):
            if isinstance(module, FormulaLayer):
                ranges = spec.get("params") or {}
                for name, p in module.params.items():
                    r = ranges.get(name) or {}
                    if "min" in r and "max" in r:
                        bounds.append((p, float(r["min"]), float(r["max"])))

    def snapshot():
        export_params(modules, info)
        curve, pred = predictions(modules, len(points))
        return {"layers": info, "curve": curve.tolist(), "pred": pred.tolist()}

    # All optimizers accept a closure that recomputes the loss; LBFGS needs it.
    def closure():
        optimizer.zero_grad()
        loss = LOSSES[loss_name](run(modules, t), target)
        loss.backward()
        return loss

    try:
        for step in range(1, steps + 1):
            if job["stop"]:
                break
            loss = optimizer.step(closure)
            if not torch.isfinite(loss):
                raise FloatingPointError("loss is not finite (nan or inf), try a smaller learning rate")
            with torch.no_grad():
                for p, lo, hi in bounds:
                    p.clamp_(lo, hi)

            with job_lock:
                job["losses"].append(loss.item())
                job["step"] = step
            if step % REPORT_EVERY == 0 or step == steps:
                snap = snapshot()
                with job_lock:
                    job["snapshot"] = snap
    except Exception as err:
        with job_lock:
            job["error"] = str(err)
    finally:
        snap = snapshot()
        with job_lock:
            job["snapshot"] = snap
            job["running"] = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(WEB, "curve_fitter.html")


@app.route("/<path:name>")
def static_file(name):
    return send_from_directory(WEB, name)


@app.get("/api/optimizers")
def optimizers():
    """[{name, settings: {beta1: 0.9, ...}}, ...] in display order."""
    return jsonify([{"name": name, "settings": defaults} for name, (_, defaults) in OPTIMIZERS.items()])


@app.post("/api/evaluate")
def evaluate():
    """Build the model with the given parameter values and return its curve."""
    data = request.get_json()
    points = data.get("points") or []
    modules, info = build_model(data.get("layers", []))
    result = {"layers": info, "ok": False}
    if modules is None:
        return jsonify(result)

    export_params(modules, info)
    width = info[-1]["out"] if info else 1
    if width != 2:
        result["error"] = f"the last layer must output [N, 2], it outputs [N, {width}]"
        return jsonify(result)

    curve, pred = predictions(modules, len(points))
    result.update(ok=True, curve=curve.tolist(), pred=pred.tolist())
    if len(points) > 1:
        loss_fn = LOSSES.get(data.get("loss"), LOSSES["MSE"])
        result["loss"] = loss_fn(pred, torch.tensor(points)).item()
    return jsonify(result)


@app.post("/api/optimize")
def start_optimize():
    data = request.get_json()
    points = data.get("points") or []
    layers = data.get("layers", [])
    if len(points) < 2:
        return jsonify({"error": "draw a curve first"}), 400

    modules, info = build_model(layers)
    if modules is None or not info or info[-1]["out"] != 2:
        return jsonify({"error": "fix the model first: it must end with [N, 2]"}), 400

    optimizer_name = data.get("optimizer", "Adam")
    if optimizer_name not in OPTIMIZERS:
        return jsonify({"error": f"unknown optimizer '{optimizer_name}'"}), 400
    loss_name = data.get("loss", "MSE")
    if loss_name not in LOSSES:
        return jsonify({"error": f"unknown loss '{loss_name}'"}), 400
    settings = data.get("settings") or {}
    lr = float(data.get("lr", 0.01))
    try:
        # torch checks the values (e.g. 0 <= beta < 1) when the optimizer is made
        make_optimizer(optimizer_name, [torch.zeros(1, requires_grad=True)], lr, settings)
    except (ValueError, TypeError) as err:
        return jsonify({"error": str(err)}), 400

    with job_lock:
        if job["running"]:
            return jsonify({"error": "already running"}), 409
        job.update(running=True, stop=False, losses=[], snapshot=None, error=None, step=0)

    threading.Thread(
        target=optimize, daemon=True,
        args=(modules, info, layers, points,
              int(data.get("steps", 2000)), lr, bool(data.get("clamp", True)),
              optimizer_name, settings, loss_name),
    ).start()
    return jsonify({"ok": True})


@app.post("/api/stop")
def stop_optimize():
    job["stop"] = True
    return jsonify({"ok": True})


@app.get("/api/status")
def status():
    """Progress of the running job. ?since=k returns only the losses after the first k."""
    since = int(request.args.get("since", 0))
    with job_lock:
        return jsonify({
            "running": job["running"],
            "step": job["step"],
            "losses": job["losses"][since:],
            "total": len(job["losses"]),
            "snapshot": job["snapshot"],
            "error": job["error"],
        })


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
