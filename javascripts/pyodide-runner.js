(function () {
  "use strict";

  const PYODIDE_CDN = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";

  let pyodideInstance = null;
  let pyodideLoading = false;
  let pyodideReady = false;
  const pendingCallbacks = [];

  async function ensurePyodide(statusCb) {
    if (pyodideReady) return pyodideInstance;

    if (pyodideLoading) {
      return new Promise((resolve) => pendingCallbacks.push(resolve));
    }

    pyodideLoading = true;
    statusCb("Loading Python runtime…");

    if (typeof loadPyodide === "undefined") {
      await new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = PYODIDE_CDN + "pyodide.js";
        s.onload = resolve;
        s.onerror = () => reject(new Error("Failed to load Pyodide"));
        document.head.appendChild(s);
      });
    }

    statusCb("Initialising Python…");
    pyodideInstance = await loadPyodide({ indexURL: PYODIDE_CDN });

    statusCb("Installing packages…");
    await pyodideInstance.loadPackage(["numpy", "matplotlib", "micropip"]);

    pyodideInstance.runPython(`
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

import io, base64
_original_show = plt.show
def _inline_show(*a, **kw):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#1e1e2e", edgecolor="none")
    buf.seek(0)
    _b64 = base64.b64encode(buf.read()).decode()
    import __main__
    if not hasattr(__main__, "_plot_outputs"):
        __main__._plot_outputs = []
    __main__._plot_outputs.append(_b64)
    plt.close("all")
plt.show = _inline_show
`);

    pyodideInstance.runPython(`
import sys, types
import numpy as _np

_jax = types.ModuleType("jax")
_jax.__package__ = "jax"
_jax.__path__ = []

def _jit(fn=None, **kw):
    if fn is None:
        return lambda f: f
    return fn
_jax.jit = _jit

def _grad(fn, argnums=0):
    eps = 1e-5
    def grad_fn(*args, **kwargs):
        idxs = [argnums] if isinstance(argnums, int) else list(argnums)
        grads = []
        for idx in idxs:
            x = _np.asarray(args[idx], dtype=float)
            g = _np.zeros_like(x)
            it = _np.nditer(x, flags=["multi_index"])
            while not it.finished:
                mi = it.multi_index
                old = float(x[mi])
                x[mi] = old + eps
                a_plus = list(args); a_plus[idx] = x.copy()
                fp = float(fn(*a_plus, **kwargs))
                x[mi] = old - eps
                a_minus = list(args); a_minus[idx] = x.copy()
                fm = float(fn(*a_minus, **kwargs))
                g[mi] = (fp - fm) / (2 * eps)
                x[mi] = old
                it.iternext()
            grads.append(g)
        return grads[0] if isinstance(argnums, int) else tuple(grads)
    return grad_fn
_jax.grad = _grad

def _value_and_grad(fn, argnums=0):
    g_fn = _grad(fn, argnums)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs), g_fn(*args, **kwargs)
    return wrapper
_jax.value_and_grad = _value_and_grad

def _vmap(fn, in_axes=0, out_axes=0):
    def vmapped(*args):
        axes = [in_axes] * len(args) if isinstance(in_axes, int) else list(in_axes)
        batch_size = None
        for a, ax in zip(args, axes):
            if ax is not None:
                batch_size = _np.asarray(a).shape[ax]
                break
        results = []
        for i in range(batch_size):
            sliced = []
            for a, ax in zip(args, axes):
                a = _np.asarray(a)
                sliced.append(a if ax is None else _np.take(a, i, axis=ax))
            results.append(fn(*sliced))
        return _np.stack(results, axis=out_axes)
    return vmapped
_jax.vmap = _vmap

def _pmap(fn, **kw):
    return _vmap(fn)
_jax.pmap = _pmap

_jnp = types.ModuleType("jax.numpy")
_jnp.__package__ = "jax"
for _attr in dir(_np):
    if not _attr.startswith("__"):
        setattr(_jnp, _attr, getattr(_np, _attr))
_jax.numpy = _jnp

_jrandom = types.ModuleType("jax.random")
_jrandom.__package__ = "jax"

def _PRNGKey(seed):
    return _np.array([0, seed], dtype=_np.uint32)
_jrandom.PRNGKey = _PRNGKey

def _split(key, num=2):
    base = int(key[1])
    return _np.array([[0, base + i + 1] for i in range(num)], dtype=_np.uint32)
_jrandom.split = _split

def _normal(key, shape=(), dtype=_np.float32):
    seed = int(key[1]) if hasattr(key, '__len__') else int(key)
    return _np.random.RandomState(seed).randn(*shape).astype(dtype)
_jrandom.normal = _normal

def _uniform(key, shape=(), dtype=_np.float32, minval=0.0, maxval=1.0):
    seed = int(key[1]) if hasattr(key, '__len__') else int(key)
    return (_np.random.RandomState(seed).rand(*shape) * (maxval - minval) + minval).astype(dtype)
_jrandom.uniform = _uniform

def _randint(key, shape, minval, maxval, dtype=_np.int32):
    seed = int(key[1]) if hasattr(key, '__len__') else int(key)
    return _np.random.RandomState(seed).randint(minval, maxval, size=shape).astype(dtype)
_jrandom.randint = _randint

_jax.random = _jrandom

_jnn = types.ModuleType("jax.nn")
_jnn.__package__ = "jax"
_jnn.relu = lambda x: _np.maximum(0, x)
_jnn.sigmoid = lambda x: 1.0 / (1.0 + _np.exp(-_np.clip(x, -500, 500)))
_jnn.softmax = lambda x, axis=-1: (lambda e: e / e.sum(axis=axis, keepdims=True))(_np.exp(x - x.max(axis=axis, keepdims=True)))
_jnn.tanh = _np.tanh
_jnn.log_softmax = lambda x, axis=-1: x - _np.log(_np.exp(x - x.max(axis=axis, keepdims=True)).sum(axis=axis, keepdims=True)) - x.max(axis=axis, keepdims=True)
_jax.nn = _jnn

_jlax = types.ModuleType("jax.lax")
_jlax.__package__ = "jax"
_jlax.dot = lambda a, b: _np.dot(a, b)
_jax.lax = _jlax

sys.modules["jax"] = _jax
sys.modules["jax.numpy"] = _jnp
sys.modules["jax.random"] = _jrandom
sys.modules["jax.nn"] = _jnn
sys.modules["jax.lax"] = _jlax
`);

    pyodideReady = true;
    pyodideLoading = false;
    pendingCallbacks.forEach((cb) => cb(pyodideInstance));
    pendingCallbacks.length = 0;
    return pyodideInstance;
  }

  async function runCode(code, outputEl, statusCb) {
    const pyodide = await ensurePyodide(statusCb);
    statusCb("Running…");

    outputEl.innerHTML = "";

    pyodide.runPython(`
import __main__
__main__._plot_outputs = []
`);

    pyodide.setStdout({ batched: (t) => appendText(outputEl, t, "stdout") });
    pyodide.setStderr({ batched: (t) => appendText(outputEl, t, "stderr") });

    try {
      const needed = detectPackages(code);
      if (needed.length > 0) {
        statusCb("Installing packages…");
        for (const pkg of needed) {
          try {
            await pyodide.loadPackage(pkg);
          } catch {
            appendText(
              outputEl,
              `⚠ Package "${pkg}" is not available in Pyodide.\n`,
              "stderr"
            );
          }
        }
        statusCb("Running…");
      }

      await pyodide.runPythonAsync(code);

      const plots = pyodide.runPython(`
import __main__
list(__main__._plot_outputs)
`);
      const plotList = plots.toJs();
      for (const b64 of plotList) {
        const img = document.createElement("img");
        img.src = "data:image/png;base64," + b64;
        img.className = "pyodide-plot";
        img.alt = "matplotlib output";
        outputEl.appendChild(img);
      }

      if (outputEl.children.length === 0 && outputEl.textContent === "") {
        appendText(outputEl, "✓ Code ran successfully (no output)", "success");
      }
    } catch (err) {
      appendText(outputEl, err.message, "stderr");
    }

    statusCb("");
  }

  function appendText(el, text, cls) {
    const span = document.createElement("span");
    span.className = "pyodide-" + cls;
    span.textContent = text + "\n";
    el.appendChild(span);
  }

  function detectPackages(code) {
    const mapping = {
      scipy: "scipy",
      sklearn: "scikit-learn",
      "scikit-learn": "scikit-learn",
      sympy: "sympy",
      pandas: "pandas",
      networkx: "networkx",
      PIL: "Pillow",
      cv2: "opencv-python",
    };
    const needed = new Set();
    for (const [token, pkg] of Object.entries(mapping)) {
      const re = new RegExp(
        `(?:^|\\s)(?:import|from)\\s+${token.replace("-", "\\-")}`,
        "m"
      );
      if (re.test(code)) needed.add(pkg);
    }
    return [...needed];
  }

  function injectButtons() {
    const wrappers = document.querySelectorAll(
      'div.language-python.highlight, div.highlight.language-python'
    );

    wrappers.forEach((wrapper) => {
      const pre = wrapper.querySelector("pre");
      const codeEl = pre && pre.querySelector("code");
      if (!pre || !codeEl) return;
      if (wrapper.dataset.pyodideInjected) return;
      wrapper.dataset.pyodideInjected = "true";
      injectRunUI(wrapper, pre, codeEl);
    });

    const standardBlocks = document.querySelectorAll(
      'pre > code.language-python'
    );
    standardBlocks.forEach((codeEl) => {
      const pre = codeEl.parentElement;
      if (pre.dataset.pyodideInjected) return;
      pre.dataset.pyodideInjected = "true";

      const wrapper = document.createElement("div");
      wrapper.className = "pyodide-wrapper";
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);
      injectRunUI(wrapper, pre, codeEl);
    });
  }

  function injectRunUI(wrapper, pre, codeEl) {
    wrapper.classList.add("pyodide-wrapper");

    const toolbar = document.createElement("div");
    toolbar.className = "pyodide-toolbar";

    const runBtn = document.createElement("button");
    runBtn.className = "pyodide-run-btn";
    runBtn.innerHTML = "▶ <span>Run</span>";
    runBtn.title = "Execute in browser (Pyodide)";

    const status = document.createElement("span");
    status.className = "pyodide-status";

    toolbar.appendChild(runBtn);
    toolbar.appendChild(status);
    wrapper.appendChild(toolbar);

    const output = document.createElement("pre");
    output.className = "pyodide-output";
    output.hidden = true;
    wrapper.appendChild(output);

    runBtn.addEventListener("click", async () => {
      runBtn.disabled = true;
      output.hidden = false;
      try {
        await runCode(codeEl.textContent, output, (msg) => {
          status.textContent = msg;
        });
      } finally {
        runBtn.disabled = false;
      }
    });
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(() => injectButtons());
  } else {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", injectButtons);
    } else {
      injectButtons();
    }
  }
})();
