"""Method wrappers and cross-thread context propagation helpers for telemetry."""

from __future__ import annotations

import contextlib
import functools
import inspect
import types
from typing import Any, Callable, Dict, Optional

from gcsfs.telemetry import manager
from gcsfs.telemetry.context import (
    ALL_DIMENSIONS,
    Dimension,
    canonicalize_method_params,
    get_telemetry_context,
    has_telemetry_context,
    reset_telemetry_context,
    set_telemetry_context,
)

_UNSET = object()

_TRACKED_METHODS = frozenset(
    {
        "cat",
        "cat_file",
        "cat_ranges",
        "copy",
        "cp",
        "cp_file",
        "created",
        "disk_usage",
        "du",
        "exists",
        "expand_path",
        "find",
        "get",
        "get_file",
        "glob",
        "head",
        "info",
        "isdir",
        "isfile",
        "ls",
        "makedirs",
        "merge",
        "mkdir",
        "mkdirs",
        "modified",
        "move",
        "mv",
        "mv_file",
        "open",
        "open_async",
        "pipe",
        "pipe_file",
        "put",
        "put_file",
        "read_text",
        "rm",
        "rm_file",
        "rmdir",
        "sign",
        "size",
        "sizes",
        "tail",
        "touch",
        "tree",
        "walk",
        "write_text",
    }
)

_SYNC_ALIAS_METHODS: Dict[str, str] = {
    "listdir": "ls",
}

_UNTRACKED_COMPOSITE_METHODS = frozenset(
    {
        "checksum",
        "read_block",
        "ukey",
    }
)

_METHOD_TRACKED_PARAMS: Dict[str, frozenset[str]] = {
    "cat": frozenset({"recursive"}),
    "copy": frozenset({"recursive"}),
    "cp": frozenset({"recursive"}),
    "disk_usage": frozenset({"prefix", "withdirs"}),
    "du": frozenset({"prefix", "withdirs"}),
    "expand_path": frozenset({"recursive"}),
    "find": frozenset({"detail", "prefix", "withdirs"}),
    "get": frozenset({"recursive"}),
    "glob": frozenset({"detail", "withdirs"}),
    "ls": frozenset({"detail", "prefix"}),
    "move": frozenset({"recursive"}),
    "mv": frozenset({"recursive"}),
    "open": frozenset({"cache_options"}),
    "open_async": frozenset({"cache_options"}),
    "put": frozenset({"recursive"}),
    "read_text": frozenset({"cache_options"}),
    "rm": frozenset({"recursive"}),
    "walk": frozenset({"detail", "prefix"}),
    "write_text": frozenset({"cache_options"}),
}


def _tracked_method_name(func: Callable) -> Optional[str]:
    """Return the tracked public method name for `func`, or None if untracked."""
    raw_name = getattr(func, "__name__", "")
    if raw_name in _SYNC_ALIAS_METHODS:
        return _SYNC_ALIAS_METHODS[raw_name]
    name = raw_name.removeprefix("_")
    return name if name in _TRACKED_METHODS else None


@functools.lru_cache(maxsize=256)
def _cached_positional_spec(
    raw_fn: Callable, method_name: str
) -> tuple[tuple[int, str], ...]:
    """Cache `(arg_index, param_name)` pairs on unbound functions so instances are never retained."""
    allowed = _METHOD_TRACKED_PARAMS.get(method_name)
    if not allowed:
        return ()
    try:
        params = list(inspect.signature(raw_fn).parameters.values())
    except (TypeError, ValueError):
        return ()
    return tuple(
        (idx, p.name)
        for idx, p in enumerate(params)
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and p.name in allowed
    )


def _positional_tracked_params(
    func: Callable, method_name: Optional[str]
) -> tuple[tuple[int, str], ...]:
    """Return `(arg_index, param_name)` pairs for tracked positional parameters on `func`."""
    if not method_name or method_name not in _METHOD_TRACKED_PARAMS:
        return ()
    unwrapped = inspect.unwrap(func, stop=lambda f: isinstance(f, types.MethodType))
    is_bound = isinstance(unwrapped, types.MethodType)
    raw_fn = inspect.unwrap(unwrapped.__func__) if is_bound else unwrapped
    try:
        spec = _cached_positional_spec(raw_fn, method_name)
    except TypeError:
        spec = _cached_positional_spec.__wrapped__(raw_fn, method_name)
    if is_bound:
        return tuple((idx - 1, name) for idx, name in spec if idx >= 1)
    return spec


def _is_truthy_param(val: Any) -> bool:
    try:
        return bool(val)
    except Exception:
        return False


def _extract_method_params(
    pos_spec: tuple[tuple[int, str], ...],
    args: tuple,
    kwargs: dict,
    method_name: Optional[str],
) -> str:
    """Extract sorted `.`-joined tracked parameter names enabled (truthy) on the call."""
    allowed = _METHOD_TRACKED_PARAMS.get(method_name or "")
    if not allowed:
        return ""
    found = set()
    n_args = len(args)
    for idx, name in pos_spec:
        if idx < n_args and _is_truthy_param(args[idx]):
            found.add(name)
    if kwargs:
        for name in allowed:
            if name in kwargs and _is_truthy_param(kwargs[name]):
                found.add(name)
    return ".".join(sorted(found)) if found else ""


def _collect_call_tokens(
    func: Callable,
    call_args: tuple,
    kwargs: dict,
    method_name: Optional[str],
    pos_spec: Optional[tuple[tuple[int, str], ...]] = None,
) -> Dict[str, str]:
    """Collect outermost call telemetry, skipping param evaluation when disabled."""
    if not manager.default_usage_tracker.is_enabled():
        return {}
    if method_name and method_name in _METHOD_TRACKED_PARAMS:
        spec = (
            pos_spec
            if pos_spec is not None
            else _positional_tracked_params(func, method_name)
        )
        params_str = _extract_method_params(spec, call_args, kwargs, method_name)
    else:
        params_str = ""
    return manager.default_usage_tracker.collect_tokens_map(
        method=method_name or "", method_params=params_str
    )


def _resolve_super_method(
    func: Callable, super_cls: Optional[type], args: tuple
) -> tuple[Callable, tuple]:
    """Resolve `func` via `super(super_cls, self)` when wrapped on a subclass."""
    if super_cls is not None and args and isinstance(args[0], super_cls):
        return getattr(super(super_cls, args[0]), func.__name__), args[1:]
    return func, args


@contextlib.contextmanager
def _telemetry_context(tokens_map: Optional[Dict[str, str]]):
    """Scope `tokens_map` as the active telemetry context for the current execution."""
    if tokens_map is None:
        yield
        return
    effective = tokens_map if tokens_map else {Dimension.METHOD.value: ""}
    token = set_telemetry_context(effective)
    try:
        yield
    finally:
        reset_telemetry_context(token)


def _disabled_tokens_map() -> Dict[str, str]:
    """Return negative-cache sentinels for all dimensions and registered detectors."""
    out = {d.value: "" for d in ALL_DIMENSIONS if d is not Dimension.METHOD_PARAMS}
    for det in manager.default_usage_tracker.detectors:
        key = det.name.value if isinstance(det.name, Dimension) else str(det.name)
        out[key] = ""
    return out


def _is_valid_reopen_tuple(reopen_tel: Any) -> bool:
    return (
        isinstance(reopen_tel, tuple)
        and len(reopen_tel) >= 2
        and isinstance(reopen_tel[0], str)
        and isinstance(reopen_tel[1], str)
    )


def _resolve_reopen_tokens(reopen_tel: Any) -> Dict[str, str]:
    """Build the telemetry token map for a pickled file handle being reopened."""
    if not manager.default_usage_tracker.is_enabled() or reopen_tel is None:
        return _disabled_tokens_map()
    if not _is_valid_reopen_tuple(reopen_tel):
        return manager.default_usage_tracker.collect_tokens_map(method="open")
    fw, method = reopen_tel[0], reopen_tel[1]
    params = canonicalize_method_params(reopen_tel[2]) if len(reopen_tel) > 2 else ""
    seed = {Dimension.FRAMEWORK.value: fw, Dimension.METHOD.value: method}
    if params:
        seed[Dimension.METHOD_PARAMS.value] = params
    with _telemetry_context(seed):
        return manager.default_usage_tracker.collect_tokens_map()


def with_telemetry_context(func: Callable) -> Callable:
    """
    Decorator for plain sync entry points that `mirror_gcs_methods` doesn't wrap (e.g. `_open`).

    Detects the caller framework once on the caller thread and scopes it for the whole call, so
    any requests issued underneath (including via `asyn.sync` to the loop thread) carry it.
    """
    method_name = _tracked_method_name(func)
    pos_spec = _positional_tracked_params(func, method_name)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if has_telemetry_context(Dimension.METHOD):
            return func(*args, **kwargs)
        tokens_map = _collect_call_tokens(func, args, kwargs, method_name, pos_spec)
        with _telemetry_context(tokens_map):
            return func(*args, **kwargs)

    return wrapper


def _run_sync(
    loop: Any,
    func: Callable,
    *args,
    tokens_map: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Any:
    """Set telemetry context on the caller thread and execute `func` via `fsspec.asyn.sync`."""
    import fsspec.asyn

    if tokens_map is None:
        if has_telemetry_context(Dimension.METHOD):
            return fsspec.asyn.sync(loop, func, *args, **kwargs)
        resolved_method = _tracked_method_name(func)
        tokens_map = _collect_call_tokens(func, args, kwargs, resolved_method)
    with _telemetry_context(tokens_map):
        return fsspec.asyn.sync(loop, func, *args, **kwargs)


def _gcs_async_wrapper(
    func: Callable,
    super_cls: Optional[type] = None,
    method_name: Any = _UNSET,
) -> Callable:
    """
    GCS-specific async wrapper for coroutines and async generators that scopes telemetry
    context during native async execution, avoiding repetitive stack-walking across
    internal coroutines, chunk fetches, or generator steps.
    """
    if getattr(func, "_is_gcs_async_wrapped", False):
        return func

    if method_name is _UNSET:
        method_name = _tracked_method_name(func)

    if inspect.isasyncgenfunction(func):

        @functools.wraps(func)
        async def gen_wrapper(*args, **kwargs):
            target, pass_args = _resolve_super_method(func, super_cls, args)
            # Snapshot once per generator creation; scope per step so it never leaks to the consumer
            tokens_map = (
                get_telemetry_context()
                if has_telemetry_context(Dimension.METHOD)
                else _collect_call_tokens(target, pass_args, kwargs, method_name)
            )
            gen = target(*pass_args, **kwargs)
            try:
                while True:
                    with _telemetry_context(tokens_map):
                        try:
                            item = await gen.__anext__()
                        except StopAsyncIteration:
                            return
                    yield item
            finally:
                with _telemetry_context(tokens_map):
                    await gen.aclose()

        gen_wrapper._is_gcs_async_wrapped = True
        return gen_wrapper

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        target, pass_args = _resolve_super_method(func, super_cls, args)
        # O(1) fast path for nested calls (already inside _run_sync or another wrapped call)
        if has_telemetry_context(Dimension.METHOD):
            return await target(*pass_args, **kwargs)
        tokens_map = _collect_call_tokens(target, pass_args, kwargs, method_name)
        with _telemetry_context(tokens_map):
            return await target(*pass_args, **kwargs)

    wrapper._is_gcs_async_wrapped = True
    return wrapper


def _resolve_target_method(func: Callable, args: tuple) -> tuple[Any, Callable, tuple]:
    """Dynamically resolve `func` on `self` to preserve subclass overrides and mocks."""
    self = args[0]
    method_name = getattr(func, "__name__", None)
    actual_func = getattr(self, method_name, None) if method_name else None
    if actual_func is not None:
        return self, actual_func, args[1:]
    return self, func, args


def _gcs_sync_wrapper(
    func: Callable,
    super_cls: Optional[type] = None,
    method_name: Any = _UNSET,
) -> Callable:
    """
    GCS-specific sync wrapper for coroutines, async generators, and synchronous methods
    that captures caller thread telemetry and bridges it into execution.
    """
    if getattr(func, "_is_gcs_sync_wrapped", False):
        return func

    if method_name is _UNSET:
        method_name = _tracked_method_name(func)

    if inspect.isasyncgenfunction(func):

        @functools.wraps(func)
        def gen_wrapper(*args, **kwargs):
            self, actual_func, pass_args = _resolve_target_method(func, args)
            # Snapshot once on the caller thread and reuse it for every yielded step
            tokens_map = (
                get_telemetry_context()
                if has_telemetry_context(Dimension.METHOD)
                else _collect_call_tokens(actual_func, pass_args, kwargs, method_name)
            )
            gen = actual_func(*pass_args, **kwargs)
            try:
                while True:
                    try:
                        yield _run_sync(self.loop, gen.__anext__, tokens_map=tokens_map)
                    except StopAsyncIteration:
                        break
            finally:
                if hasattr(gen, "aclose"):
                    with contextlib.suppress(Exception):
                        _run_sync(self.loop, gen.aclose, tokens_map=tokens_map)

        gen_wrapper._is_gcs_sync_wrapped = True
        return gen_wrapper

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self, actual_func, pass_args = _resolve_target_method(func, args)
            tokens_map = (
                None
                if has_telemetry_context(Dimension.METHOD)
                else _collect_call_tokens(actual_func, pass_args, kwargs, method_name)
            )
            return _run_sync(
                self.loop,
                actual_func,
                *pass_args,
                tokens_map=tokens_map,
                **kwargs,
            )

        wrapper._is_gcs_sync_wrapped = True
        return wrapper

    @functools.wraps(func)
    def sync_fn_wrapper(*args, **kwargs):
        target, pass_args = _resolve_super_method(func, super_cls, args)
        reopen_tel = kwargs.get("_reopen_telemetry", _UNSET)
        if reopen_tel is not _UNSET and (
            reopen_tel is None or _is_valid_reopen_tuple(reopen_tel)
        ):
            with _telemetry_context(_resolve_reopen_tokens(reopen_tel)):
                return target(*pass_args, **kwargs)
        if has_telemetry_context(Dimension.METHOD):
            return target(*pass_args, **kwargs)
        tokens_map = _collect_call_tokens(target, pass_args, kwargs, method_name)
        with _telemetry_context(tokens_map):
            return target(*pass_args, **kwargs)

    sync_fn_wrapper._is_gcs_sync_wrapped = True
    return sync_fn_wrapper


def mirror_gcs_methods(obj: type) -> None:
    """
    Binds sync and native async methods on a GCSFileSystem class using
    telemetry-aware wrappers.
    Uses class __mro__ to find all async coroutines and async generators,
    avoiding eager property evaluation.

    Note: methods inherited from fsspec (e.g. `AsyncFileSystem._cat`) are wrapped and stored
    on `obj` itself, since fsspec's shared base classes must not be modified. A subclass that
    lists another `AsyncFileSystem` subclass *after* `obj` (e.g. `class MyFS(GCSFileSystem,
    Mixin)`) will therefore resolve those methods to `obj`'s wrapper instead of the mixin's
    override. Mixins listed before `obj` (as in `ExtendedGcsFileSystem`) are unaffected.
    """
    from fsspec.asyn import AsyncFileSystem, async_methods

    # Collect coroutine and async generator method names strictly from class dictionaries in MRO
    method_names = set(async_methods)

    for cls in obj.__mro__:
        if cls is object:
            continue
        for name, attr in cls.__dict__.items():
            # Pick non-dunder private methods (e.g. '_ls') or tracked public async methods (e.g. 'open_async')
            if (
                name.startswith("_")
                and not (name.startswith("__") and name.endswith("__"))
            ) or name in _TRACKED_METHODS:
                if inspect.iscoroutinefunction(attr) or inspect.isasyncgenfunction(
                    attr
                ):
                    method_names.add(name)

    # Bind the wrappers for valid methods
    from fsspec.spec import AbstractFileSystem

    sync_method_names = set(getattr(obj, "_gcs_sync_method_names", ()))
    for method in method_names:
        is_private = method.startswith("_")
        smethod = method.removeprefix(
            "_"
        )  # e.g., "_ls" -> "ls", "_cat_file" -> "cat_file", "_walk" -> "walk"
        tracked_name = smethod if smethod in _TRACKED_METHODS else None

        is_public_sync = is_private and (
            hasattr(AbstractFileSystem, smethod)
            or method in async_methods
            or hasattr(obj, smethod)
        )
        # Only wrap public API or AsyncFileSystem methods (skip gcsfs-internal helpers like _request)
        if not (
            is_public_sync
            or hasattr(AsyncFileSystem, method)
            or method in _TRACKED_METHODS
        ):
            continue

        async_fn = getattr(obj, method, None)

        if inspect.iscoroutinefunction(async_fn) or inspect.isasyncgenfunction(
            async_fn
        ):
            # Wrap each native async method once. Skip ones a parent already wrapped, so subclasses
            # inherit the parent's wrapper instead of a copy that would hide patches on the parent.
            was_already_async_wrapped = getattr(
                async_fn, "_is_gcs_async_wrapped", False
            )
            if not was_already_async_wrapped:
                super_cls = None if method in obj.__dict__ else obj
                async_fn = _gcs_async_wrapper(
                    async_fn, super_cls=super_cls, method_name=tracked_name
                )
                setattr(obj, method, async_fn)
            if is_public_sync and (
                not was_already_async_wrapped
                or not getattr(
                    getattr(obj, smethod, None), "_is_gcs_sync_wrapped", False
                )
            ):
                local_sync = obj.__dict__.get(smethod)
                if (
                    local_sync is not None
                    and method not in obj.__dict__
                    and callable(local_sync)
                    and not inspect.iscoroutinefunction(local_sync)
                    and not inspect.isasyncgenfunction(local_sync)
                    and not getattr(local_sync, "_is_gcs_sync_wrapped", False)
                ):
                    continue
                mth = _gcs_sync_wrapper(async_fn, method_name=tracked_name)
                mth.__dict__.pop("_is_gcs_async_wrapped", None)
                if not getattr(mth, "__doc__", None):
                    mth.__doc__ = getattr(
                        getattr(AbstractFileSystem, smethod, None), "__doc__", ""
                    )
                setattr(obj, smethod, mth)
                sync_method_names.add(smethod)
    obj._gcs_sync_method_names = frozenset(sync_method_names)

    # Wrap synchronous public entry points, aliases (e.g. listdir -> ls), and composite helpers
    sync_targets: list[tuple[str, Optional[str]]] = [
        (name, name) for name in _TRACKED_METHODS
    ]
    sync_targets.extend(_SYNC_ALIAS_METHODS.items())
    sync_targets.extend((name, None) for name in _UNTRACKED_COMPOSITE_METHODS)

    for sync_name, target_method in sync_targets:
        raw_attr = inspect.getattr_static(obj, sync_name, None)
        if isinstance(raw_attr, (staticmethod, classmethod)):
            continue
        sync_fn = getattr(obj, sync_name, None)
        if (
            callable(sync_fn)
            and not inspect.iscoroutinefunction(sync_fn)
            and not inspect.isasyncgenfunction(sync_fn)
            and not getattr(sync_fn, "_is_gcs_sync_wrapped", False)
        ):
            super_cls = None if sync_name in obj.__dict__ else obj
            setattr(
                obj,
                sync_name,
                _gcs_sync_wrapper(
                    sync_fn, super_cls=super_cls, method_name=target_method
                ),
            )


def bind_gcs_sync_methods(fs: Any) -> None:
    """
    Bind the class-level telemetry sync wrappers set by `mirror_gcs_methods` onto `fs`.

    This matches `fsspec.asyn.mirror_sync_methods`, which fsspec runs for every async filesystem
    instance (and which bound GCSFileSystem's sync methods before they moved to the class). The
    bound methods stored in `fs.__dict__` reference `fs`, so the instance is only freed by the
    cyclic garbage collector rather than as soon as its last outside reference is dropped. This
    matters after `fork()`: fsspec clears `cls._cache` in the child (e.g. in
    `subprocess.Popen(..., preexec_fn=...)`, used by `ray.init()`), and an instance only the cache
    referenced must not be torn down there, because deallocating its inherited `grpc.aio` channel
    in the child hangs or crashes.

    This only protects children that exec right away (CPython disables the cyclic GC in the child
    around `preexec_fn`). A forked child that keeps running Python, such as a fork-mode
    `multiprocessing` or PyTorch `DataLoader` worker, can still collect an inherited instance once
    ordinary allocations trigger the cyclic GC, and hang the same way. That was also the case
    before the wrappers moved to the class; use the "spawn" or "forkserver" start method there.

    Methods a subclass overrides with a plain sync function are left alone.
    """
    cls = type(fs)
    for name in getattr(cls, "_gcs_sync_method_names", ()):
        method = getattr(cls, name, None)
        # Require a real function carrying the marker: mocks (e.g. `mock.patch.object(cls, "ls")`)
        # auto-create any attribute, and binding one would pass `fs` as an extra positional arg.
        if (
            inspect.isfunction(method)
            and getattr(method, "_is_gcs_sync_wrapped", False) is True
        ):
            setattr(fs, name, types.MethodType(method, fs))


def _setup_file_telemetry(file_obj) -> Optional[Any]:
    if not manager.default_usage_tracker.is_enabled():
        return None

    current = get_telemetry_context()
    if getattr(file_obj, "_telemetry_disabled", False):
        disabled_map = _disabled_tokens_map()
        if (
            all(current.get(k) == "" for k in disabled_map)
            and Dimension.METHOD_PARAMS.value not in current
        ):
            return None
        current.update(disabled_map)
        current.pop(Dimension.METHOD_PARAMS.value, None)
        return set_telemetry_context(current)

    fw = getattr(file_obj, "caller_framework", None)
    has_fw = isinstance(fw, str)
    has_method = hasattr(file_obj, "caller_method")
    has_params = hasattr(file_obj, "caller_params")
    if not has_fw and not has_method and not has_params:
        return None

    if has_method or has_params:
        method = getattr(file_obj, "caller_method", "open")
        params = canonicalize_method_params(getattr(file_obj, "caller_params", ""))
    elif Dimension.METHOD.value in current:
        method = current[Dimension.METHOD.value]
        params = canonicalize_method_params(
            current.get(Dimension.METHOD_PARAMS.value, "")
        )
    else:
        method = "open"
        params = ""

    # Fast path: telemetry already active (e.g. nested file method calls), skip a redundant set/reset
    if (
        (not has_fw or current.get(Dimension.FRAMEWORK.value) == fw)
        and Dimension.METHOD.value in current
        and current.get(Dimension.METHOD.value) == method
        and current.get(Dimension.METHOD_PARAMS.value, "") == params
    ):
        return None

    if has_fw:
        current[Dimension.FRAMEWORK.value] = fw
    current[Dimension.METHOD.value] = method
    if params:
        current[Dimension.METHOD_PARAMS.value] = params
    else:
        current.pop(Dimension.METHOD_PARAMS.value, None)
    token = set_telemetry_context(current)

    missing_detector = any(
        (det.name.value if isinstance(det.name, Dimension) else str(det.name))
        not in current
        for det in manager.default_usage_tracker.detectors
    )
    if missing_detector:
        try:
            filled = manager.default_usage_tracker.collect_tokens_map()
            if filled != current:
                set_telemetry_context(filled)
        except BaseException:
            reset_telemetry_context(token)
            raise
    return token


@contextlib.contextmanager
def _file_telemetry_context(file_obj):
    """Context manager to safely handle file telemetry token setup and teardown."""
    token = _setup_file_telemetry(file_obj)
    try:
        yield
    finally:
        if token is not None:
            reset_telemetry_context(token)


def _file_telemetry_wrapper(func: Callable) -> Callable:
    """Wraps a File sync or async method to propagate caller telemetry."""
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            with _file_telemetry_context(self):
                return await func(self, *args, **kwargs)

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        with _file_telemetry_context(self):
            return func(self, *args, **kwargs)

    return sync_wrapper


def wrap_file_methods(cls: Any = None) -> None:
    """Wrap all callable methods and properties on a File class (`GCSFile` by default, or `ZonalFile`)."""
    from gcsfs.core import GCSFile

    if cls is None:
        cls = GCSFile
    is_gcs_file = isinstance(cls, type) and issubclass(cls, GCSFile)

    orig_init = cls.__dict__.get("__init__", cls.__init__)
    if not getattr(orig_init, "_is_telemetry_init_wrapped", False):
        init_pos_spec = _positional_tracked_params(orig_init, "open")

        @functools.wraps(orig_init)
        def init_wrapper(*args, **kwargs):
            self = args[0]
            reopen_tel = kwargs.pop("_reopen_telemetry", _UNSET)
            if (
                reopen_tel is not _UNSET
                and reopen_tel is not None
                and not _is_valid_reopen_tuple(reopen_tel)
            ):
                reopen_tel = _UNSET
            enabled = (
                manager.default_usage_tracker.is_enabled()
                and reopen_tel is not None
                and not getattr(self, "_telemetry_disabled", False)
            )
            if reopen_tel is not _UNSET and enabled:
                self.caller_framework = reopen_tel[0]
                self.caller_method = reopen_tel[1]
                self.caller_params = (
                    canonicalize_method_params(reopen_tel[2])
                    if len(reopen_tel) > 2
                    else ""
                )
            ctx = get_telemetry_context() if has_telemetry_context() else None
            if not enabled:
                self._telemetry_disabled = True
                self.caller_framework = ""
                self.caller_method = ""
                self.caller_params = ""
            else:
                has_ctx_method = ctx is not None and Dimension.METHOD.value in ctx
                if not hasattr(self, "caller_method"):
                    self.caller_method = (
                        ctx[Dimension.METHOD.value] if has_ctx_method else "open"
                    )
                if not hasattr(self, "caller_params"):
                    self.caller_params = (
                        canonicalize_method_params(
                            ctx.get(Dimension.METHOD_PARAMS.value, "")
                        )
                        if has_ctx_method
                        else (
                            _extract_method_params(init_pos_spec, args, kwargs, "open")
                            if self.caller_method
                            else ""
                        )
                    )

            if is_gcs_file and (
                not enabled
                or reopen_tel is not _UNSET
                or not has_telemetry_context(Dimension.METHOD)
            ):
                if not enabled:
                    tokens_map = _disabled_tokens_map()
                elif reopen_tel is not _UNSET:
                    tokens_map = _resolve_reopen_tokens(reopen_tel)
                else:
                    tokens_map = manager.default_usage_tracker.collect_tokens_map(
                        method=self.caller_method, method_params=self.caller_params
                    )
                with _telemetry_context(tokens_map):
                    orig_init(*args, **kwargs)
            else:
                orig_init(*args, **kwargs)

            if not enabled:
                self.caller_framework = ""
                self.caller_method = ""
                self.caller_params = ""
            elif (
                ctx is not None
                and not hasattr(self, "caller_framework")
                and Dimension.FRAMEWORK.value in ctx
            ):
                self.caller_framework = ctx[Dimension.FRAMEWORK.value]

        init_wrapper._is_telemetry_init_wrapped = True
        setattr(cls, "__init__", init_wrapper)

    for name, attr in list(cls.__dict__.items()):
        if name.startswith("__") or isinstance(attr, (classmethod, staticmethod)):
            continue
        is_prop = isinstance(attr, property)
        func = attr.fget if is_prop else (attr if callable(attr) else None)
        if func is not None and not getattr(func, "_is_telemetry_wrapped", False):
            wrapped = _file_telemetry_wrapper(func)
            wrapped._is_telemetry_wrapped = True
            setattr(cls, name, attr.getter(wrapped) if is_prop else wrapped)
