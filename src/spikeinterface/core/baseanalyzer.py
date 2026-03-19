"""
Base classes for domain-agnostic analyzers and their extensions.

* ``BaseAnalyzerExtension`` — persistence (binary_folder / zarr), parameter
  management, dependency tracking, and run lifecycle.
* ``BaseAnalyzer`` — extension management (compute / load / delete),
  with a generic *input_extractor* / *output_extractor* pattern so
  SpikeInterface (recording / sorting) and photon-mosaic (imaging / rois)
  share the same code.

Domain-specific subclasses add their own properties
(e.g. sorting_analyzer, sparsity, merge/split) on top.
"""

import inspect
import json
import pickle
import shutil
import warnings
import weakref
from copy import copy
from itertools import chain
from time import perf_counter

import numpy as np

from .core_tools import check_json, is_path_remote, retrieve_importing_provenance
from .zarrextractors import get_default_zarr_compressor


class BaseAnalyzerExtension:
    """
    Domain-agnostic base class for analyzer extensions.

    Handles persistence (binary_folder / zarr), parameter management,
    dependency tracking, and run lifecycle.

    Subclasses must set ``extension_name`` and implement the abstract methods:

      * ``_set_params(**params)`` — validate/clean params, return dict
      * ``_run(**kwargs)`` — populate ``self.data``
      * ``_select_extension_data(ids)`` — filter data to a subset
      * ``_get_data()`` — return the computed result

    Optional:

      * ``_get_pipeline_nodes()`` — if ``use_nodepipeline = True``
      * ``_handle_backward_compatibility_on_load()`` — if ``need_backward_compatibility_on_load = True``
    """

    extension_name = None
    depend_on = []
    use_nodepipeline = False
    nodepipeline_variables = None
    need_job_kwargs = False
    need_backward_compatibility_on_load = False
    need_input = False

    def __init__(self, analyzer):
        self._analyzer_ref = weakref.ref(analyzer)
        self.params = None
        self.run_info = self._default_run_info_dict()
        self.data = dict()

    def _default_run_info_dict(self):
        return dict(run_completed=False, runtime_s=None)

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    def _run(self, **kwargs):
        # must populate self.data
        raise NotImplementedError

    def _set_params(self, **params):
        # must return a cleaned params dict
        raise NotImplementedError

    def _select_extension_data(self, ids):
        raise NotImplementedError

    def _get_pipeline_nodes(self):
        raise NotImplementedError

    def _get_data(self):
        raise NotImplementedError

    def _handle_backward_compatibility_on_load(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def analyzer(self):
        """The parent analyzer (weakref-resolved)."""
        a = self._analyzer_ref()
        if a is None:
            raise ValueError(f"The extension {self.extension_name} has lost its analyzer " "(garbage collected)")
        return a

    @property
    def format(self):
        return self.analyzer.format

    @property
    def folder(self):
        return self.analyzer.folder

    # ------------------------------------------------------------------
    # Hook for CSV index coercion during load_data
    # ------------------------------------------------------------------

    def _get_entity_ids(self):
        """Return entity IDs for CSV index dtype coercion, or None to skip.

        Override in subclasses (e.g. return self.sorting_analyzer.unit_ids
        or self.roi_analyzer.roi_ids).
        """
        return None

    # ------------------------------------------------------------------
    # Folder / zarr helpers
    # ------------------------------------------------------------------

    def _get_binary_extension_folder(self):
        return self.folder / "extensions" / self.extension_name

    def _get_zarr_extension_group(self, mode="r+"):
        zarr_root = self.analyzer._get_zarr_root(mode=mode)
        return zarr_root["extensions"][self.extension_name]

    # ------------------------------------------------------------------
    # Dependency introspection
    # ------------------------------------------------------------------

    @classmethod
    def get_required_dependencies(cls, **params):
        """Return required parent extension names."""
        return cls.depend_on

    @classmethod
    def get_optional_dependencies(cls, **params):
        """Return optional parent extension names."""
        return []

    @classmethod
    def get_any_dependencies(cls, **params):
        """Return all parent extension names (required + optional), flattened."""
        required = cls.get_required_dependencies(**params)
        optional = cls.get_optional_dependencies(**params)
        all_deps = required + optional
        return list(chain.from_iterable(dep.split("|") for dep in all_deps))

    @classmethod
    def get_default_params(cls):
        """Get the default params for the extension from its ``_set_params`` signature."""
        sig = inspect.signature(cls._set_params)
        return {k: v.default for k, v in sig.parameters.items() if k != "self" and v.default != inspect.Parameter.empty}

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, analyzer):
        ext = cls(analyzer)
        ext.load_params()
        ext.load_run_info()
        if ext.run_info is not None:
            if ext.run_info["run_completed"]:
                ext.load_data()
                if cls.need_backward_compatibility_on_load:
                    ext._handle_backward_compatibility_on_load()
                if len(ext.data) > 0:
                    return ext
        else:
            # back-compatibility: old analyzers without run_info
            ext.load_data()
            if cls.need_backward_compatibility_on_load:
                ext._handle_backward_compatibility_on_load()
            if len(ext.data) > 0:
                return ext
        # run not completed or data missing → should be (re)computed
        return None

    def load_run_info(self):
        run_info = None
        if self.format == "binary_folder":
            run_info_file = self._get_binary_extension_folder() / "run_info.json"
            if run_info_file.is_file():
                with open(str(run_info_file), "r") as f:
                    run_info = json.load(f)
        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode="r")
            run_info = extension_group.attrs.get("run_info", None)

        if run_info is None:
            warnings.warn(f"Found no run_info file for {self.extension_name}, extension should be re-computed.")
        self.run_info = run_info

    def load_params(self):
        if self.format == "binary_folder":
            params_file = self._get_binary_extension_folder() / "params.json"
            assert params_file.is_file(), f"No params file in extension {self.extension_name} folder"
            with open(str(params_file), "r") as f:
                params = json.load(f)
        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode="r")
            assert "params" in extension_group.attrs, f"No params file in extension {self.extension_name} folder"
            params = extension_group.attrs["params"]

        self.params = params

    def load_data(self):
        ext_data = None
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            for ext_data_file in extension_folder.iterdir():
                if (
                    ext_data_file.name == "params.json"
                    or ext_data_file.name == "info.json"
                    or ext_data_file.name == "run_info.json"
                    or str(ext_data_file.name).startswith("._")  # ignore AppleDouble format files
                ):
                    continue
                ext_data_name = ext_data_file.stem
                if ext_data_file.suffix == ".json":
                    with ext_data_file.open("r") as f:
                        ext_data = json.load(f)
                elif ext_data_file.suffix == ".npy":
                    ext_data = np.load(ext_data_file)
                elif ext_data_file.suffix == ".csv":
                    import pandas as pd

                    ext_data = pd.read_csv(ext_data_file, index_col=0)
                    # coerce index dtype to match entity IDs if available
                    entity_ids = self._get_entity_ids()
                    if entity_ids is not None and ext_data.shape[0] == entity_ids.size:
                        if ext_data.index.dtype != entity_ids.dtype:
                            ext_data.index = ext_data.index.astype(entity_ids.dtype)
                elif ext_data_file.suffix == ".pkl":
                    with ext_data_file.open("rb") as f:
                        ext_data = pickle.load(f)
                else:
                    continue
                self.set_data(ext_data_name, ext_data)

        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode="r")
            for ext_data_name in extension_group.keys():
                ext_data_ = extension_group[ext_data_name]
                if "dict" in ext_data_.attrs:
                    ext_data = ext_data_[0]
                elif "dataframe" in ext_data_.attrs:
                    import pandas as pd

                    index = ext_data_["index"]
                    ext_data = pd.DataFrame(index=index)
                    for col in ext_data_.keys():
                        if col != "index":
                            ext_data.loc[:, col] = ext_data_[col][:]
                    ext_data = ext_data.convert_dtypes()
                elif "object" in ext_data_.attrs:
                    ext_data = ext_data_[0]
                else:
                    ext_data = np.array(ext_data_)
                self.set_data(ext_data_name, ext_data)

        if len(self.data) == 0:
            warnings.warn(f"Found no data for {self.extension_name}, extension should be re-computed.")

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self, new_analyzer, ids=None):
        """Copy this extension to a new analyzer, optionally filtering to *ids*."""
        new_extension = self.__class__(new_analyzer)
        new_extension.params = self.params.copy()
        if ids is None:
            new_extension.data = self.data
        else:
            new_extension.data = self._select_extension_data(ids)
        new_extension.run_info = copy(self.run_info)
        new_extension.save()
        return new_extension

    # ------------------------------------------------------------------
    # Run / save lifecycle
    # ------------------------------------------------------------------

    def run(self, save=True, **kwargs):
        if save and not self.analyzer.is_read_only():
            # NB: _save_params() also resets the folder or zarr group
            self._save_params()
            self._save_importing_provenance()

        t_start = perf_counter()
        self._run(**kwargs)
        t_end = perf_counter()
        self.run_info["runtime_s"] = t_end - t_start
        self.run_info["run_completed"] = True

        if save and not self.analyzer.is_read_only():
            self._save_run_info()
            self._save_data()
            if self.format == "zarr":
                import zarr

                zarr.consolidate_metadata(self.analyzer._get_zarr_root().store)

    def save(self):
        self._save_params()
        self._save_importing_provenance()
        self._save_run_info()
        self._save_data()

        if self.format == "zarr":
            import zarr

            zarr.consolidate_metadata(self.analyzer._get_zarr_root().store)

    def _save_data(self):
        if self.format == "memory":
            return

        if self.analyzer.is_read_only():
            raise ValueError(f"The analyzer is read-only; saving extension {self.extension_name} is not possible")

        try:
            import pandas as pd

            HAS_PANDAS = True
        except ImportError:
            HAS_PANDAS = False

        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            for ext_data_name, ext_data in self.data.items():
                if isinstance(ext_data, dict):
                    ext_data_ = check_json(ext_data)
                    with (extension_folder / f"{ext_data_name}.json").open("w") as f:
                        json.dump(ext_data_, f)
                elif isinstance(ext_data, np.ndarray):
                    data_file = extension_folder / f"{ext_data_name}.npy"
                    if isinstance(ext_data, np.memmap) and data_file.exists():
                        pass
                    else:
                        np.save(data_file, ext_data)
                elif HAS_PANDAS and isinstance(ext_data, pd.DataFrame):
                    ext_data.to_csv(extension_folder / f"{ext_data_name}.csv", index=True)
                else:
                    try:
                        with (extension_folder / f"{ext_data_name}.pkl").open("wb") as f:
                            pickle.dump(ext_data, f)
                    except Exception:
                        raise Exception(f"Could not save {ext_data_name} as extension data")

        elif self.format == "zarr":
            import numcodecs

            saving_options = self.analyzer._backend_options.get("saving_options", {})
            extension_group = self._get_zarr_extension_group(mode="r+")

            if "compressor" not in saving_options:
                saving_options["compressor"] = get_default_zarr_compressor()

            for ext_data_name, ext_data in self.data.items():
                if ext_data_name in extension_group:
                    del extension_group[ext_data_name]

                if isinstance(ext_data, (dict, list)):
                    ext_data_ = check_json(ext_data)
                    extension_group.create_dataset(
                        name=ext_data_name, data=np.array([ext_data_], dtype=object), object_codec=numcodecs.JSON()
                    )
                    extension_group[ext_data_name].attrs["dict"] = True
                elif isinstance(ext_data, np.ndarray):
                    extension_group.create_dataset(name=ext_data_name, data=ext_data, **saving_options)
                elif HAS_PANDAS and isinstance(ext_data, pd.DataFrame):
                    df_group = extension_group.create_group(ext_data_name)
                    indices = ext_data.index.to_numpy()
                    if indices.dtype.kind == "O":
                        indices = indices.astype(str)
                    df_group.create_dataset(name="index", data=indices)
                    for col in ext_data.columns:
                        col_data = ext_data[col].to_numpy()
                        if col_data.dtype.kind == "O":
                            col_data = col_data.astype(str)
                        df_group.create_dataset(name=col, data=col_data)
                    df_group.attrs["dataframe"] = True
                else:
                    try:
                        extension_group.create_dataset(
                            name=ext_data_name, data=np.array([ext_data], dtype=object), object_codec=numcodecs.Pickle()
                        )
                    except Exception:
                        raise Exception(f"Could not save {ext_data_name} as extension data")
                    extension_group[ext_data_name].attrs["object"] = True

    # ------------------------------------------------------------------
    # Folder management
    # ------------------------------------------------------------------

    def _reset_extension_folder(self):
        """Delete the extension folder/group and create an empty one."""
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            if extension_folder.is_dir():
                shutil.rmtree(extension_folder)
            extension_folder.mkdir(exist_ok=False, parents=True)
        elif self.format == "zarr":
            import zarr

            zarr_root = self.analyzer._get_zarr_root(mode="r+")
            _ = zarr_root["extensions"].create_group(self.extension_name, overwrite=True)
            zarr.consolidate_metadata(zarr_root.store)

    def _delete_extension_folder(self):
        """Delete the extension folder/group."""
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            if extension_folder.is_dir():
                shutil.rmtree(extension_folder)
        elif self.format == "zarr":
            import zarr

            zarr_root = self.analyzer._get_zarr_root(mode="r+")
            if self.extension_name in zarr_root["extensions"]:
                del zarr_root["extensions"][self.extension_name]
                zarr.consolidate_metadata(zarr_root.store)

    def delete(self):
        """Delete the extension from disk and clear in-memory state."""
        self._delete_extension_folder()
        self.params = None
        self.run_info = self._default_run_info_dict()
        self.data = dict()

    def reset(self):
        """Reset the extension: recreate empty folder, clear in-memory state."""
        self._reset_extension_folder()
        self.params = None
        self.run_info = self._default_run_info_dict()
        self.data = dict()

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------

    def set_params(self, save=True, **params):
        """Set parameters for the extension and optionally persist."""
        if save:
            self._reset_extension_folder()

        params = self._set_params(**params)
        self.params = params

        if self.analyzer.is_read_only():
            return

        if save:
            self._save_params()
            self._save_importing_provenance()

    def _save_params(self):
        params_to_save = self.params.copy()
        self._reset_extension_folder()

        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            extension_folder.mkdir(exist_ok=True, parents=True)
            param_file = extension_folder / "params.json"
            param_file.write_text(json.dumps(check_json(params_to_save), indent=4), encoding="utf8")
        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode="r+")
            extension_group.attrs["params"] = check_json(params_to_save)

    def _save_importing_provenance(self):
        info = retrieve_importing_provenance(self.__class__)
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            extension_folder.mkdir(exist_ok=True, parents=True)
            info_file = extension_folder / "info.json"
            info_file.write_text(json.dumps(info, indent=4), encoding="utf8")
        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode="r+")
            extension_group.attrs["info"] = info

    def _save_run_info(self):
        if self.run_info is not None:
            run_info = self.run_info.copy()

            if self.format == "binary_folder":
                extension_folder = self._get_binary_extension_folder()
                run_info_file = extension_folder / "run_info.json"
                run_info_file.write_text(json.dumps(run_info, indent=4), encoding="utf8")
            elif self.format == "zarr":
                extension_group = self._get_zarr_extension_group(mode="r+")
                extension_group.attrs["run_info"] = run_info

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_pipeline_nodes(self):
        assert self.use_nodepipeline, "get_pipeline_nodes() must be called only when use_nodepipeline=True"
        return self._get_pipeline_nodes()

    def get_data(self, *args, **kwargs):
        if self.run_info is not None:
            assert self.run_info[
                "run_completed"
            ], f"You must run the extension {self.extension_name} before retrieving data"
        assert len(self.data) > 0, "Extension has been run but no data found."
        return self._get_data(*args, **kwargs)

    def set_data(self, ext_data_name, ext_data):
        self.data[ext_data_name] = ext_data


# ---------------------------------------------------------------------------
# Shared utility functions (parameterised — no module-level global state)
# ---------------------------------------------------------------------------


def sort_extensions_by_dependency(extensions, get_extension_class_fn):
    """Sort *extensions* dict so parents come before children.

    Parameters
    ----------
    extensions : dict
        Mapping of extension_name → params.
    get_extension_class_fn : callable
        ``get_extension_class(name)`` for the project.

    Returns
    -------
    dict
        Re-ordered copy of *extensions*.
    """
    ext_list = list(extensions.keys())
    params_list = list(extensions.values())

    i = 0
    while i < len(ext_list):
        name = ext_list[i]
        params = params_list[i]

        ext_class = get_extension_class_fn(name)
        required = ext_class.get_required_dependencies(**params)
        optional = ext_class.get_optional_dependencies(**params)
        all_deps = list(chain.from_iterable(d.split("|") for d in required + optional))

        did_nothing = True
        for dep in all_deps:
            if dep in ext_list[i:]:
                dep_idx = ext_list.index(dep)
                params_list.insert(i, params_list.pop(dep_idx))
                ext_list.insert(i, ext_list.pop(dep_idx))
                did_nothing = False

        if did_nothing:
            i += 1

    return dict(zip(ext_list, params_list))


def get_children_dependencies(extension_name, extension_children):
    """Recursively collect all children that depend on *extension_name*.

    Parameters
    ----------
    extension_name : str
    extension_children : dict
        Mapping ``parent_name → [child_name, …]``.

    Returns
    -------
    list[str]
    """
    names = []
    children = extension_children.get(extension_name, [])
    for child in children:
        if child not in names:
            names.append(child)
        names.extend(get_children_dependencies(child, extension_children))
    return names


# ---------------------------------------------------------------------------
# BaseAnalyzer
# ---------------------------------------------------------------------------


class BaseAnalyzer:
    """Domain-agnostic base for analyzer objects.

    Captures the extension management lifecycle shared by
    ``SortingAnalyzer`` (recording / sorting) and ``RoiAnalyzer``
    (imaging / rois).  Storage backends (create_memory, create_zarr, …)
    remain in the subclasses.

    Subclasses **must** set the class attributes ``_input_name`` and
    ``_output_name`` and override every method listed under
    *Registry hooks*.
    """

    # -- Class-level names (override in subclass) --------------------------
    _input_name: str = "input"  # e.g. "recording" / "imaging"
    _output_name: str = "output"  # e.g. "sorting" / "rois"

    # -- Init --------------------------------------------------------------

    def _init_base(
        self,
        output_extractor,
        input_extractor=None,
        input_attributes=None,
        format=None,
        backend_options=None,
    ):
        """Initialise the generic parts shared by every analyzer.

        Call this from the subclass ``__init__``.
        """
        self._output_extractor = output_extractor
        self._input_extractor = input_extractor
        self._input_attributes = input_attributes
        self.format = format
        self.folder = None
        self._temporary_input = None
        self._backend_options = {} if backend_options is None else backend_options
        self.extensions = dict()

    # -- Input / output accessors ------------------------------------------

    def has_input(self) -> bool:
        """Whether the primary input extractor is available."""
        return self._input_extractor is not None

    def has_temporary_input(self) -> bool:
        """Whether a temporary input extractor has been set."""
        return self._temporary_input is not None

    @property
    def input_extractor(self):
        """Resolve the input extractor (temporary takes precedence)."""
        if not self.has_input() and not self.has_temporary_input():
            raise ValueError(f"{self.__class__.__name__} could not load the {self._input_name}")
        return self._temporary_input or self._input_extractor

    # -- Shared read-only / zarr helpers -----------------------------------

    def is_read_only(self) -> bool:
        import os

        if self.format == "memory":
            return False
        if self.format == "binary_folder":
            return not os.access(self.folder, os.W_OK)
        # zarr or other
        if not is_path_remote(str(self.folder)):
            return not os.access(self.folder, os.W_OK)
        return False

    def _get_zarr_root(self, mode="r+"):
        from .zarrextractors import super_zarr_open

        storage_options = self._backend_options.get("storage_options", {})
        return super_zarr_open(str(self.folder), mode=mode, storage_options=storage_options)

    # ======================================================================
    # Registry hooks — subclasses MUST override
    # ======================================================================

    def _get_extension_class(self, extension_name):
        """Return the extension class for *extension_name* (with auto-import)."""
        raise NotImplementedError

    def _get_children_dependencies(self, extension_name):
        """Return list of transitive child extension names."""
        raise NotImplementedError

    def _sort_extensions_by_dependency(self, extensions):
        """Return *extensions* dict sorted so parents precede children."""
        raise NotImplementedError

    def _get_available_extensions(self):
        """Return list of all built-in extension names."""
        raise NotImplementedError

    def _get_default_extension_params(self, extension_name):
        """Return default params dict for *extension_name*."""
        raise NotImplementedError

    def _get_extra_pipeline_kwargs(self):
        """Extra keyword arguments passed to ``run_node_pipeline``.

        Override in subclasses that need project-specific flags
        (e.g. ``check_for_peak_source=False``).
        """
        return {}

    # ======================================================================
    # Extension management — fully generic
    # ======================================================================

    def compute(self, input, save=True, extension_params=None, verbose=False, **kwargs):
        """Compute one or several extensions.

        Parameters
        ----------
        input : str | dict | list
            Extension name (str), dict of {name: params}, or list of names.
        save : bool, default: True
            Whether to persist computed extensions.
        extension_params : dict | None
            Per-extension params when *input* is a list.
        verbose : bool
            Print progress.
        **kwargs
            Passed to the extension's ``set_params`` (if str) or as job_kwargs.

        Returns
        -------
        BaseAnalyzerExtension | None
            The extension instance when *input* is a string, ``None`` otherwise.
        """
        from .job_tools import split_job_kwargs

        if isinstance(input, str):
            return self.compute_one_extension(extension_name=input, save=save, verbose=verbose, **kwargs)
        elif isinstance(input, dict):
            params_, job_kwargs = split_job_kwargs(kwargs)
            assert len(params_) == 0, f"Unexpected arguments: {set(params_)}"
            self.compute_several_extensions(extensions=input, save=save, verbose=verbose, **job_kwargs)
        elif isinstance(input, list):
            params_, job_kwargs = split_job_kwargs(kwargs)
            assert len(params_) == 0, f"Unexpected arguments: {set(params_)}"
            extensions = {k: {} for k in input}
            if extension_params is not None:
                for name, params in extension_params.items():
                    assert name in input, f"Extension '{name}' not in input list"
                    extensions[name] = params
            self.compute_several_extensions(extensions=extensions, save=save, verbose=verbose, **job_kwargs)
        else:
            raise ValueError("compute() expects a str, dict, or list")
        return None

    def compute_one_extension(self, extension_name, save=True, verbose=False, **kwargs):
        """Compute a single extension.

        Automatically deletes dependent extensions to keep data coherent.
        """
        from .job_tools import split_job_kwargs

        extension_class = self._get_extension_class(extension_name)

        for child in self._get_children_dependencies(extension_name):
            if self.has_extension(child):
                if verbose:
                    print(f"Deleting extension: {child}")
                self.delete_extension(child)

        params, job_kwargs = split_job_kwargs(kwargs)

        # Check input-data requirement
        if extension_class.need_input:
            assert (
                self.has_input() or self.has_temporary_input()
            ), f"Extension '{extension_name}' requires the {self._input_name}"

        # Check extension dependencies
        for dep in extension_class.get_required_dependencies(**params):
            if "|" in dep:
                ok = any(self.get_extension(d) is not None for d in dep.split("|"))
            else:
                ok = self.get_extension(dep) is not None
            assert ok, f"Extension '{extension_name}' requires '{dep}' to be computed first"

        extension_instance = extension_class(self)
        extension_instance.set_params(save=save, **params)
        if extension_class.need_job_kwargs:
            extension_instance.run(save=save, verbose=verbose, **job_kwargs)
        else:
            extension_instance.run(save=save, verbose=verbose)

        self.extensions[extension_name] = extension_instance
        return extension_instance

    def compute_several_extensions(self, extensions, save=True, verbose=False, **job_kwargs):
        """Compute several extensions respecting dependency order."""
        from .node_pipeline import run_node_pipeline

        # Validate dependencies
        ext_names = list(extensions.keys())
        for name, params in extensions.items():
            for dep in self._get_extension_class(name).get_required_dependencies(**params):
                if "|" in dep:
                    ok = any(self.has_extension(d) or d in ext_names for d in dep.split("|"))
                else:
                    ok = self.has_extension(dep) or dep in ext_names
                assert ok, f"Extension '{name}' requires '{dep}' to be computed first"

        sorted_exts = self._sort_extensions_by_dependency(extensions)

        # Delete children of extensions we're about to recompute
        for name in sorted_exts:
            for child in self._get_children_dependencies(name):
                if verbose:
                    print(f"Deleting extension: {child}")
                self.delete_extension(child)

        # Group: pipeline vs non-pipeline
        pipeline_exts = {}
        pre_pipeline_exts = {}
        post_pipeline_exts = {}
        for name, params in sorted_exts.items():
            ext_class = self._get_extension_class(name)
            if ext_class.use_nodepipeline:
                pipeline_exts[name] = params
            elif any(
                self._get_extension_class(d).use_nodepipeline
                for d in ext_class.get_any_dependencies(**params)
                if d in sorted_exts
            ):
                post_pipeline_exts[name] = params
            else:
                pre_pipeline_exts[name] = params

        # Pre-pipeline
        for name, params in pre_pipeline_exts.items():
            ext_class = self._get_extension_class(name)
            if ext_class.need_job_kwargs:
                self.compute_one_extension(name, save=save, verbose=verbose, **params, **job_kwargs)
            else:
                self.compute_one_extension(name, save=save, verbose=verbose, **params)

        # Pipeline extensions (run together)
        if len(pipeline_exts) > 0:
            all_nodes = []
            result_routage = []
            instances = {}

            for name, params in pipeline_exts.items():
                ext_class = self._get_extension_class(name)
                assert (
                    self.has_input() or self.has_temporary_input()
                ), f"Extension '{name}' requires the {self._input_name}"

                for var in ext_class.nodepipeline_variables:
                    result_routage.append((name, var))

                inst = ext_class(self)
                inst.set_params(save=save, **params)
                instances[name] = inst
                all_nodes.extend(inst.get_pipeline_nodes())

            job_name = "Compute: " + " + ".join(pipeline_exts.keys())
            t0 = perf_counter()
            results = run_node_pipeline(
                self.input_extractor,
                all_nodes,
                job_kwargs=job_kwargs,
                job_name=job_name,
                gather_mode="memory",
                squeeze_output=False,
                verbose=verbose,
                **self._get_extra_pipeline_kwargs(),
            )
            runtime_s = perf_counter() - t0

            for r, result in enumerate(results):
                ext_name, var_name = result_routage[r]
                instances[ext_name].data[var_name] = result
                instances[ext_name].run_info["runtime_s"] = runtime_s
                instances[ext_name].run_info["run_completed"] = True

            for name, inst in instances.items():
                self.extensions[name] = inst
                if save:
                    inst.save()

        # Post-pipeline
        for name, params in post_pipeline_exts.items():
            ext_class = self._get_extension_class(name)
            if ext_class.need_job_kwargs:
                self.compute_one_extension(name, save=save, verbose=verbose, **params, **job_kwargs)
            else:
                self.compute_one_extension(name, save=save, verbose=verbose, **params)

    def get_saved_extension_names(self):
        """Get extension names saved on disk (without loading data)."""
        saved = []
        if self.format == "binary_folder":
            ext_folder = self.folder / "extensions"
            if ext_folder.is_dir():
                for d in ext_folder.iterdir():
                    if d.is_dir() and (d / "params.json").is_file():
                        saved.append(d.stem)
        elif self.format == "zarr":
            zarr_root = self._get_zarr_root(mode="r")
            if "extensions" in zarr_root.keys():
                for name in zarr_root["extensions"].keys():
                    if "params" in zarr_root["extensions"][name].attrs.keys():
                        saved.append(name)
        return saved

    def get_extension(self, extension_name):
        """Get an extension, auto-loading from disk if needed.

        Returns ``None`` if the extension has not been computed.
        """
        if extension_name in self.extensions:
            return self.extensions[extension_name]
        elif self.format != "memory" and self.has_extension(extension_name):
            self.load_extension(extension_name)
            return self.extensions[extension_name]
        return None

    def load_extension(self, extension_name):
        """Load an extension from disk into memory."""
        assert self.format != "memory", "load_extension() is for non-memory formats"

        extension_class = self._get_extension_class(extension_name)
        if extension_class is None:
            return None

        ext_instance = extension_class.load(self)
        self.extensions[extension_name] = ext_instance
        return ext_instance

    def load_all_saved_extension(self):
        """Load all saved extensions into memory."""
        for name in self.get_saved_extension_names():
            self.load_extension(name)

    def delete_extension(self, extension_name):
        """Delete an extension from memory and from disk."""
        if self.format != "memory" and self.has_extension(extension_name):
            ext = self.load_extension(extension_name)
            if ext is not None:
                ext.delete()
        self.extensions.pop(extension_name, None)

    def get_loaded_extension_names(self):
        """Return names of currently loaded extensions."""
        return list(self.extensions.keys())

    def has_extension(self, extension_name):
        """Check if an extension exists (in memory or on disk)."""
        if extension_name in self.extensions:
            return True
        if self.format == "memory":
            return False
        return extension_name in self.get_saved_extension_names()

    def get_computable_extensions(self):
        """List all registered extension names."""
        return self._get_available_extensions()

    def get_default_extension_params(self, extension_name):
        """Get default params for an extension."""
        return self._get_default_extension_params(extension_name)
