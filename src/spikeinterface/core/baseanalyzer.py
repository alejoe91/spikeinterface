"""
BaseAnalyzerExtension: domain-agnostic base class for analyzer extensions.

Provides persistence (binary_folder / zarr), parameter management,
dependency tracking, and run lifecycle shared by SpikeInterface's
AnalyzerExtension and downstream projects (e.g. photon-mosaic).

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

from .core_tools import check_json, retrieve_importing_provenance
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
