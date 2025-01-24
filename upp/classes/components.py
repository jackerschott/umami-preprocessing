from __future__ import annotations

import logging as log
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import numpy as np
from ftag import Cuts, Label, Sample
from ftag.hdf5 import H5Reader, H5Writer
from ftag.labels import LabelContainer

from upp.classes.region import Region
from upp.classes.variable_config import VariableConfig
from upp.stages.hist import Hist
from upp.types import Split


@dataclass
class Component:
    JETS_NAME = "jets"

    region: Region
    sample: Sample
    flavour: Label | None
    global_cuts: Cuts
    dirname: Path
    num_jets: int
    num_jets_estimate_available: int
    equal_jets: bool
    sampling_fraction: float | None = None

    def __post_init__(self):
        self.hist = Hist(self.dirname.parent.parent / "hists" / f"hist_{self.name}.h5")

    def setup_reader(
        self,
        path: Path | str | list[Path | str],
        batch_size: int,
        jets_name: str = JETS_NAME,
        **kwargs,
    ):
        self.reader = H5Reader(
            path,
            batch_size,
            jets_name=jets_name,
            equal_jets=self.equal_jets,
            **kwargs
        )
        log.debug(f"Setup component reader at: {path}")

    def setup_writer(
        self,
        path: Path | str,
        variables: VariableConfig,
        dtypes: dict[str, np.dtype],
        shapes: dict[str, tuple[int, ...]],
        jets_name: str = JETS_NAME
    ):
        dtypes = self.reader.dtypes(variables.combined())
        shapes = self.reader.shapes(self.num_jets, list(variables.keys()))
        self.writer = H5Writer(path, dtypes, shapes, jets_name)
        log.debug(f"Setup component writer at: {self.out_path}")


    @property
    def name(self):
        if self.flavour is None:
            return f"{self.region.name}_{self.sample.name}"
        else:
            return f"{self.region.name}_{self.sample.name}_{self.flavour.name}"

    def get_flavour(self) -> Label:
        assert self.flavour is not None
        return self.flavour

    @property
    def cuts(self):
        if self.flavour is None:
            return self.global_cuts + self.region.cuts
        else:
            return self.global_cuts + self.flavour.cuts + self.region.cuts

    @property
    def out_path(self) -> Path:
        return self.dirname / f"{self.name}.h5"

    def is_target(self, target_str):
        assert self.flavour is not None, (
            "expected is_target to only be called"
            " in resampling code, when self.flavour is expected to be set"
        )
        return self.flavour.name == target_str

    def get_jets(self, variables: list, num_jets: int, cuts: Cuts | None = None):
        jn = self.reader.jets_name
        return self.reader.load({jn: variables}, num_jets, cuts)[jn]

    def check_num_jets(
        self, num_required, sampling_frac=None, cuts=None, silent=False, raise_error=True
    ):
        # Check if num_jets jets are available after the cuts and sampling fraction
        num_estimated = (
            None if self.num_jets_estimate_available <= 0 else self.num_jets_estimate_available
        )
        total = self.reader.estimate_available_jets(cuts, num_estimated)
        available = total
        if sampling_frac:
            available = int(total * sampling_frac)

        # check with tolerance to avoid failure midway through preprocessing
        if available < num_required and raise_error:
            raise ValueError(
                f"{num_required:,} jets requested, but only {total:,} are estimated to be"
                f" in {self}. With a sampling fraction of {sampling_frac}, at most"
                f" {available:,} of these are available. You can either reduce the"
                " number of requested jets or increase the sampling fraction."
            )

        if not silent:
            log.debug(f"Sampling fraction {sampling_frac}")
            log.info(
                f"Estimated {available:,} {self} jets available - {num_required:,} requested"
                f"({self.reader.num_jets:,} in {self.sample})"
            )

    def get_auto_sampling_frac(self):
        num_estimated = (
            None if self.num_jets_estimate_available <= 0
            else self.num_jets_estimate_available
        )

        # TODO: estimated_available_jets can take None; fix this in ftag and remove the
        # pyright ignore comment
        total = self.reader.estimate_available_jets(
            self.cuts, num_estimated # pyright: ignore
        )

        # 1.1 is a tolerance factor
        auto_sampling_frac = round(1.1 * self.num_jets / total, 3) 

        return auto_sampling_frac

    def __str__(self):
        return self.name

    @property
    def unique_jets(self) -> int:
        return sum([r.get_attr("unique_jets") for r in self.reader.readers])

    def write(self, batch: dict[str, np.ndarray]) -> None:
        assert self.writer.num_written <= self.num_jets

        jet_count_after_write = (
            self.writer.num_written + len(batch[self.writer.jets_name])
        )
        if jet_count_after_write < self.num_jets:
            self.writer.write(batch)
        elif jet_count_after_write == self.num_jets:
            self.writer.write(batch)
        elif jet_count_after_write > self.num_jets:
            truncated_jet_count = self.num_jets - self.writer.num_written
            self.writer.write({
                name: variable_values[:truncated_jet_count]
                for name, variable_values in batch.items()
            })
        else:
            raise AssertionError()

    @property
    def write_is_complete(self) -> bool:
        return self.writer.num_written == self.num_jets


class Components:
    def __init__(self, components: list[Component]):
        self.components = components

    @classmethod
    def from_config(
        cls,
        components_config: dict,
        num_jets_estimate_available: int,
        split: Split,
        global_cuts: Cuts,
        ntuple_dir: Path,
        components_dir: Path,
        flavour_container: LabelContainer,
        is_test: bool,
        check_flavour_ratios: bool,
    ):
        components_list = []
        for component_config in components_config:
            assert (
                "equal_jets" not in component_config
            ), "equal_jets flag should be set in the sample config"
            region_cuts = (
                Cuts.empty() if is_test else Cuts.from_list(component_config["region"]["cuts"])
            )
            region = Region(component_config["region"]["name"], region_cuts + global_cuts)
            pattern = component_config["sample"]["pattern"]
            equal_jets = component_config["sample"].get("equal_jets", True)
            if isinstance(pattern, list):
                pattern = tuple(pattern)
            sample = Sample(
                pattern=pattern,
                ntuple_dir=ntuple_dir,
                name=component_config["sample"]["name"],
            )

            num_jets = component_config["num_jets"]
            if split == "val":
                num_jets = component_config.get("num_jets_val", num_jets // 10)
            elif split == "test":
                num_jets = component_config.get("num_jets_test", num_jets // 10)

            assert num_jets_estimate_available is not None
            if component_config.get("flavours") is None:
                components_list.append(
                    Component(
                        region,
                        sample,
                        None,
                        global_cuts,
                        components_dir,
                        num_jets,
                        num_jets_estimate_available,
                        equal_jets,
                    )
                )
            else:
                for name in component_config["flavours"]:
                    components_list.append(
                        Component(
                            region,
                            sample,
                            flavour_container[name],
                            global_cuts,
                            components_dir,
                            num_jets,
                            num_jets_estimate_available,
                            equal_jets,
                        )
                    )

        components = cls(components_list)
        if check_flavour_ratios:
            components.check_flavour_ratios()
        return components

    def check_flavour_ratios(self):
        assert self.flavours is not None, "expected "

        ratios = {}
        for region, components in self.groupby_region():
            this_ratios = {}
            for flavour in self.flavours:
                this_ratios[flavour.name] = components[flavour].num_jets / components.num_jets
            ratios[region] = this_ratios

        ref = next(iter(ratios.values()))
        ref_region = next(iter(ratios.keys()))
        for i, (region, ratio) in enumerate(ratios.items()):
            if i != 0 and not np.allclose(list(ratio.values()), list(ref.values())):
                raise ValueError(
                    f"Found inconsistent flavour ratios: \n - {ref_region}: {ref} \n -"
                    f" {region}: {ratio}"
                )

    @property
    def regions(self):
        return list(set(c.region for c in self))

    @property
    def samples(self):
        return list(set(c.sample for c in self))

    @property
    def flavours(self) -> list[Label] | None:
        if any(c.flavour is None for c in self):
            assert all(
                c.flavour is None for c in self
            ), "expected to never have mixed components with and without flavours"
            return None
        else:
            # the if is needed to satisfy type checkers, c.flavour should never be None
            # due to the if-statement here
            return list(
                set(component.flavour for component in self if component.flavour is not None)
            )

    @property
    def cuts(self):
        return sum((c.cuts for c in self), Cuts.from_list([]))

    @property
    def num_jets(self):
        return sum(c.num_jets for c in self)

    @property
    def unique_jets(self):
        return sum(c.unique_jets for c in self)

    @property
    def out_dir(self):
        out_dir = {c.out_path.parent for c in self}
        assert len(out_dir) == 1
        return next(iter(out_dir))

    @property
    def jet_counts(self):
        num_dict = {
            c.name: {"num_jets": int(c.num_jets), "unique_jets": int(c.unique_jets)} for c in self
        }
        num_dict["total"] = {
            "num_jets": int(self.num_jets),
            "unique_jets": int(self.unique_jets),
        }
        return num_dict

    @property
    def dsids(self):
        return list(set(sum([c.sample.dsid for c in self], [])))  # noqa: RUF017

    @property
    def equal_jets(self) -> bool:
        equal_jet_flags = [component.equal_jets for component in self]
        if len(set(equal_jet_flags)) != 1:
            raise ValueError(
                "expected equal_jets to only be accessed for components"
                "belonging to a single sample in which case the values are expected to"
                f"all be equal, found {equal_jet_flags} however"
            )
        return equal_jet_flags[0]

    def groupby_region(self) -> list[tuple[Region, Components]]:
        return [
            (r, Components([c for c in self if c.region == r])) for r in self.regions
        ]

    def groupby_sample(self) -> list[tuple[Sample, Components]]:
        return [
            (s, Components([c for c in self if c.sample == s])) for s in self.samples
        ]

    def __iter__(self):
        yield from self.components

    def __getitem__(self, index_or_label: int | Label):
        if isinstance(index_or_label, int):
            return self.components[index_or_label]
        elif isinstance(index_or_label, Label):
            assert (
                self.flavours is not None
            ), "expected to only index components by label when flavours are available"
            return self.components[self.flavours.index(index_or_label)]
        else:
            raise AssertionError()

    def __len__(self):
        return len(self.components)

    def setup_writers(
        self,
        directory_path: Path,
        variables: VariableConfig,
        dtypes: dict[str, np.dtype],
        get_shape: Callable[[Component], dict[str, tuple[int, ...]]],
        jets_name: str = Component.JETS_NAME,
    ):
        for component in self:
            component.setup_writer(
                directory_path / f"{component.name}.h5",
                variables,
                dtypes,
                get_shape(component),
                jets_name
            )

    def write(self, batch: dict[str, np.ndarray]):
        assert not all(component.write_is_complete for component in self)

        for component in self:
            if not component.write_is_complete:
                component.write(batch)
