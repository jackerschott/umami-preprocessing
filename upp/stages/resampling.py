import logging as log
from pathlib import Path
from typing import Literal
from ftag.hdf5 import H5Reader
import numpy as np
import random

from upp.classes.components import Components
from upp.classes.resampling_config import ResamplingConfig
from upp.classes.variable_config import VariableConfig
from upp.stages.hist import bin_jets
from upp.stages.interpolation import subdivide_bins, upscale_array_regionally

# TODO: use a function-local fixed seed generator instead
random.seed(42)

def resample(
    components: Components,
    components_directory_path: Path,
    output_directory_path: Path,
    # TODO: remove, just stream all variables that were selected in the decompose stage
    variables_to_stream: VariableConfig,
    batch_size: int,
    config: ResamplingConfig
):
    SELECT_FUNCTION_BY_METHOD = {
        "pdf": pdf_select_func,
        "countup": countup_select_func,
        "none": None,
        None: None,
    }

    # TODO: this should be checked when creating the config
    if config.method not in SELECT_FUNCTION_BY_METHOD:
        raise ValueError(
            f"Unknown resampling method `{config.method}',"
            " expected one of {resampling_methods}".format(
                resampling_methods=", ".join(
                    [str(key) for key in SELECT_FUNCTION_BY_METHOD.keys()]
                )
            )
        )

    log.info("[bold green]{'Running resampling':-^100}")
    log.info(f"Resampling method: {config.method}")

    # TODO: remove paths in component class, path's should not be hidden in the
    # arguments of functions with disk i/o
    for component in components:
        component.setup_reader(
            components_directory_path / f"{component.name}.h5",
            batch_size
        )
        component.setup_writer(
            output_directory_path / f"{component.name}",
            variables_to_stream,
            dtypes,
            shapes
        )

    set_sampling_fractions(
        components,
        config.sampling_fraction or "auto",
        config.target_flavour_label,
        config.method
    )

    log.info(
        "[bold green]Checking requested num_jets based on"
        f"a sampling fraction of {config.sampling_fraction}..."
    )
    for component in components:
        # TODO: why are we passing arguments that are already known by the
        # component object? Since this function is also used in hist.py we probably
        # want to use default arguments
        component.check_num_jets(
            component.num_jets, component.sampling_fraction, component.cuts
        )

    for component in components:
        for jet_batch in component.reader.stream(
            variables_to_stream.combined(), batch_size
        ):
            flavour_mask_indices, _ = component.get_flavour().cuts(
                jet_batch[component.reader.jets_name]
            )
            jet_batch = apply_batch_mask(
                jet_batch,
                flavour_mask_indices,
            )

            resample_function = SELECT_FUNCTION_BY_METHOD[config.method]
            sample_collection_indices = np.arange(
                len(jet_batch[component.reader.jets_name])
            )
            if resample_function is not None \
                and not component.is_target(config.target_flavour_label):
                sample_collection_indices = resample_function(
                    jet_batch[component.reader.jets_name], component
                )
                if len(sample_collection_indices) == 0:
                    continue

                jet_batch = apply_batch_mask(jet_batch, sample_collection_indices)

            track_upsampling_stats(sample_collection_indices, component)

            component.writer.write(jet_batch)

    num_unique_jets_total = 0
    for component in components:
        num_unique_jets = component.writer.get_attr("unique_jets")
        assert isinstance(num_unique_jets, int)
        num_unique_jets_total += num_unique_jets

    log.info(
        f"[bold green]Finished resampling a total of {components.num_jets:,} jets!"
    )
    log.info(
        f"[bold green]Estimated number of unique jets: {num_unique_jets_total:,.0f}"
    )
    log.info(f"[bold green]Saved to {output_directory_path}/")


def countup_select_func(self, jets, component):
    random_generator = np.random.default_rng(42)

    if self.upscale_pdf != 1:
        raise ValueError("Upscaling of histogrms is not supported for countup method")
    num_jets = int(len(jets) * component.sampling_fraction)
    target_pdf = self.target.hist.pbin
    target_hist = target_pdf * num_jets
    target_hist = np.floor(
        target_hist + random_generator.random(target_pdf.shape)
    ).astype(int)

    _hist, binnumbers = bin_jets(jets[self.config.vars], self.config.flat_bins)
    assert target_pdf.shape == _hist.shape

    # loop over bins and select relevant jets
    all_idx = []
    for bin_id in np.ndindex(*target_hist.shape):
        idx = np.where((bin_id == binnumbers.T).all(axis=-1))[0][: target_hist[bin_id]]
        if len(idx) and len(idx) < target_hist[bin_id]:
            idx = np.concatenate(
                [idx, random_generator.choice(idx, target_hist[bin_id] - len(idx))]
            )
        all_idx.append(idx)
    idx = np.concatenate(all_idx).astype(int)
    if len(idx) < num_jets:
        idx = np.concatenate([idx, random_generator.choice(idx, num_jets - len(idx))])
    random_generator.shuffle(idx)
    return idx


def pdf_select_func(self, jets, component):
    # bin jets
    if self.upscale_pdf > 1:
        bins = [subdivide_bins(bins, self.upscale_pdf) for bins in self.config.flat_bins]
    else:
        bins = self.config.flat_bins

    _hist, binnumbers = bin_jets(jets[self.config.vars], bins)
    # assert target_shape == _hist.shape
    if binnumbers.ndim > 1:
        binnumbers = tuple(binnumbers[i] for i in range(len(binnumbers)))

    # importance sample with replacement
    num_samples = int(len(jets) * component.sampling_fraction)
    ratios = safe_divide(self.target.hist.pbin, component.hist.pbin)
    if self.upscale_pdf > 1:
        ratios = upscale_array_regionally(ratios, self.upscale_pdf, self.num_bins)
    probs = ratios[binnumbers]
    # TODO: why are we using random here and not numpy.random?
    idx = random.choices(np.arange(len(jets)), weights=list(probs), k=num_samples)
    return idx


def set_sampling_fractions(
    components: Components,
    sampling_fraction: float | Literal["auto"],
    target_flavour_label: str,
    method: str | None
):
    if sampling_fraction == "auto":
        log.info(f"[bold green]Choosing sampling fractions automatically...")

    for component in components:
        if component.is_target(target_flavour_label) or method is None:
            component.sampling_fraction = 1.0
        elif sampling_fraction == "auto":
            # TODO: why is max used here? Please explain this with a comment
            component.sampling_fraction = max(component.get_auto_sampling_frac(), 0.1)
        else:
            component.sampling_fraction = sampling_fraction

        if sampling_fraction == "auto" and component.sampling_fraction > 1.0 \
            and method == "countup":
            raise ValueError(
                f"[bold red]A sampling fraction of {component.sampling_fraction:.3f}"
                f"> 1 is needed for the component `{component}'. This is not supported"
                "for the countup method."
            )
        elif sampling_fraction == "auto" and component.sampling_fraction > 1.0 \
            and method != "countup":
            log.warning(
                f"[bold yellow]A sampling fraction of {component.sampling_fraction:.3f}"
                f"> 1 is needed for the component `{component}'"
            )


def track_upsampling_stats(idx, component):
    unique, ups_counts = np.unique(idx, return_counts=True)
    component._unique_jets += len(unique)
    max_ups = ups_counts.max()
    component._ups_max = max_ups if max_ups > component._ups_max else component._ups_max


def apply_batch_mask(batch: dict[str, np.ndarray], mask_indices: np.ndarray):
    return {name: array[mask_indices] for name, array in batch.items()}


def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
