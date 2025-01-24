"""
Preprocessing pipeline for jet taggging.

By default all stages for the training split are run.
To run with only specific stages enabled, include the flag for the required stages.
To run without certain stages, include the corresponding negative flag.

Note that all stages are required to run the pipeline. If you want to disable resampling,
you need to set method: none in your config file.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from ftag.cli_utils import HelpFormatter, valid_path

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.logger import setup_logger
from upp.stages.decompose import decompose
from upp.stages.hist import create_histograms
from upp.stages.merging import Merging
from upp.stages.normalisation import Normalisation
from upp.stages.plot import plot_initial_resampling_dists, plot_resampled_dists
from upp.stages.resampling import resample


def parse_args(args):
    _st = "store_true"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument("--config", required=True, type=valid_path, help="Path to config file")
    parser.add_argument("--prep", action=_st, default=None, help="Estimate and write PDFs")
    parser.add_argument("--no-prep", dest="prep", action="store_false")
    parser.add_argument("--merge", action=_st, default=None, help="Run merging")
    parser.add_argument("--no-merge", dest="merge", action="store_false")
    parser.add_argument("--norm", action=_st, default=None, help="Compute normalisations")
    parser.add_argument("--no-norm", dest="norm", action="store_false")
    parser.add_argument("--plot", action=_st, default=None, help="Plot output distributions")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    splits = ["train", "val", "test", "all"]
    parser.add_argument("--split", default="train", choices=splits, help="Which file to produce")

    args = parser.parse_args(args)
    d = vars(args)
    ignore = ["config", "split"]
    if not any(v for a, v in d.items() if a not in ignore):
        for v in d:
            if v not in ignore and d[v] is None:
                d[v] = True
    return args


def run_pp(args) -> None:
    log = setup_logger()

    # print start info
    log.info("[bold green]Starting preprocessing...")
    start = datetime.now()
    log.info(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    # load config
    config = PreprocessingConfig.from_file(args.config, args.split)

    # create virtual datasets and pdf files
    # TODO: we might also want to write histograms when we
    # don't have the bins from the resampling config
    if args.prep and args.split == "train" and hasattr(config, "resampling_config"):
        create_histograms(config)

    # run the resampling
    # TODO: pass paths explicitly
    decompose(
        config.components,
        config.variables,
        config.batch_size,
        config.jets_name,
        config.transform
    )

    components_dir = config.components_dir
    if hasattr(config, "resampling_config"):
        resample(
            config.components,
            config.components_dir,
            config.resampled_components_dir,
            config.variables,
            config.batch_size,
            config.resampling_config,
        )
        components_dir = config.resampled_components_dir

    # run the merging
    if args.merge:
        # TODO: this should be a function; making this a class
        # has no benefits while providing access to the config
        # in every method of the class, making it very hard to track
        # the responsibilities of each function
        # TODO: the run function has major sideeffects in terms of writing
        # multiple files/directories to disk, none of which are evident
        # by the call in main; path's should be passed explicitly
        merging = Merging(config)
        merging.run(components_dir)

    # run the normalisation
    if args.norm and args.split == "train":
        # TODO: this should be a function, see above comment
        # TODO: path's should be passed explicitly to run, see above comment
        norm = Normalisation(config)
        norm.run()

    # make plots
    if args.plot:
        title = " Plotting "
        log.info(f"[bold green]{title:-^100}")
        plot_initial_resampling_dists(config=config)
        plot_resampled_dists(config=config, stage=args.split)

    # print end info
    end = datetime.now()
    title = " Finished Preprocessing! "
    log.info(f"[bold green]{title:-^100}")
    log.info(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Elapsed time: {str(end - start).split('.')[0]}")


def main(args=None) -> None:
    args = parse_args(args)
    log = setup_logger()

    if args.split == "all":
        d = vars(args)
        for split in ["train", "val", "test"]:
            d["split"] = split
            log.info(f"[bold blue]{'-'*100}")
            title = f" {args.split} "
            log.info(f"[bold blue]{title:-^100}")
            log.info(f"[bold blue]{'-'*100}")
            run_pp(args)
    else:
        run_pp(args)


if __name__ == "__main__":
    main()
