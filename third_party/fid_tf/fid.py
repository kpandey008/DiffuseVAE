import os

import click
import numpy as np
import tensorflow as tf

from eval_utils import (
    calculate_fid,
    compute_sample_stats,
    get_inception_features,
    load_samples_from_path,
)


@click.group()
def cli():
    pass


def generate_fid_stats(
    samples, use_v3=False, num_batches=1, write_path="", stats_prefix=""
):
    inception_feats = get_inception_features(
        samples, inception_v3=use_v3, num_batches=num_batches
    )
    activations = inception_feats["pool_3"]
    mu, sigma = compute_sample_stats(activations.numpy())

    if write_path != "":
        # Write mu and sigma
        data = {"mu": mu.numpy(), "sigma": sigma.numpy()}
        np.save(os.path.join(write_path, f"precomputed_stats_{stats_prefix}.npz"), data)

    return mu, sigma


@cli.command()
@click.option("--num-batches", default=1)
@click.option("--use-v3", default=False, type=bool)
@click.option("--write-path", default="")
@click.option("--mode1", default="numpy", type=click.Choice(["numpy", "image"]))
@click.option("--mode2", default="numpy", type=click.Choice(["numpy", "image"]))
@click.argument("sample-path-1")
@click.argument("sample-path-2")
def compute_fid_from_samples(
    sample_path_1,
    sample_path_2,
    write_path="",
    use_v3=False,
    num_batches=1,
    mode1="numpy",
    mode2="numpy",
):
    if sample_path_1.endswith(".npz"):
        assert os.path.isfile(sample_path_1)
        print(f"Found precomputed statistics: {sample_path_1}")
        data = np.load(sample_path_1)
        mu_1, sigma_1 = data["mu"], data["sigma"]
    else:
        # Load the samples from the directories
        samples_1 = tf.convert_to_tensor(
            load_samples_from_path(sample_path_1, mode=mode1)
        )

        # Compute sample statistics
        print("Computing Inception stats for set1")
        mu_1, sigma_1 = generate_fid_stats(
            samples_1,
            write_path=write_path,
            use_v3=use_v3,
            num_batches=num_batches,
            stats_prefix="1",
        )

    if sample_path_2.endswith(".npz"):
        assert os.path.isfile(sample_path_2)
        print(f"Found precomputed statistics: {sample_path_2}")
        data = np.load(sample_path_2)
        mu_2, sigma_2 = data["mu"], data["sigma"]
    else:
        samples_2 = tf.convert_to_tensor(
            load_samples_from_path(sample_path_2, mode=mode2)
        )

        print("Computing Inception stats for set2")
        mu_2, sigma_2 = generate_fid_stats(
            samples_2,
            write_path=write_path,
            use_v3=use_v3,
            num_batches=num_batches,
            stats_prefix="2",
        )

    fid = calculate_fid(mu_1, mu_2, sigma_1, sigma_2)
    print(f"FID: {fid}")
    return fid


if __name__ == "__main__":
    cli()
