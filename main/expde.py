import click
import os
import numpy as np
from joblib import dump

from sklearn.mixture import GaussianMixture


@click.group()
def cli():
    pass


@cli.command()
@click.argument("latent-path")
@click.option("--save-path", default=os.getcwd())
@click.option("--n-components", default=10)
def fit_gmm(
    latent_path,
    save_path=os.getcwd(),
    n_components=10,
):
    # Load latents
    z = np.load(latent_path).squeeze()

    # Fit a GMM model
    gmm = GaussianMixture(
        n_components=n_components, random_state=0, verbose=2, verbose_interval=1
    )
    gmm.fit(z)

    # Save the sklearn model on disk
    os.makedirs(save_path, exist_ok=True)
    s = dump(gmm, os.path.join(save_path, f"gmm_{n_components}.joblib"))


if __name__ == "__main__":
    cli()
