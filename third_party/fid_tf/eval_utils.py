# CREDITS: https://github.com/yang-song/score_sde/blob/main/evaluation.py
"""Utility functions for computing FID/Inception scores."""

import os

import jax
import numpy as np
import six
import tensorflow as tf
import tensorflow_hub as tfhub
from PIL import Image

from classifier_metrics_numpy import trace_sqrt_product

INCEPTION_TFHUB = "https://tfhub.dev/tensorflow/tfgan/eval/inception/1"
INCEPTION_OUTPUT = "logits"
INCEPTION_FINAL_POOL = "pool_3"
_DEFAULT_DTYPES = {INCEPTION_OUTPUT: tf.float32, INCEPTION_FINAL_POOL: tf.float32}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def get_inception_model(inceptionv3=False):
    print(f"Using InceptionV3: {inceptionv3}")
    if inceptionv3:
        return tfhub.load(
            "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
        )
    else:
        return tfhub.load(INCEPTION_TFHUB)


# TODO: Update this as we add more datasets
def load_dataset_stats(config):
    """Load the pre-computed dataset statistics."""
    if config.data.dataset == "CIFAR10":
        filename = "assets/stats/cifar10_stats.npz"
    elif config.data.dataset == "CELEBA":
        filename = "assets/stats/celeba_stats.npz"
    elif config.data.dataset == "LSUN":
        filename = f"assets/stats/lsun_{config.data.category}_{config.data.image_size}_stats.npz"
    else:
        raise ValueError(f"Dataset {config.data.dataset} stats not found.")

    with tf.io.gfile.GFile(filename, "rb") as fin:
        stats = np.load(fin)
        return stats


def classifier_fn_from_tfhub(output_fields, inception_model, return_tensor=False):
    """Returns a function that can be as a classifier function.
    Copied from tfgan but avoid loading the model each time calling _classifier_fn
    Args:
      output_fields: A string, list, or `None`. If present, assume the module
        outputs a dictionary, and select this field.
      inception_model: A model loaded from TFHub.
      return_tensor: If `True`, return a single tensor instead of a dictionary.
    Returns:
      A one-argument function that takes an image Tensor and returns outputs.
    """
    if isinstance(output_fields, six.string_types):
        output_fields = [output_fields]

    def _classifier_fn(images):
        output = inception_model(images)
        if output_fields is not None:
            output = {x: output[x] for x in output_fields}
        if return_tensor:
            assert len(output) == 1
            output = list(output.values())[0]
        return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

    return _classifier_fn


def run_classifier_fn(
    input_tensor, classifier_fn, num_batches=1, dtypes=None, name="RunClassifierFn"
):
    """Copied from tfgan to avoid tfgan dependency. Runs a network from a TF-Hub module.
    If there are multiple outputs, cast them to tf.float32.
    Args:
      input_tensor: Input tensors.
      classifier_fn: A function that takes a single argument and returns the
        outputs of the classifier. If `num_batches` is greater than 1, the
        structure of the outputs of `classifier_fn` must match the structure of
        `dtypes`.
      num_batches: Number of batches to split `tensor` in to in order to
        efficiently run them through the classifier network. This is useful if
        running a large batch would consume too much memory, but running smaller
        batches is feasible.
      dtypes: If `classifier_fn` returns more than one element or `num_batches` is
        greater than 1, `dtypes` must have the same structure as the return value
        of `classifier_fn` but with each output replaced by the expected dtype of
        the output. If `classifier_fn` returns on element or `num_batches` is 1,
        then `dtype` can be `None.
      name: Name scope for classifier.
    Returns:
      The output of the module, or just `outputs`.
    Raises:
      ValueError: If `classifier_fn` return multiple outputs but `dtypes` isn't
        specified, or is incorrect.
    """
    if num_batches > 1:
        # Compute the classifier splits using the memory-efficient `map_fn`.
        input_list = tf.split(input_tensor, num_or_size_splits=num_batches)
        classifier_outputs = tf.map_fn(
            fn=classifier_fn,
            elems=tf.stack(input_list),
            dtype=dtypes,
            parallel_iterations=1,
            back_prop=False,
            swap_memory=True,
            name=name,
        )
        classifier_outputs = tf.nest.map_structure(
            lambda x: tf.concat(tf.unstack(x), 0), classifier_outputs
        )
    else:
        classifier_outputs = classifier_fn(input_tensor)

    return classifier_outputs


@tf.function
def run_inception_jit(inputs, inception_model, num_batches=1, inceptionv3=False):
    """Running the inception network. Assuming input is within [0, 255]."""
    if not inceptionv3:
        inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
    else:
        inputs = tf.cast(inputs, tf.float32) / 255.0

    return run_classifier_fn(
        inputs,
        num_batches=num_batches,
        classifier_fn=classifier_fn_from_tfhub(None, inception_model),
        dtypes=_DEFAULT_DTYPES,
    )


@tf.function
def run_inception_distributed(
    input_tensor, inception_model, num_batches=1, inceptionv3=False
):
    """Distribute the inception network computation to all available TPUs.
    Args:
      input_tensor: The input images. Assumed to be within [0, 255].
      inception_model: The inception network model obtained from `tfhub`.
      num_batches: The number of batches used for dividing the input.
      inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.
    Returns:
      A dictionary with key `pool_3` and `logits`, representing the pool_3 and
        logits of the inception network respectively.
    """
    num_tpus = jax.local_device_count()
    input_tensors = tf.split(input_tensor, num_tpus, axis=0)
    pool3 = []
    logits = [] if not inceptionv3 else None
    device_format = "/TPU:{}" if "TPU" in str(jax.devices()[0]) else "/GPU:{}"
    for i, tensor in enumerate(input_tensors):
        with tf.device(device_format.format(i)):
            tensor_on_device = tf.identity(tensor)
            res = run_inception_jit(
                tensor_on_device,
                inception_model,
                num_batches=num_batches,
                inceptionv3=inceptionv3,
            )

            if not inceptionv3:
                pool3.append(res["pool_3"])
                logits.append(res["logits"])  # pytype: disable=attribute-error
            else:
                pool3.append(res)

    with tf.device("/CPU"):
        return {
            "pool_3": tf.concat(pool3, axis=0),
            "logits": tf.concat(logits, axis=0) if not inceptionv3 else None,
        }


def load_samples_from_path(dirpath, mode="image"):
    # Read either images or numpy samples
    assert mode in ["image", "numpy"]
    assert os.path.isdir(dirpath)

    # Whitelisted extensions for either mode
    ext_dict = {"image": [".jpg", ".png", ".jpeg"], "numpy": [".npy", ".npz"]}

    # Placeholder for all samples
    samples_list = []

    for sample in os.listdir(dirpath):
        # Allow only whitelisted extensions. Maybe relax this?
        _, ext = os.path.splitext(sample)
        assert ext in ext_dict[mode]

        sample_path = os.path.join(dirpath, sample)
        if mode == "image":
            samples_list.append(np.asarray(Image.open(sample_path)))
        else:
            # Read the numpy image file (assuming between -1 and 1)
            sample_np = np.load(sample_path).astype(np.float)
            sample_np = sample_np * 0.5 + 0.5
            sample_np = (sample_np * 255).clip(0, 255).astype(np.uint8)
            samples_list.append(sample_np)

    # Concat all samples into a big numpy array
    samples = np.stack(samples_list, axis=0)

    # Assuming the directory only contains samples
    assert samples.shape[0] == len(os.listdir(dirpath))
    return samples


def get_inception_features(samples, inception_v3=False, num_batches=1):
    # Get the inception network activations (with "pool_3" and "logits")
    model = get_inception_model(inceptionv3=inception_v3)

    feature_dict = run_inception_distributed(
        samples,
        model,
        num_batches=num_batches,
        inceptionv3=inception_v3,
    )
    return feature_dict


def compute_sample_stats(activations):
    """A helper function for computing the mean and covariance for FID computation.
    Adapted from _frechet_classifier_distance_from_activations_helper in tfgan"""
    assert len(activations.shape) == 2

    activations_dtype = activations.dtype
    if activations_dtype != np.float64:
        activations = activations.astype(np.float64)

    # Compute mean and covariance matrices of activations.
    m = np.mean(activations, axis=0)
    # Calculate the unbiased covariance matrix of first activations.
    num_examples = float(activations.shape[0])
    sigma = num_examples / (num_examples - 1) * np.cov(activations, rowvar=False)
    return m, sigma


def calculate_fid(m, m_w, sigma, sigma_w, dtype=np.float64):
    """Returns the Frechet distance given the sample mean and covariance.
    Adapted from _calculate_fid in tfgan"""
    # Find the Tr(sqrt(sigma sigma_w)) component of FID
    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

    # Compute the two components of FID.

    # First the covariance component.
    # Here, note that trace(A + B) = trace(A) + trace(B)
    trace = np.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    # Next the distance between means.
    mean = np.sum(np.square(m - m_w))  # Equivalent to L2 but more stable.
    fid = trace + mean
    fid = fid.astype(dtype)
    return fid
