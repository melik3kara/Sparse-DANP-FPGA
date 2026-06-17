from __future__ import annotations

import tensorflow as tf
import tensorflow.keras as keras


class DecorrelatedDense(keras.layers.Dense):
    """
    Dense layer with:
    - optional linear input decorrelation x -> R x
    - cached clean/noisy inputs and pre-activations for perturbation updates
    """

    def __init__(self, input_dim: int, units: int, activation, name: str):
        super().__init__(
            units=units,
            activation=None,
            use_bias=True,
            kernel_initializer=keras.initializers.HeNormal(),
            bias_initializer=keras.initializers.Zeros(),
            name=name,
        )
        self.activation_fn = activation
        self.R = tf.Variable(tf.eye(input_dim), trainable=False, name=f"{name}_R")

        self.inputs_clean = None
        self.outputs_clean = None
        self.inputs_noisy = None
        self.outputs_noisy = None
        self.noise = None
        self.noise_mask = None

    def decorrelate_inputs(self, x: tf.Tensor) -> tf.Tensor:
        return tf.einsum("ji,ni->nj", self.R, x)

    def forward(self, x: tf.Tensor, decorrelate: bool) -> tf.Tensor:
        if len(x.shape) > 2:
            x = tf.reshape(x, [tf.shape(x)[0], -1])
        if decorrelate:
            x = self.decorrelate_inputs(x)
        self.inputs_clean = x
        self.outputs_clean = self(self.inputs_clean)
        return self.activation_fn(self.outputs_clean)

    def forward_noisy(self, x: tf.Tensor, decorrelate: bool, add_noise: bool) -> tf.Tensor:
        if len(x.shape) > 2:
            x = tf.reshape(x, [tf.shape(x)[0], -1])
        if decorrelate:
            x = self.decorrelate_inputs(x)
        self.inputs_noisy = x
        self.outputs_noisy = self(self.inputs_noisy)
        if add_noise:
            self.outputs_noisy = self.outputs_noisy + self.noise
        return self.activation_fn(self.outputs_noisy)

    def reset_noise(self, noise_std: float, mask: tf.Tensor | None = None) -> None:
        if self.outputs_clean is None:
            raise RuntimeError("Call a clean forward pass before reset_noise().")
        noise = tf.random.normal(
            shape=tf.shape(self.outputs_clean),
            mean=0.0,
            stddev=noise_std,
            dtype=self.outputs_clean.dtype,
        )
        if mask is not None:
            # mask has shape [1, units] and broadcasts over the batch axis.
            # Unselected nodes receive exactly zero noise (sparse perturbation).
            noise = noise * tf.cast(mask, noise.dtype)
        self.noise = noise
        self.noise_mask = mask


class MLP(keras.Model):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        output_dim: int,
        hidden_activation=tf.nn.leaky_relu,
        output_activation=tf.nn.softmax,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_sizes) + [output_dim]
        self.layers_list: list[DecorrelatedDense] = []

        for i in range(len(dims) - 1):
            act = output_activation if i == len(dims) - 2 else hidden_activation
            layer = DecorrelatedDense(
                input_dim=dims[i],
                units=dims[i + 1],
                activation=act,
                name=f"dense_{i}",
            )
            self.layers_list.append(layer)

        dummy = tf.zeros([1, input_dim], dtype=tf.float32)
        _ = self.forward(dummy, decorrelate=False)

    def forward(self, x: tf.Tensor, decorrelate: bool) -> tf.Tensor:
        for layer in self.layers_list:
            x = layer.forward(x, decorrelate=decorrelate)
        return x

    def forward_noisy(
        self,
        x: tf.Tensor,
        decorrelate: bool,
        noise_layer_idx: int | None = None,
    ) -> tf.Tensor:
        for i, layer in enumerate(self.layers_list):
            add_noise = (noise_layer_idx is None) or (noise_layer_idx == i)
            x = layer.forward_noisy(x, decorrelate=decorrelate, add_noise=add_noise)
        return x

    def reset_all_noise(self, noise_std: float, masks: list[tf.Tensor] | None = None) -> None:
        for i, layer in enumerate(self.layers_list):
            mask = masks[i] if masks is not None else None
            layer.reset_noise(noise_std, mask=mask)

    def ordered_trainable_variables(self) -> list[tf.Variable]:
        vars_out = []
        for layer in self.layers_list:
            vars_out.extend(layer.trainable_variables)
        return vars_out