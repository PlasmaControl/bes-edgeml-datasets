import os
from typing import Tuple, List, Union

import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.25)


class RawFeatureModel(nn.Module):
    def __init__(
        self,
        sws: int,
        dropout_rate: float = 0.4,
        negative_slope: float = 0.02,
        maxpool_size: int = 2,
        num_filters: int = 48,
    ):
        """
        Use the raw BES channels values as features. This function takes in a 5-dimensional
        tensor of size: `(N, 1, signal_window_size, 8, 8)`, N=batch_size and
        performs maxpooling to downsample the spatial dimension by half, perform a
        3-d convolution with a filter size identical to the spatial dimensions of the
        input to avoid the sliding of the kernel over the input. Finally, a feature map
        is generated that can be concatenated with other features.

        Args:
        -----
            sws (int): Size of signal window.
            dropout_rate (float, optional): Fraction of total hidden units that will
                be turned off for drop out. Defaults to 0.2.
            negative_slope (float, optional): Slope of LeakyReLU activation for negative
                `x`. Defaults to 0.02.
            maxpool_size (int, optional): Size of the kernel used for maxpooling. Use
                0 to skip maxpooling. Defaults to 2.
            num_filters (int, optional): Dimensionality of the output space.
                Essentially, it gives the number of output kernels after convolution.
                Defaults to 10.
        """
        super(RawFeatureModel, self).__init__()
        pool_size = [1, maxpool_size, maxpool_size]
        self.sws = sws
        spatial_dim = int(8 // maxpool_size)
        filter_size = (self.sws, spatial_dim, spatial_dim)
        self.maxpool = nn.MaxPool3d(kernel_size=pool_size)
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=num_filters, kernel_size=filter_size
        )
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        x = self.dropout3d(self.conv(x))
        x = self.relu(x)
        x = torch.flatten(x, 1)

        return x


class FFTFeatureModel(nn.Module):
    def __init__(
        self,
        sws: int,
        dropout_rate: float = 0.4,
        negative_slope: float = 0.02,
        num_filters: int = 48,
    ):
        """
        Use the raw BES channels values as input and perform a Fast Fourier Transform
        to input signals. This function takes in a 5-dimensional
        tensor of size: `(N, 1, signal_window_size, 8, 8)`, N=batch_size and
        performs a FFT followed by an absolute value of the input tensor. It
        then performs a 3-d convolution with a filter size identical to the spatial
        dimensions of the input so that the receptive field is the same
        size as input in both spatial and temporal axes. Again, we will use the
        feature map and combine it with other features before feeding it into a
        classifier.

        Args:
        -----
            sws (int): Size of signal window.
            dropout_rate (float, optional): Fraction of total hidden units that will
                be turned off for drop out. Defaults to 0.2.
            negative_slope (float, optional): Slope of LeakyReLU activation for negative
                `x`. Defaults to 0.02.
            num_filters (int, optional): Dimensionality of the output space.
                Essentially, it gives the number of output kernels after convolution.
                Defaults to 10.
        """
        super(FFTFeatureModel, self).__init__()
        self.sws = sws
        temporal_size = int(self.sws // 2) + 1
        filter_size = (temporal_size, 8, 8)
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=num_filters, kernel_size=filter_size
        )
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):
        # apply FFT to the input along the time dimension
        # x = x.to(self.args.device)  # needed for PowerPC architecture
        x = torch.abs(torch.fft.rfft(x, dim=2))
        x = self.dropout3d(self.conv(x))
        x = self.relu(x)
        x = torch.flatten(x, 1)

        return x


class CWTFeatureModel(nn.Module):
    def __init__(
        self,
        dropout_rate: float = 0.4,
        negative_slope: float = 0.02,
        num_filters: int = 48,
    ):
        """
        Use features from the output of continuous wavelet transform. The model architecture
        is similar to the `RawFeatureModel`. This model takes in a 6-dimensional
        tensor of size: `(N, 1, signal_window_size, n_scales, 8, 8)`, where `N`=batch_size, and
        `n_scales`=number of different scales used (which are equal to `signal_window_size`).
        For each signal block, only the scales and BES channels for the leading time
        steps are used as model input which is a 5-dimensional tensor of size (`N, 1, n_scales, 8, 8)`.
        The model then performs a 3-d convolution with a filter size identical to
        the spatial dimensions of the input so that the receptive field is the same
        size as input in both spatial and temporal axes. In the end, we will have
        a feature map that can be concatenated with other features.

        Args:
        -----
            dropout_rate (float, optional): Fraction of total hidden units that will
                be turned off for drop out. Defaults to 0.2.
            negative_slope (float, optional): Slope of LeakyReLU activation for negative
                `x`. Defaults to 0.02.
            maxpool_size (int, optional): Size of the kernel used for maxpooling. Use
                0 to skip maxpooling. Defaults to 2.
            num_filters (int, optional): Dimensionality of the output space.
                Essentially, it gives the number of output kernels after convolution.
                Defaults to 10.
        """
        super(CWTFeatureModel, self).__init__()
        filter_size = (int(np.log2(1024)) + 1, 8, 8)
        self.conv = nn.Conv3d(
            in_channels=1, out_channels=num_filters, kernel_size=filter_size
        )
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout3d = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):
        x = x[:, :, -1, ...]  # take only the last time step
        x = self.dropout3d(self.conv(x))
        x = self.relu(x)
        x = torch.flatten(x, 1)

        return x


class MultiFeaturesModel(nn.Module):
    def __init__(
        self,
        raw_features_model: RawFeatureModel,
        fft_features_model: FFTFeatureModel,
        cwt_features_model: CWTFeatureModel,
        dropout_rate: float = 0.4,
        negative_slope: float = 0.02,
    ):
        """Encapsulate all the feature models to create a composite model that
        uses all the feature maps. It takes in the class instances of
        `RawFeatureModel`, `FFTFeatureModel` and `CWTFeatureModel` along with the
        dropout rate and negative slope of the LeakyReLU activation. Once all
        the feature maps are computed, it concatenates them together and pass them
        through a couple of fully connected layers wich act as the classifier.

        Args:
        -----
            raw_features_model (RawFeatureModel): Instance of `RawFeatureModel` class.
            fft_features_model (FFTFeatureModel): Instance of `FFTFeatureModel` class.
            cwt_features_model (CWTFeatureModel): Instance of `CWTFeatureModel` class.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.4.
            negative_slope (float, optional): Slope of activation functions. Defaults to 0.02.
        """
        super(MultiFeaturesModel, self).__init__()
        self.raw_features_model = raw_features_model
        self.fft_features_model = fft_features_model
        self.cwt_features_model = cwt_features_model
        input_features = 144  # if self.args.use_fft else 96
        self.fc1 = nn.Linear(in_features=input_features, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x_raw, x_cwt):
        # extract raw and cwt processed signals
        # if self.args.use_fft:
        raw_features = self.raw_features_model(x_raw)
        fft_features = self.fft_features_model(x_raw)
        cwt_features = self.cwt_features_model(x_cwt)
        x = torch.cat([raw_features, fft_features, cwt_features], dim=1)
        # else:
        #     raw_features = self.raw_features_model(x_raw)
        #     cwt_features = self.cwt_features_model(x_cwt)
        #     x = torch.cat([raw_features, cwt_features], dim=1)

        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x


def get_unlabeled_data(path: str, num_events: int = 50):
    hf = h5py.File(path, "r")
    ids = list(hf.keys())
    random_ids = np.random.choice(ids, num_events)
    return hf, random_ids


def get_model(sws: int):
    raw_model = RawFeatureModel(sws=sws)
    fft_model = FFTFeatureModel(sws=sws)
    cwt_model = CWTFeatureModel()
    model = MultiFeaturesModel(raw_model, fft_model, cwt_model)

    return model


def get_cwt(signal: np.ndarray):
    max_scale = 1024
    num = int(np.log2(max_scale)) + 1
    widths = np.round(
        np.geomspace(1, max_scale, num=num, endpoint=True)
    ).astype(int)
    signal_cwt, _ = pywt.cwt(signal, scales=widths, wavelet="morl", axis=0)
    signal_cwt = np.transpose(signal_cwt, (1, 0, 2))
    return signal_cwt


def predict(
    sws: int, la: int, device: torch.device, file_obj: h5py.File, ids: np.ndarray
):
    elm_predictions = dict()
    for i_elm, elm_event in enumerate(ids):
        signal = np.array(file_obj[elm_event]["signals"])
        signal = signal.T
        signal[:, :32] = signal[:, :32] / np.max(signal[:, :32])
        signal[:, 32:] = signal[:, 32:] / np.max(signal[:, 32:])
        time = np.array(file_obj[elm_event]["time"])
        print(
            f"Processing {i_elm + 1} of {len(ids)} elm events with id {elm_event} and length {time.shape[0]}"
        )
        # signal_cwt = get_cwt(signal)

        predictions = []
        effective_len = time.shape[0] - sws - la + 1
        for j in range(effective_len):
            input_signals = np.array(
                signal[j : j + sws],
                dtype=np.float32,
            )
            if j % 1000 == 0:
                print(f'Calculating CWT for index {j+1} of {effective_len}')
            input_signals_cwt = get_cwt(input_signals)
            input_signals = input_signals.reshape([1, 1, sws, 8, 8])
            # input_signals_cwt = np.array(
            #     signal_cwt[j : j + sws].reshape(
            #         [
            #             1,
            #             1,
            #             sws,
            #             11,
            #             8,
            #             8,
            #         ]
            #     ),
            #     dtype=np.float32,
            # )
            input_signals_cwt = input_signals_cwt.reshape([1, 1, sws, 11, 8, 8])
            input_signals = torch.as_tensor(input_signals, dtype=torch.float32)
            input_signals_cwt = torch.as_tensor(
                input_signals_cwt, dtype=torch.float32
            )
            input_signals = input_signals.to(device)
            input_signals_cwt = input_signals_cwt.to(device)
            outputs = model(input_signals, input_signals_cwt)
            predictions.append(outputs.item())
        predictions = np.array(predictions)
        micro_predictions = (
            torch.sigmoid(torch.as_tensor(predictions, dtype=torch.float32))
            .cpu()
            .numpy()
        )
        micro_predictions = np.pad(
            micro_predictions,
            pad_width=(
                sws + la - 1,
                0,
            ),
            mode="constant",
            constant_values=0,
        )
        print(f"Signals shape: {signal.shape}")
        # print(f"Signals CWT shape: {signal_cwt.shape}")
        print(f"Time shape: {time.shape[0]}")
        print(f"Micro predictions shape: {micro_predictions.shape}")
        elm_predictions[elm_event] = {
            "signals": signal,
            "micro_predictions": micro_predictions,
        }
    return elm_predictions


def plot(
    sws: int,
    la: int,
    elm_predictions: dict,
    plot_dir: str,
    elms: List[int],
    elm_range: str,
    n_rows: Union[int, None] = None,
    n_cols: Union[int, None] = None,
    figsize: tuple = (14, 12),
    dry_run: bool = True,
    show_plots: bool = True,
) -> None:
    fig = plt.figure(figsize=figsize)
    for i, elm_event in enumerate(elms):
        signals = elm_predictions[elm_event]["signals"]
        predictions = elm_predictions[elm_event]["micro_predictions"]
        print(f"ELM {i + 1} of {len(elms)} with {len(signals)} time points")
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(
            signals[:, 0] / np.max(signals),
            label="Ch. 1",
            lw=0.85,
        )
        plt.plot(
            signals[:, 21] / np.max(signals),
            label="Ch. 22",
            lw=0.85,
        )
        plt.plot(
            signals[:, 63] / np.max(signals),
            label="Ch. 64",
            lw=0.85,
        )
        plt.plot(predictions, label="Prediction", ls="-", lw=1.25, alpha=0.75)
        plt.title(f"ELM ID: {elm_event}", fontsize=12)
        plt.xlabel("Time (micro-s)", fontsize=10)
        plt.ylabel("Signal | label", fontsize=10)
        plt.tick_params(axis="x", labelsize=8)
        plt.tick_params(axis="y", labelsize=8)
        plt.ylim([None, 1.1])
        sns.despine(offset=10, trim=False)
        plt.legend(fontsize=6, ncol=2, frameon=False)
        plt.gca().spines["left"].set_color("lightgrey")
        plt.gca().spines["bottom"].set_color("lightgrey")
        plt.grid(axis="y")
    plt.suptitle(
        f"ELM index: {elm_range}, Signal window size: {sws}, Lookahead: {la}",
        fontsize=20,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not dry_run:
        fig.savefig(
            os.path.join(
                plot_dir,
                f"testing_unlabeled_sws_{sws}_la_{la}_time_series_{elm_range}.png",
            ),
            dpi=200,
        )
    if show_plots:
        plt.show()


def plot_all(
    sws: int,
    la: int,
    elm_predictions: dict,
    plot_dir: str,
    dry_run: bool = True,
    show_plots: bool = True,
) -> None:
    elm_id = list(elm_predictions.keys())
    for i in range(1):
        i_elms = elm_id[i * 12 : (i + 1) * 12]
        i_elm_predictions = {
            k: v for k, v in elm_predictions.items() if k in i_elms
        }
        plot(
            sws,
            la,
            i_elm_predictions,
            plot_dir,
            i_elms,
            elm_range=str((i * 12) + 1) + "-" + str((i + 1) * 12),
            n_rows=4,
            n_cols=3,
            dry_run=dry_run,
            show_plots=show_plots,
        )
    # i_elms_1_12 = elm_id[:12]
    # i_elms_12_24 = elm_id[12:24]
    # i_elms_24_36 = elm_id[24:36]
    # i_elms_36_48 = elm_id[36:48]
    # i_elms_48_60 = elm_id[48:60]
    # i_elms_60_64 = elm_id[60:64]

    # plot 1-12
    # plot(
    #     elm_predictions,
    #     plot_dir,
    #     i_elms_1_12,
    #     elm_range="1-12",
    #     n_rows=4,
    #     n_cols=3,
    #     dry_run=dry_run,
    # )
    # # plot 12-24
    # plot(
    #     elm_predictions,
    #     plot_dir,
    #     i_elms_12_24,
    #     elm_range="12-24",
    #     n_rows=4,
    #     n_cols=3,
    #     dry_run=dry_run,
    # )
    # # plot 24-36
    # plot(
    #     elm_predictions,
    #     plot_dir,
    #     i_elms_24_36,
    #     elm_range="24-36",
    #     n_rows=4,
    #     n_cols=3,
    #     dry_run=dry_run,
    # )
    # # plot 36-48
    # plot(
    #     elm_predictions,
    #     plot_dir,
    #     i_elms_36_48,
    #     elm_range="36-48",
    #     n_rows=4,
    #     n_cols=3,
    #     dry_run=dry_run,
    # )
    # # plot 48-60
    # plot(
    #     elm_predictions,
    #     plot_dir,
    #     i_elms_48_60,
    #     elm_range="48-60",
    #     n_rows=4,
    #     n_cols=3,
    #     dry_run=dry_run,
    # )
    # # plot 60-66
    # plot(
    #     elm_predictions,
    #     plot_dir,
    #     i_elms_60_64,
    #     elm_range="60-64",
    #     n_rows=2,
    #     n_cols=2,
    #     figsize=(10, 6),
    #     dry_run=dry_run,
    # )


if __name__ == "__main__":
    # signal window size and label lookahead
    sws = 512
    la = 0
    num_unlabeled_events = 12

    # define paths for model checkpoint and unlabeled data
    base_path = os.path.dirname(os.getcwd())
    model_ckpt_path = os.path.join(
        base_path,
        f"elm_classification/model_checkpoints/signal_window_{sws}/multi_features_lookahead_{la}_unprocessed.pth",
    )

    unlabeled_data_path = "step_5_unlabeled_elm_events.hdf5"

    # instantiate model
    model = get_model(sws)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # load saved model from checkpoint
    model.load_state_dict(
        torch.load(model_ckpt_path, map_location=device)["model"]
    )

    # get unlabeled data
    file_obj, ids = get_unlabeled_data(
        unlabeled_data_path, num_events=num_unlabeled_events
    )

    # make predictions
    predictions = predict(sws, la, device, file_obj, ids)
    file_obj.close()

    # plot
    plot_all(
        sws,
        la,
        predictions,
        plot_dir=os.getcwd(),
        dry_run=True,
        show_plots=True,
    )
