import numpy as np
import torch
from sionna.phy.channel import ChannelModel, OFDMChannel
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.mapping import Demapper, Mapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    LMMSEEqualizer,
    LSChannelEstimator,
    ResourceGrid,
    ResourceGridDemapper,
    ResourceGridMapper,
)
from sionna.phy.utils import ebnodb2no


class IdentityChannelModel(ChannelModel):
    def __call__(
        self,
        batch_size: int,
        num_time_steps: int,
        sampling_frequency: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del sampling_frequency
        path_coefficients = torch.ones(
            batch_size,
            1,
            1,
            1,
            1,
            1,
            num_time_steps,
            dtype=self.cdtype,
            device=self.device,
        )
        path_delays = torch.zeros(
            batch_size,
            1,
            1,
            1,
            dtype=self.dtype,
            device=self.device,
        )

        return path_coefficients, path_delays


class DigitalTextOFDMLink:
    def __init__(
        self,
        k: int,
        n: int,
        num_bits_per_symbol: int,
        fft_size: int,
        num_ofdm_symbols: int,
        subcarrier_spacing: float,
        cyclic_prefix_length: int,
        pilot_pattern: str,
        pilot_ofdm_symbol_indices: list[int] | None,
        dc_null: bool,
        num_guard_carriers: tuple[int, int],
        channel_model,
        normalize_channel: bool,
        device: torch.device,
    ) -> None:
        self.device = device
        device_str = str(device)
        self.num_bits_per_symbol = num_bits_per_symbol
        self.encoder = LDPC5GEncoder(k=k, n=n, device=device_str)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True, device=device_str)

        self.mapper = Mapper(
            "qam",
            num_bits_per_symbol=self.num_bits_per_symbol,
            device=device_str,
        )
        self.demapper = Demapper(
            "app",
            "qam",
            num_bits_per_symbol=num_bits_per_symbol,
            device=device_str,
        )

        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=cyclic_prefix_length,
            pilot_pattern=pilot_pattern,
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
            dc_null=dc_null,
            num_guard_carriers=num_guard_carriers,
            device=device_str,
        )
        self.effective_subcarrier_ind = torch.as_tensor(
            self.resource_grid.effective_subcarrier_ind,
            dtype=torch.long,
            device=device,
        )
        self.stream_management = StreamManagement(
            np.array([[True]], dtype=bool),
            num_streams_per_tx=1,
        )

        self.rg_mapper = ResourceGridMapper(self.resource_grid, device=device_str)
        self.rg_demapper = ResourceGridDemapper(
            self.resource_grid,
            self.stream_management,
            device=device_str,
        )

        self.ofdm_channel = OFDMChannel(
            channel_model=channel_model,
            resource_grid=self.resource_grid,
            normalize_channel=normalize_channel,
            return_channel=True,
            device=device_str,
        )

        self.channel_estimator = LSChannelEstimator(
            self.resource_grid,
            device=device_str,
        )
        self.equalizer = LMMSEEqualizer(
            self.resource_grid,
            self.stream_management,
            device=device_str,
        )

    def transmit(
        self,
        bits: torch.Tensor,
        ebno_db: float,
        perfect_csi: bool = False,
    ) -> torch.Tensor:
        coded_bits = self.encoder(bits)
        tx_symbols = self.mapper(coded_bits)
        tx_grid = self.rg_mapper(tx_symbols.unsqueeze(1).unsqueeze(1))

        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.num_bits_per_symbol,
            coderate=bits.shape[-1] / coded_bits.shape[-1],
            resource_grid=self.resource_grid,
        )

        rx_grid, h_freq = self.ofdm_channel(tx_grid, no)

        if perfect_csi:
            h_hat = torch.index_select(h_freq, -1, self.effective_subcarrier_ind)
            err_var = torch.zeros_like(h_hat.real)
        else:
            h_hat, err_var = self.channel_estimator(rx_grid, no)

        rx_symbols, no_eff = self.equalizer(rx_grid, h_hat, err_var, no)
        llr = self.demapper(
            rx_symbols.squeeze(1).squeeze(1),
            no_eff.squeeze(1).squeeze(1),
        )
        decoded_bits = self.decoder(llr)

        return decoded_bits
