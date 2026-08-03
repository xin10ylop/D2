"""Bot configuration. Everything risk-bearing is explicit and conservative."""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- mode -----------------------------------------------------------
    # Live trading requires BOTH: mode="live" AND KALSHI_LIVE=yes in the env.
    # Two independent switches, because one is too easy to flip by accident.
    mode: str = os.getenv("BOT_MODE", "paper")          # paper | live
    assets: tuple = ("BTC", "ETH")
    series: tuple = ("KXBTCD", "KXETHD")                # digital ladders
    poll_secs: float = 2.0

    # ---- edge thresholds (dollars per $1 contract) ----------------------
    # Take only when the book is wrong by more than this AFTER fees. The
    # clock mis-pricing runs 2-6c at the extremes, so 2c keeps us in the
    # part of the distribution the model is actually confident about.
    take_edge: float = 0.025
    # Quote this far from fair on each side. Must exceed the round-trip fee
    # plus an adverse-selection allowance.
    quote_margin: float = 0.015
    max_spread_to_quote: float = 0.12   # ignore books wider than this (stale)

    # ---- fees -----------------------------------------------------------
    kalshi_taker_mult: float = 0.07     # ceil(0.07*C*P*(1-P))
    kalshi_maker_fee: float = 0.0025    # per contract; 0 on some series
    deribit_perp_taker: float = 0.0005  # fraction of notional

    # ---- position limits ------------------------------------------------
    max_contracts_per_strike: int = 250
    max_contracts_per_event: int = 2000
    max_gross_contracts: int = 8000
    max_net_delta_usd: float = 15_000.0   # hedged above this
    max_notional_usd: float = 25_000.0
    daily_loss_limit_usd: float = 1_500.0

    # ---- model sanity gates --------------------------------------------
    # If the model disagrees with the market by more than this, assume the
    # market knows something we do not and stand down on that strike.
    max_model_disagreement: float = 0.08
    # Below the front Deribit expiry the smile is extrapolated, not observed,
    # and that is precisely where the model was worst. Require the settlement
    # to sit at least this fraction of the way to the front anchor.
    min_frac_of_front_anchor: float = 0.55
    min_seconds_to_settle: float = 3600.0   # no new risk inside an hour
    max_seconds_to_settle: float = 36 * 3600.0
    max_data_age_secs: float = 20.0         # stale data => flatten & stop

    # ---- hedging --------------------------------------------------------
    hedge_enabled: bool = True
    hedge_instrument: dict = field(default_factory=lambda: {
        "BTC": "BTC-PERPETUAL", "ETH": "ETH-PERPETUAL"})
    hedge_band_usd: float = 4_000.0     # only hedge outside this dead-band

    @property
    def live(self):
        return self.mode == "live" and os.getenv("KALSHI_LIVE") == "yes"


CFG = Config()
