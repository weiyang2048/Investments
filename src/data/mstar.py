import mstarpy as ms


import functools


@functools.lru_cache(maxsize=128)  # % remnember last 128 calls
def get_fund_snap(symbol: str) -> dict:
    itemRange = 0
    while itemRange <= 10:
        fund = ms.Funds(
            symbol,
            filters={"domicile": "USA"},
            page=1,
            pageSize=10,
            itemRange=itemRange,
            sortby="alpha",
        )
        snap = fund.snapshot()
        if snap.get("Symbol") == symbol:
            break
        itemRange += 1
    return snap
