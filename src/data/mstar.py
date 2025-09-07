import mstarpy as ms


import functools


@functools.lru_cache(maxsize=128)  # % remnember last 128 calls
def get_fund_snap(symbol: str) -> dict:
    itemRange = 0
    try:
        while itemRange <= 10:
            fund = ms.Funds(
                symbol,
                # filters={"domicile": "USA"},
                page=1,
                pageSize=10,
                itemRange=itemRange,
                sortby="alpha",
            )
            snap = fund.snapshot()
            if snap.get("Symbol") == symbol:
                break
            itemRange += 1

    except Exception as e:
        try:
            fund = ms.Funds(
                symbol,
                # filters={"domicile": "USA"},
                page=1,
                pageSize=10,
                itemRange=0,
                sortby="alpha",
            )
            snap = fund.snapshot()
            if snap.get("Symbol") == symbol:
                return snap
            else:
                raise e
        except Exception as e:
            print(e)
            return None
    return snap


def get_number_of_holdings(snap: dict) -> int:
    return snap.get("Portfolios")[0].get("HoldingAggregates")[0].get("NumberOfHolding")
