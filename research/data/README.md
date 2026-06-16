# research/data/ — drop your exported data here

## unlock_dates.csv  (for token_unlock_confirm.py)
Export the historical token-unlock schedule from DefiLlama (Sheets/CSV) and save it
here as `unlock_dates.csv` with columns:

    cg_id,date
    arbitrum,2024-03-16
    optimism,2025-01-31
    aptos,2025-02-12
    ...

- `cg_id` = the CoinGecko id (must match the ids in research/token_unlock_full.py:
  arbitrum, optimism, aptos, sui, celestia, sei-network, immutable-x, starknet,
  worldcoin-wld, pyth-network, jupiter-exchange-solana, wormhole, ethena,
  jito-governance-token, dydx-chain, ...). Map DefiLlama protocol → CoinGecko id.
- `date` = unlock date, YYYY-MM-DD. One row per unlock event.

Only dates within the last ~365 days will match the free cached prices
(/tmp/cg_unlocks). That overlap is enough for the decisive artifact check:
does the −1.6% unlock-day effect hold on REAL scheduled dates as it did on
supply-jump-detected dates?

(Secrets/API keys go in the repo-root `.env`, which is gitignored — never here.)
