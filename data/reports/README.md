# Batch logbook

Batches are 0-indexed. A prediction of -1 indicates a prediction that was flagged by the early prediction model as unreliable.

- oed_0: Started on 2018-08-28. Two cells failed on start (ch 17 & 27), one cell had anomalous prediction (ch 7). 45 successful cells
- oed_1: Started on 2018-09-02. Two cells failed mid-run (ch 4 & 5). 46 successful cells
- oed_2: Started on 2018-09-06. One cell failed due to anomalous prediction (ch 12). 47 successful cells
- oed_3: Started on 2018-09-10. One cell failed due to anomalous prediction (ch 6). 47 successful cells

Note: Cells often run past cycle 100 (max 120). However, data up to only the 100th cycle is used in the prediction. Due to cell-to-cell variation during the constant-voltage hold steps, not all cells finish at exactly the same time. We cycle the cells past cycle 100 to maintain a uniform temperature for the last cells to reach cycle 100 within a batch.