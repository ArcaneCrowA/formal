# Formal Verification

Information how it works in docs

## How to install

I recommend using [uv](https://docs.astral.sh/uv/)

### Using uv

```bash
uv sync --locked
```

## How to run 

Could use cli by running 

### Using uv

```bash
uv run main.py help
```


## How to change parameters and dataset

They could be changed in [config](./config.py)



======================================================================
Depth  Fair   Robust  Verification (s)   Inference Mean (ms)  Total Inference (s)
----------------------------------------------------------------------
1      Yes    Yes     0.0331             0.0365               0.2206
2      Yes    No      0.0042             3.7034               22.3440
3      Yes    No      0.0018             4.9965               30.1452
4      Yes    No      0.0026             7.4930               45.2063
5      Yes    No      0.0041             11.0517              66.6758
6      Yes    No      0.0071             18.0761              109.0540
7      Yes    No      0.0115             29.1141              175.6465
8      No     No      0.0223             45.5468              274.7853
9      No     No      0.0316             66.7521              402.7173
10     No     No      0.0471             98.3084              593.0958
11     No     No      0.0710             139.8901             843.9585
======================================================================
⏎
