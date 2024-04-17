This repository is based on `CosmoTransitions` developed by Carroll L. Wainwright [arXiv paper](https://arxiv.org/pdf/1109.4189.pdf), [github tutorial](https://clwainwright.github.io/CosmoTransitions/).

Before first run, you need to install the required packages by
```
pip install -r requirements.txt
```
Before each run, setup the environment by
```
. setup.sh
```

There are two copies of codes implementing effective potential with and without CW correction, which can be distinguished from the imported base model script:

Without CW: ``import baseMo_s_b_d as bm``

With CW and counterterms: ``import baseMo_s_b_cwd as bmcwd``  
