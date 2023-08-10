# DDi-OCF
we can not release DrugBank data because it's licensed.

paper : Predicting Drug-Drug Interactions: A Deep Learning Approach with GCN-based Collaborative Filtering

Execution example

python main.py --dataset="ocr-ddi" --model="ddiocf" --solver="rk4" --adjoint=False --learnable_time=False --dual_res=False --K=4 --lr=1e-3 --decay=1e-4 --topks="[10, 20, 100, 200]" --comment="drug only" --tensorboard=1 --gpuid=0 --bpr_batch=800 --testbatch=200 --epochs=200


