from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils import metrics


def eval_cross_entropy_loss(model, device, loader, sigma=1.0):
    model.eval()
    with torch.set_grad_enabled(False):
        total_cost = 0
        total_pairs = loader.get_num_pairs()
        pairs_in_compute = 0
        for X, Y in loader.generate_batch_per_query(loader.df):
            Y = Y.reshape(-1, 1)
            rel_diff = Y - Y.T
            pos_pairs = (rel_diff > 0).astype(np.float32)
            num_pos_pairs = np.sum(pos_pairs, (0, 1))
            # skip negative sessions, no relevant info:
            if num_pos_pairs == 0:
                continue
            neg_pairs = (rel_diff < 0).astype(np.float32)
            num_pairs = 2 * num_pos_pairs  # num pos pairs and neg pairs are always the same
            pos_pairs = torch.tensor(pos_pairs, device=device)
            neg_pairs = torch.tensor(neg_pairs, device=device)
            Sij = pos_pairs - neg_pairs
            # only calculate the different pairs
            diff_pairs = pos_pairs + neg_pairs
            pairs_in_compute += num_pairs

            X_tensor = torch.Tensor(X).to(device)
            y_pred = model(X_tensor)
            y_pred_diff = y_pred - y_pred.t()

            # logsigmoid(x) = log(1 / (1 + exp(-x))) equivalent to log(1 + exp(-x))
            C = 0.5 * (1 - Sij) * sigma * y_pred_diff - F.logsigmoid(-sigma * y_pred_diff)
            C = C * diff_pairs
            cost = torch.sum(C, (0, 1))
            total_cost += cost

        assert total_pairs == pairs_in_compute
        avg_cost = total_cost / total_pairs
    return avg_cost.item()


def eval_ndcg_at_k(inference_model, device, df_valid, valid_loader, k_list):
    ndcg_metrics = {k: metrics.NDCG(k) for k in k_list}
    qids, rels, scores = [], [], []
    inference_model.eval()
    with torch.no_grad():
        for qid, rel, x in valid_loader.generate_query_batch(df_valid):
            if x is None or x.shape[0] == 0:
                continue
            y_tensor = inference_model.forward(torch.Tensor(x).to(device))
            scores.append(y_tensor.cpu().numpy().squeeze())
            qids.append(qid)
            rels.append(rel)

    qids = np.hstack(qids)
    rels = np.hstack(rels)
    scores = np.hstack(scores)
    result_df = pd.DataFrame({'qid': qids, 'rel': rels, 'score': scores})
    session_ndcgs = defaultdict(list)
    for qid in result_df.qid.unique():
        result_qid = result_df[result_df.qid == qid].sort_values('score', ascending=False)
        rel_rank = result_qid.rel.values
        for k, ndcg in ndcg_metrics.items():
            if ndcg.maxDCG(rel_rank) == 0:
                continue
            ndcg_k = ndcg.evaluate(rel_rank)
            if not np.isnan(ndcg_k):
                session_ndcgs[k].append(ndcg_k)

    ndcg_result = {k: np.mean(session_ndcgs[k]) for k in k_list}
    return ndcg_result
