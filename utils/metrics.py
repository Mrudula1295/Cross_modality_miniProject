import numpy as np
import torch


def evaluate_cross_modal(
    query_feats: torch.Tensor,
    query_pids: torch.Tensor,
    query_camids: torch.Tensor,
    gallery_feats: torch.Tensor,
    gallery_pids: torch.Tensor,
    gallery_camids: torch.Tensor,
    max_rank: int = 20
) -> dict:
    """
    Computes Cumulative Matching Characteristics (CMC) and mean Average Precision (mAP)
    for Cross-Modality Person Re-Identification evaluation protocol.
    
    Protocol Rule: Matches with same Person ID AND same Camera ID are excluded as junk.
    """
    query_feats = query_feats.cpu().numpy() if isinstance(query_feats, torch.Tensor) else query_feats
    query_pids = query_pids.cpu().numpy() if isinstance(query_pids, torch.Tensor) else query_pids
    query_camids = query_camids.cpu().numpy() if isinstance(query_camids, torch.Tensor) else query_camids

    gallery_feats = gallery_feats.cpu().numpy() if isinstance(gallery_feats, torch.Tensor) else gallery_feats
    gallery_pids = gallery_pids.cpu().numpy() if isinstance(gallery_pids, torch.Tensor) else gallery_pids
    gallery_camids = gallery_camids.cpu().numpy() if isinstance(gallery_camids, torch.Tensor) else gallery_camids

    num_q, num_g = len(query_pids), len(gallery_pids)
    
    # Compute Cosine Distance Matrix (1 - Cosine Similarity for L2-normalized embeddings)
    distmat = 1.0 - np.dot(query_feats, gallery_feats.T)

    all_cmc = []
    all_ap = []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = query_pids[q_idx]
        q_cam = query_camids[q_idx]

        # Identify gallery indices with same PID and same Camera (junk)
        junk_mask = (gallery_pids == q_pid) & (gallery_camids == q_cam)
        good_mask = (gallery_pids == q_pid) & (gallery_camids != q_cam)

        if not np.any(good_mask):
            # Skip query if no valid gallery images exist for this identity in different camera
            continue

        # Rank gallery items by ascending distance
        order = np.argsort(distmat[q_idx])
        
        # Remove junk elements from ranked list
        remove_mask = junk_mask[order]
        order = order[~remove_mask]
        
        # Binary match vector (1 for true identity match, 0 otherwise)
        matches = (gallery_pids[order] == q_pid).astype(np.int32)
        
        if not np.any(matches):
            continue

        # CMC Computation
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        
        # mAP Computation
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        precisions = tmp_cmc / (np.arange(len(matches)) + 1.0)
        ap = (precisions * matches).sum() / num_rel
        all_ap.append(ap)
        num_valid_q += 1

    if num_valid_q == 0:
        return {"rank1": 0.0, "rank5": 0.0, "rank10": 0.0, "mAP": 0.0}

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(axis=0) / num_valid_q

    rank1 = float(all_cmc[0]) * 100.0
    rank5 = float(all_cmc[4]) * 100.0 if max_rank >= 5 else 0.0
    rank10 = float(all_cmc[9]) * 100.0 if max_rank >= 10 else 0.0
    mAP = float(np.mean(all_ap)) * 100.0

    return {
        "rank1": rank1,
        "rank5": rank5,
        "rank10": rank10,
        "mAP": mAP,
        "valid_queries": num_valid_q
    }


if __name__ == "__main__":
    q_feats = np.random.randn(10, 512)
    q_feats /= np.linalg.norm(q_feats, axis=1, keepdims=True)
    g_feats = np.random.randn(50, 512)
    g_feats /= np.linalg.norm(g_feats, axis=1, keepdims=True)
    
    q_pids = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    q_cams = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    g_pids = np.array([1, 2, 3, 4, 5] * 10)
    g_cams = np.array([1, 2, 1, 2, 1] * 10)
    
    metrics = evaluate_cross_modal(q_feats, q_pids, q_cams, g_feats, g_pids, g_cams)
    print("Dummy Evaluation Metrics:", metrics)
