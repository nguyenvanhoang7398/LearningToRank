from dataset.mslr import MSLR
import numpy as np
import os
from ranking.rank_net.model import RankNet, RankNetPairs
import torch
from torch.utils.tensorboard import SummaryWriter
import utils
from utils import eval_utils, ml_utils


class MslrConfig(object):
    train_file = "data/mslr-web10k/train.small.txt"
    dev_file = "data/mslr-web10k/vali.small.txt"
    test_file = "data/mslr-web10k/vali.small.txt"
    batch_size = 512
    task_name = "mslr"


class TrainingConfig(object):
    num_epochs = 10
    lr = 0.0001
    step_size = 10
    gamma = 0.75
    eval_and_save_every = 5
    ndcg_k_list = [10, 30]
    log_dir = "exp_log"


def run_mslr(model_name, mslr_config=MslrConfig(), training_config=TrainingConfig()):
    exp_name = utils.get_exp_name(model_name, mslr_config.task_name)
    writer = SummaryWriter(os.path.join(training_config.log_dir, exp_name))

    mslr = MSLR(mslr_config.train_file, mslr_config.dev_file, mslr_config.test_file, mslr_config.batch_size)
    train_loader, train_df, dev_loader, dev_df, test_loader, test_df = mslr.load_data()
    model, model_inference = get_train_inference_model(model_name, train_loader.num_features)
    device = ml_utils.get_device()
    model.to(device)
    model_inference.to(device)
    model.apply(ml_utils.init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_config.step_size,
                                                gamma=training_config.gamma)
    loss_op = torch.nn.BCELoss().to(device)
    losses = []

    print("Start training")
    for epoch in range(training_config.num_epochs):

        scheduler.step()
        model.zero_grad()
        model.train()

        train_ce_loss = pairwise_train_fn(model, loss_op, optimizer, train_loader, device)
        print('Finish training for epoch {}'.format(epoch))
        train_results = {
            "loss": train_ce_loss
        }
        writer.add_scalars("train", train_results, epoch)
        print(train_results)
        losses.append(train_ce_loss)

        # save to checkpoint every 5 step, and run eval
        if epoch % training_config.eval_and_save_every == 0:
            dev_ce_loss, dev_ndcg_results = eval_model_fn(model_inference, device, dev_df, dev_loader,
                                                          training_config.ndcg_k_list)
            eval_results = {
                "loss": dev_ce_loss
            }
            print("Validation at epoch {}".format(epoch))
            for k in dev_ndcg_results:
                ndcg_at_str = "NDCG@{}".format(k)
                eval_results[ndcg_at_str] = dev_ndcg_results[k]
            print(eval_results)
            writer.add_scalars("eval", eval_results, epoch)

    print("Training finished, start testing...")
    test_ce_loss, test_ndcg_results = eval_model_fn(model_inference, device, test_df, test_loader,
                                                    training_config.ndcg_k_list)
    print("Testing loss: {}".format(test_ce_loss))
    for k in test_ndcg_results:
        print("NDCG@{}: {:.5f}".format(k, test_ndcg_results[k]))


def eval_model_fn(model_inference, device, dev_df, dev_loader, ndcg_k_list):
    model_inference.eval()  # Set model to evaluate mode

    with torch.no_grad():
        cross_entropy_loss = eval_utils.eval_cross_entropy_loss(model_inference, device, dev_loader)
        ndcg_result = eval_utils.eval_ndcg_at_k(model_inference, device, dev_df, dev_loader, ndcg_k_list)
    return cross_entropy_loss, ndcg_result


def pairwise_train_fn(model, loss_func, optimizer, train_loader, device):
    minibatch_loss = []

    for x_i, y_i, x_j, y_j in train_loader.generate_query_pair_batch(None):
        if x_i is None or x_i.shape[0] == 0:
            continue
        x_i, x_j = torch.tensor(x_i, device=device), torch.tensor(x_j, device=device)
        # binary label
        y = torch.tensor((y_i > y_j).astype(np.float32), device=device)

        model.zero_grad()

        y_pred = model(x_i, x_j)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()
        minibatch_loss.append(loss.item())

        # for name, param in list(filter(lambda p: p[1].grad is not None, model.named_parameters())):
        #     print("Back-prop grad", name, param.grad.data.norm(2).item())

    return np.mean(minibatch_loss)


def get_train_inference_model(model_name, num_features):
    ranknet_structure = [num_features, 64, 16]

    if model_name == "rank_net":
        model = RankNetPairs(ranknet_structure)
        model_inference = RankNet(ranknet_structure)  # inference always use single precision
    else:
        raise ValueError("Model name {} is not supported".format(model_name))

    return model, model_inference
