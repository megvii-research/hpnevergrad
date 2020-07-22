#!/usr/bin/env python3
from tqdm import tqdm
import functools
import numpy as np
import argparse
import torch
import yaml
from torch import optim
import os

from hpman.m import _
import hpargparse
from hpnevergrad import hpng
import nevergrad as ng

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def run():
    parser = argparse.ArgumentParser()
    #_.parse_file(BASE_DIR)
    hpargparse.bind(parser, _)
    parser.parse_args()  # we need not to use args

    # print all hyperparameters
    print("-" * 10 + " Hyperparameters " + "-" * 10)
    print(yaml.dump(_.get_values()))

    optimizer_cls = {
        "adam": optim.Adam,
        "sgd": functools.partial(optim.SGD, momentum=0.9),
    }[_("optimizer", "adam")  # <-- hyperparameter
      ]

    import model

    net = model.get_model()
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        net.cuda()

    optimizer = optimizer_cls(
        net.parameters(),
        lr=_("learning_rate", 1e-3, choices=[1e-4,
                                             1e-3]),  # <-- hyperparameter
        weight_decay=_("weight_decay", 1e-5,
                       choices=[1e-5, 1e-4, 5e-4]),  # <-- hyperparameter
    )

    import dataset

    train_ds = dataset.get_data_and_labels("train")
    test_ds = dataset.get_data_and_labels("test")
    if torch.cuda.is_available():
        # since mnist is a small dataset, we store the test dataset all in the
        # gpu memory
        test_ds = {k: v.cuda() for k, v in test_ds.items()}

    rng = np.random.RandomState(_("seed", 42))  # <-- hyperparameter

    for epoch in range(_("num_epochs", 30,
                         choices=[5, 15, 30])):  # <-- hyperparameter
        net.train()
        tq = tqdm(
            enumerate(
                dataset.iter_dataset_batch(
                    rng,
                    train_ds,
                    _("batch_size", 256, choices=[256,
                                                  1024]),  # <-- hyperparameter
                    cuda=torch.cuda.is_available(),
                )))
        for step, minibatch in tq:
            optimizer.zero_grad()

            Y_pred = net(minibatch["data"])
            loss = model.compute_loss(Y_pred, minibatch["labels"])

            loss.backward()
            optimizer.step()

            metrics = model.compute_metrics(Y_pred, minibatch["labels"])
            metrics["loss"] = loss.detach().cpu().numpy()
            tq.desc = "e:{} s:{} {}".format(
                epoch,
                step,
                " ".join([
                    "{}:{}".format(k, v) for k, v in sorted(metrics.items())
                ]),
            )

        net.eval()

        # since mnist is a small dataset, we predict all values at once.
        Y_pred = net(test_ds["data"])
        metrics = model.compute_metrics(Y_pred, test_ds["labels"])
        print("eval: {}".format(" ".join(
            ["{}:{}".format(k, v) for k, v in sorted(metrics.items())])))

        # Save the model. We intentionally not saving the model here for
        # tidiness of the example
        # torch.save(net, "model.pt")
        return float(metrics['misclassify'])


if __name__ == "__main__":
    _.parse_file(BASE_DIR)
    # define hyperparameters in nevergrad parametrization type
    parametrization = hpng.get_parametrization(_)
    # load hyperparameter in hpman to objective_function: a warpper
    objective_function = hpng.get_objective_function(run, _)
    optimizer = ng.optimizers.NGO(parametrization=parametrization, budget=10)
    recommendation = optimizer.minimize(objective_function)

    print(recommendation.kwargs)
