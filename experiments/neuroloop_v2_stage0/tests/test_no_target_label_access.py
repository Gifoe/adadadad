import inspect

from eval.evaluate_loop_trajectory import predict_loop_trajectory
from train.train_neuroloop_v2 import select_then_refit


def test_training_and_label_free_trajectory_interfaces():
    train_parameters = set(inspect.signature(select_then_refit).parameters)
    assert "target" not in " ".join(train_parameters).lower()
    trajectory_parameters = set(inspect.signature(predict_loop_trajectory).parameters)
    assert trajectory_parameters == {"model", "x", "device"}


if __name__ == "__main__":
    test_training_and_label_free_trajectory_interfaces()
    print("PASS test_training_and_label_free_trajectory_interfaces")
