import torch


def available_gpus():
    available_gpus = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]
    return available_gpus


def current_device():
    return torch.cuda.current_device()


def info_gpus():
    print(f"{current_device()=}")
    print(f"{available_gpus()=}")


def empty_gpu():
    torch.cuda.empty_cache()
