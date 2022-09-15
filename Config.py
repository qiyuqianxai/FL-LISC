
class Config():
    Stu_model_name = "resnet18"
    dataset = "mnist"
    n_clients = 20
    n_strong_clients = 5
    n_weak_clients = n_clients - n_strong_clients
    data_diribution_balancedness_for_clents = False
    batch_size_for_clients = 32
    communication_rounds = 30
    epochs_for_clients = 15
    SNR = 15
    SNR_MAX = 25
    SNR_MIN = 0
    use_Rali = False
    use_RTN = False
    isc_lr = 1e-3
    channel_lr = 1e-3
    weight_delay = 1e-5
    device = "cpu"
    checkpoints_dir = "checkpoints"
    logs_dir = "logs"