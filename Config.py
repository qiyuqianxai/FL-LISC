
class Config():
    Stu_model_name = "resnet18"
    hp = {"dataset":"mnist",
            "n_clients":20,
            "classes_per_client":10,
            "balancedness":2,
            "batch_size_for_client":8}
    SNR = 25
    use_Rali = False
    use_RTN = False
    communication_rounds = 30
    local_epochs = 50
    isc_lr = 1e-5
    channel_lr = 1e-3
    weight_delay = 1e-5
    device = "cpu"
    checkpoints_dir = "checkpoints"
    logs_dir = "logs"