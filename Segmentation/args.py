import argparse

class Args(object):
    parser = argparse.ArgumentParser(description='Arguments for Segmentation models')
    parser.add_argument('--model', default='unet')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--scheduler_patience', type=int, default=4)
    parser.add_argument('--earlystopping_patience', type=int, default=7)


    parse = parser.parse_args()
    params = {
        "MODEL": parse.model, 
        "DEVICE": parse.device,
        "BATCH_SIZE": parse.batch_size,
        "LEARNING_RATE": parse.learning_rate,
        "NUM_EPOCHS": parse.num_epochs,
        "RANDOM_SEED": parse.random_seed,
        "SCHEDULDER_PATIENCE": parse.scheduler_patience,
        "EARLYSTOPPING_PATIENCE": parse.earlystopping_patience
    }