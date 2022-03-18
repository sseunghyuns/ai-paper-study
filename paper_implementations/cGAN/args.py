import argparse

class Args(object):
    parser = argparse.ArgumentParser(description='Arguments for Training Conditional GAN')
    parser.add_argument('--model', default='cgan')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--random_seed', type=int, default=42)


    parse = parser.parse_args()
    params = {
        "MODEL": parse.model, 
        "DEVICE": parse.device,
        "BATCH_SIZE": parse.batch_size,
        "LEARNING_RATE": parse.learning_rate,
        "NUM_EPOCHS": parse.num_epochs,
        "NUM_CLASSES": parse.num_classes,
        "LATENT_DIM": parse.latent_dim,
        "RANDOM_SEED": parse.random_seed,
    }