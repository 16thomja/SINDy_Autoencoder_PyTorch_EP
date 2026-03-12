import os
import secrets
import numpy as np
from cmd_line import parse_args
from src.dataset.elastic_pendulum_functions import get_elastic_pendulum_data
from src.utils.other import get_data_paths


def main():
    # get args
    args = parse_args()

    seed = args.dataset_seed if args.dataset_seed is not None else secrets.randbits(32)
    seed_seq = np.random.SeedSequence(seed)
    train_rng, val_rng, test_rng = [np.random.default_rng(s) for s in seed_seq.spawn(3)]
    
    # create data
    train_data = get_elastic_pendulum_data(
        n_ics=args.train_initial_conds, 
        timesteps=args.timesteps,
        k=args.spring_constant,
        m=args.mass,
        L=args.natural_length,
        g=args.gravitational_acceleration,
        rng=train_rng,
        metadata={"seed": seed, "split": "train"},
    )
    val_data = get_elastic_pendulum_data(
        n_ics=args.val_initial_conds, 
        timesteps=args.timesteps,
        k=args.spring_constant,
        m=args.mass,
        L=args.natural_length,
        g=args.gravitational_acceleration,
        rng=val_rng,
        metadata={"seed": seed, "split": "val"},
    )
    test_data = get_elastic_pendulum_data(
        n_ics=args.test_initial_conds, 
        timesteps=args.timesteps,
        k=args.spring_constant,
        m=args.mass,
        L=args.natural_length,
        g=args.gravitational_acceleration,
        rng=test_rng,
        metadata={"seed": seed, "split": "test"},
    )

    # save data
    folder, data_paths = get_data_paths()
    if not os.path.isdir(folder):
        os.system("mkdir -p " + folder)
    np.save(data_paths[0], train_data)
    np.save(data_paths[1], val_data)
    np.save(data_paths[2], test_data)
    

if __name__ == '__main__':
    main()
