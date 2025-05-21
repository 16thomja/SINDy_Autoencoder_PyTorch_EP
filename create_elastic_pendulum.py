import os
import numpy as np
from cmd_line import parse_args
from src.dataset.elastic_pendulum_functions import get_elastic_pendulum_data
from src.utils.other import get_data_paths


def main():
    # get args
    args = parse_args()
    
    # create data
    train_data = get_elastic_pendulum_data(
        n_ics=args.train_initial_conds, 
        timesteps=args.timesteps,
        k=args.spring_constant,
        m=args.mass,
        L=args.natural_length,
        g=args.gravitational_acceleration
    )
    val_data = get_elastic_pendulum_data(
        n_ics=args.train_initial_conds, 
        timesteps=args.timesteps,
        k=args.spring_constant,
        m=args.mass,
        L=args.natural_length,
        g=args.gravitational_acceleration
    )
    test_data = get_elastic_pendulum_data(
        n_ics=args.train_initial_conds, 
        timesteps=args.timesteps,
        k=args.spring_constant,
        m=args.mass,
        L=args.natural_length,
        g=args.gravitational_acceleration
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