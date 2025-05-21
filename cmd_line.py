import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description="Template")

    # anything that affects the name of the saved folders (for checkpoints, experiments, tensorboard)
    # path saved as: initial_folder/data_set/model/session_name
    # where initial_folder is experiments, model_folder, or tensorboard_folder
    parser.add_argument('-sess', '--session_name', default="7-27-2022_1", type=str, help="Session name")
    parser.add_argument('-M',  '--model', default="SINDyAE_o2", type=str, help="Model to use (SINDyAE_o2, SINDyConvAE_o2")
    parser.add_argument('-EX', '--experiments', default='./experiments/', type=str, help="Output folder for experiments")
    parser.add_argument('-MF', '--model_folder', default='./trained_models/', type=str, help="Output folder for experiments")
    parser.add_argument('-TB', '--tensorboard_folder', default='./tb_runs/', type=str, help="Output folder for tensorboard")
    parser.add_argument('-DT', '--data_set', default='elastic_pendulum', type=str, help="Which dataset to use (elastic_pendulum)")
    
    # network parameters
    parser.add_argument('-Z', '--z_dim', default=2, type=int, help="Size of latent vector")
    parser.add_argument('-U',  '--u_dim', default=2601, type=int, help="Sise of u vector in Elastic Pendulum data")
    parser.add_argument('-HD', '--hidden_dims', default=[128, 64, 32], type=int, nargs='+', help="Dimensions of hidden layers in FC autoencoder")
    parser.add_argument('-UI', '--use_inverse', default=True, type=bool, help="Iff true, includes inverse of state in library")
    parser.add_argument('-US', '--use_sine', default=True, type=bool, help="Iff true, includes sine of state in library")
    parser.add_argument('-UC', '--use_cosine', default=True, type=bool, help="Iff true, includes cosine of state in library")
    parser.add_argument('-PO', '--poly_order', default=3, type=str, help="Highest polynomial degree to include in library")
    parser.add_argument('-IC', '--include_constant', default=True, type=bool, help="Iff true, includes constant term in library")
    parser.add_argument('-NL', '--nonlinearity', default='elu', type=str, help="Nonlinearity to use in autoencoder (elu, sig, relu, None)")
    
    # training parameters
    parser.add_argument('-E', '--epochs', default=100, type=int, help="Number of epochs to train for")
    parser.add_argument('-LR', '--learning_rate', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('-ARE', '--adam_regularization', default=1e-5, type=float, help="Regularization to use in ADAM optimizer")
    parser.add_argument('-GF', '--gamma_factor', default=0.995, type=float, help="Learning rate decay factor")
    parser.add_argument('-BS', '--batch_size', default=50, type=int, help="Batch size")
    parser.add_argument('-L1', '--lambda_ddx', default=5e-4, type=float, help="Weight of ddx loss")
    parser.add_argument('-L2', '--lambda_ddz', default=5e-5, type=float, help="Weight of ddz loss")
    parser.add_argument('-L3', '--lambda_reg', default=1e-5, type=float, help="Weight of regularization loss")
    parser.add_argument('-C', '--clip', default=None, type=float, help="Gradient clipping value during training (None for no clipping)")
    parser.add_argument('-TI', '--test_interval', default=1, type=int, help="Epoch interval to evaluate on val (test) data during training")
    parser.add_argument('-CPI', '--checkpoint_interval', default=1, type=int, help="Epoch interval to save model during training")
    parser.add_argument('-ST', '--sequential_threshold', default=5e-2, type=float, help="Sequential thresholding value for coefficients")

    # dataset parameters
    parser.add_argument('-K', '--spring_constant', default=24.0, type=float, help='Spring constant in simulation')
    parser.add_argument('-MA', '--mass', default=1.0, type=float, help='Mass of pendulum bob in simulation')
    parser.add_argument('-NLE', '--natural_length', default=1.0, type=float, help='Natural length of spring in simulation')
    parser.add_argument('-GA', '--gravitational_acceleration', default=9.81, type=float, help='Gravitational acceleration in simulation')
    parser.add_argument('-TIC', '--train_initial_conds', default=100, type=int, help='Number of initial conditions in the training set')
    parser.add_argument('-VIC', '--val_initial_conds', default=10, type=int, help='Number of initial conditions in the validation set')
    parser.add_argument('-TEIC', '--test_initial_conds', default=10, type=int, help='Number of initial conditions in the test set')
    parser.add_argument('-TS', '--timesteps', default=500, type=int, help='Number of timesteps per trajectory')

    # other
    parser.add_argument('-LCP', '--load_cp', default=0, type=int, help='If 1, loads the model from the checkpoint. If 0, does not')
    parser.add_argument('-D', '--device', default=0, type=int, help='Which GPU to use')
    parser.add_argument('-PF', '--print_folder', default=1, type=int, help='Iff true, prints the folder for different logs')

    return parser.parse_args() 