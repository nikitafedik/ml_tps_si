# for more polished version of traning scripts, please visit official examples: https://github.com/lanl/hippynn/tree/development/examples
# and docs:  

import os
import torch
from hippynn.graphs import inputs, networks, targets, physics
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau,PatienceController
from hippynn.graphs import loss
from hippynn.experiment import assemble_for_training
from hippynn.databases import DirectoryDatabase
from hippynn.pretraining import set_e0_values
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau,PatienceController
from hippynn.experiment import setup_and_train
import hippynn

os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "0" #supress zillions of numba warning at each step
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
hippynn.settings.WARN_LOW_DISTANCES = False

torch.set_default_dtype(torch.float32)
device = torch.device('cuda')
import hippynn
from hippynn.graphs.nodes.base.multi import IndexNode

# Define model name
netname = "hipnn_ani_tps_10800_dataseed0_modelseed0"
dirname = netname
if not os.path.exists(dirname):
    os.mkdir(dirname)
else:
    pass
os.chdir(dirname)

# Define architecture params

with hippynn.tools.log_terminal("training_log.txt", 'wt'):

    network_params = {
        "possible_species": [0,1,6,7,8],   # Z values of the elements
        'n_features': 128,                     # Number of neurons at each layer
        "n_sensitivities": 20,                 # Number of sensitivity functions in an interaction layer
        "dist_soft_min": 0.75,                 #
        "dist_soft_max": 5.5,
        "dist_hard_max": 6.5,
        "n_interaction_layers": 2,            # Number of interaction blocks; 2 - neighbours of neighbours
        "n_atom_layers": 4,                   # Linear layers
        "sensitivity_type": 'inverse',        # Number of atom layers in an interaction block
    }

    # Define a model
    # data defined in arrays (Z, R, T, G) is expected to be in .npy arrays
    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    network = networks.HipnnQuad("HIPNN", (species, positions), module_kwargs = network_params)
    henergy = targets.HEnergyNode("HEnergy",network)
    force = physics.GradientNode("forces", (henergy, positions), sign=1) # 1 for gradients; -1 for forces
    molecule_energy = henergy.mol_energy
    molecule_energy.db_name="T"
    force.db_name="G"  # G for gradient

    hierarchicality = henergy.hierarchicality

    # Define loss quantities
    from hippynn.graphs import loss
    rmse_energy = loss.MSELoss(molecule_energy.pred,molecule_energy.true)**(1/2)
    mae_energy = loss.MAELoss(molecule_energy.pred,molecule_energy.true)
    rsq_energy =  loss.Rsq(molecule_energy.pred,molecule_energy.true)
    
    pred_per_atom = physics.PerAtom("PeratomPredicted",(molecule_energy,species)).pred
    true_per_atom = physics.PerAtom("PeratomTrue",(molecule_energy.true,species.true))
    mae_per_atom = loss.MAELoss(pred_per_atom,true_per_atom)

    force_losses = {
        "F-RMSE": loss.MSELoss.of_node(force) ** (1 / 2),
        "F-MAE": loss.MAELoss.of_node(force),
        "F-RSQ": loss.Rsq.of_node(force),
    }

    loss_energy = (rmse_energy + mae_energy ) 
    loss_forces = force_losses["F-RMSE"] + force_losses["F-MAE"]
    loss_error=loss_energy + loss_forces # equal weight for E and grads
    
    rbar = loss.Mean.of_node(hierarchicality)
    l2_reg = loss.l2reg(network)
    loss_regularization = 1e-6 * l2_reg + rbar    # L2 regularization and hierarchicality regularization

    train_loss = loss_error + loss_regularization

    validation_losses = {
        "T-RMSE"      : rmse_energy,
        "T-MAE"       : mae_energy,
        "T-RSQ"       : rsq_energy,
        "TperAtom MAE": mae_per_atom,
        "T-Hier"      : rbar,
        "L2Reg"       : l2_reg,
        "Loss-Err"    : loss_error,
        "Loss-Reg"    : loss_regularization,
        "Loss"        : train_loss, 
    }
    validation_losses.update(force_losses)

    early_stopping_key = "Loss-Err" 

    #Assemble for training
    training_modules,db_info = assemble_for_training(train_loss, validation_losses)
    training_modules[0].print_structure()

    # RUN MODEL 

    database_params = {
        'name': 'ani_10800_seed0-',                            
        'directory':'<INSERT YOUR PATH TO ANI-1X HERE>', 
        'quiet': False,
        'test_size':0.1,
        'valid_size':0.1,
        'num_workers': 2, # dataloader in 
        'seed':0,
        **db_info                
    }

    from hippynn.pretraining import set_e0_values # substract self-energies: keep only formations energies

    database = DirectoryDatabase(**database_params)
    set_e0_values(henergy,database,trainable_after=False)
    optimizer = torch.optim.Adam(training_modules.model.parameters(),lr=0.0005)
    scheduler =  RaiseBatchSizeOnPlateau(optimizer=optimizer,
                                         max_batch_size=256,
                                         patience=5,
                                         factor=0.5)
    
    controller = PatienceController(optimizer=optimizer,
                                    scheduler=scheduler,
                                    batch_size=256,
                                    eval_batch_size=256,
                                    max_epochs = 150,
                                    termination_patience=10,
                                    stopping_key=early_stopping_key,
                                    )
     
    setup_params = hippynn.experiment.SetupParams(
        controller=controller,
        device=device,
    )

    setup_and_train(training_modules=training_modules,
                            database=database,
                            setup_params=setup_params,
                            )