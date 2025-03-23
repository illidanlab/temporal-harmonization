import os
import argparse
from time import localtime, strftime
from tools.utils import load_raw_data
from Data.DataInit import data_init
from Data.DataPreProcessing import data_pre_processing
from configs.cfg import init_cfg
import pickle
import matplotlib.pyplot as plt
import numpy as np

def main(cfg_proj, cfg_m):
    from Solvers.Solver_loader import solver_loader
    solver = solver_loader(cfg_proj, cfg_m)

    # Load raw data
    dic_id2feature, df_labels, nl_subject, mci_subject = load_raw_data(cfg_proj)
    
    solver.setLabels(df_labels)

    AUCs_permute_index = []
    for step in range(cfg_proj.num_total_runs):
        seed = step if cfg_proj.seed is None else cfg_proj.seed
        solver.set_random_seed(seed)
        
        # Split to train and test
        x_train, y_train, g_train, x_test, y_test, g_test = data_init(cfg_proj, mci_subject, nl_subject, dic_id2feature, df_labels, seed)
        
        x_train, y_train, g_train, x_test, y_test, g_test = data_pre_processing(cfg_proj, cfg_m, x_train, y_train, g_train, x_test, y_test, g_test)
        
        # Run the experiment
        auc, f1, sens, spec, auc_sbj, f1_sbj, sens_sbj, spec_sbj = solver.run(x_train, y_train, g_train, x_test, y_test, g_test, seed)

        if cfg_proj.experiment == "FeatureImportance":
            AUCs_permute_index.append(solver.AUC_permute_index)
        
        print("step-%d, sbj:auc=%.3f,f1=%.3f,sens=%.3f,spec=%.3f"%(seed, auc_sbj, f1_sbj, sens_sbj, spec_sbj))

    # print results
    AUC, AUC_std = solver.save_results()

    if cfg_proj.experiment == "FeatureImportance":
        return AUCs_permute_index

    return AUC, AUC_std
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="template")
    
    # Basic Configuration
    parser.add_argument("--gpu", type=str, default="0", required=False)
    parser.add_argument("--seed", type=int, default = None, required=False) 
    parser.add_argument("--num_total_runs", type=int, default = 100, required=False) 
    parser.add_argument("--flag_generatePredictions", default = ["Sex", "Edu", "Age"])
    parser.add_argument("--number_of_feature", type=int, default = 99, required=False)  
    parser.add_argument("--vote_threshold", type=int, default = 0.5, required=False)

    # Standard_solver, subject_harmonization_solver, temporal_solver, temporal_sequence_harmonization_adversarial_solver
    parser.add_argument("--solver", type=str, default= "temporal_sequence_harmonization_adversarial_solver", required=False)  
    parser.add_argument("--model_name", type = str, default="LSTM", required=False) # LSTM, GRU, RNN, LR, MLP

    # Performance, MaskTest, Coefficients, Main, FeatureImportance
    parser.add_argument("--experiment", type=str, default= "Coefficients")

    # Configuration for Sequence
    parser.add_argument("--mask", type = int, default = 80, required=False)  
    parser.add_argument("--mask_test", type = int, default = 0, required=False)   
    parser.add_argument("--test_length", type = int, default=-1, required=False) 
    parser.add_argument("--permute_index", type = int, default=-1, required=False)

    # Flag
    parser.add_argument("--flag_log", type=str, default = True, required=False) 
    parser.add_argument("--save_harmonized_features", type=bool, default = False, required=False) 
    parser.add_argument("--flag_time", type=str, default = strftime("%Y-%m-%d_%H-%M-%S", localtime()), required=False)
    parser.add_argument("--flag_load", type=str, default = None, required=False)    #if is not None, then the file of loaded para need to contain the str
    cfg_proj = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(cfg_proj.gpu)

    # Make sure match the model type with method
    if "temporal" in cfg_proj.solver and cfg_proj.model_name not in ["LSTM", "GRU", "RNN"]:
        cfg_proj.model_name = "LSTM"
    elif "temporal" not in cfg_proj.solver and cfg_proj.model_name not in ["LR", "MLP"]:
        cfg_proj.model_name = "MLP"
        
    # Save Harmonized Models for Visualization (need to run python visualize/visualize_last_hidden.py after)
    if cfg_proj.save_harmonized_features:
        cfg_proj.num_total_runs = 1
        cfg_proj.seed = 6

    # Different Loss Coefficient
    if cfg_proj.experiment == "Coefficients":
        weighting_loss = [
            [1.0, 0, 0],
            [0.5, 0, 0],
            [0.1, 0, 0],

            [0.5, 0.7, 0],
            [0.5, 0.5, 0],
            [0.5, 0.3, 0],

            [0.5, 0, 0.2],
            [0.5, 0, 0.1],
            [0.5, 0, 0.05],

            [0.5, 0.5, 0.2],
            [0.5, 0.5, 0.1],
            [0.5, 0.3, 0.2],
            [0.5, 0.3, 0.1],
        ]
        
        for weight in weighting_loss:
            print(weight)
            cfg_m = init_cfg(cfg_proj)
            cfg_m.alpha1 = weight[0]
            cfg_m.alpha2 = weight[1]
            cfg_m.alpha3 = weight[2]
            main(cfg_proj, cfg_m)

    # Masking Test Experiment
    if cfg_proj.experiment == "MaskTest":
        masks = [i for i in range(0, 100, 5)]

        plt.figure(figsize=(12, 8))
        for cfg_proj.solver in ["temporal_solver", "temporal_sequence_harmonization_adversarial_solver"]:
            for cfg_proj.model_name in ["GRU", "LSTM"]:
                AUCs = []
                AUCs_std = []
                for cfg_proj.mask_test in masks:
                    print(cfg_proj.mask_test)
                    cfg_m = init_cfg(cfg_proj)
                    AUC, AUC_std = main(cfg_proj, cfg_m)
                    AUCs.append(AUC)
                    AUCs_std.append(AUC_std)
                
                print(cfg_proj.solver, cfg_proj.model_name)
                plt.plot(masks, AUCs, label = ("No Harmonization" if cfg_proj.solver == "temporal_solver" else "Temporal Harmonization") + "-" + cfg_proj.model_name, marker = ("o" if cfg_proj.solver == "temporal_solver" else "x"))
                print(AUCs, AUCs_std)
                
        plt.grid()
        plt.xlabel("Masking Ratio For Test Subject", fontsize = 15)
        plt.ylabel("AUC", fontsize = 15)
        plt.legend(loc = "upper right", fontsize = 10, ncol=2)

        plt.savefig("figures/TestMaskingRatio.png", dpi = 300)
        plt.savefig("figures/TestMaskingRatio.pdf", dpi = 300)
   
    # Feature Importance Experiment
    if cfg_proj.experiment == "FeatureImportance":        
        cfg_m = init_cfg(cfg_proj)
        AUCs_permute_index = main(cfg_proj, cfg_m)
        with open(f"checkpoints/feature_importance_{cfg_proj.solver}.pkl", "wb") as f:
            pickle.dump(np.mean(AUCs_permute_index, axis = 0), f)

    # Main Performance Experiment
    if cfg_proj.experiment == "Performance":
        for cfg_proj.solver in ["temporal_solver", "temporal_sequence_harmonization_adversarial_solver"]:
            for cfg_proj.model_name in ["GRU", "LSTM"]:
                print(cfg_proj.solver, cfg_proj.model_name)
                cfg_m = init_cfg(cfg_proj)
                main(cfg_proj, cfg_m)

    # Main
    if cfg_proj.experiment == "Main":
        cfg_m = init_cfg(cfg_proj)
        AUC, _ = main(cfg_proj, cfg_m)