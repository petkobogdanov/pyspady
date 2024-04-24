import dictionary_generation
import tgsd_home, tgsd_outlier, tgsd_clustering
import mdtd_home, mdtd_outlier, mdtd_clustering
import tgsd_smartsearch
#import taxi_demo
import json
from pyfiglet import Figlet
TGSD_Driver = tgsd_home.TGSD_Home
TGSD_Outlier = tgsd_outlier.TGSD_Outlier
TGSD_Cluster = tgsd_clustering.TGSD_Cluster
MDTD_Driver = mdtd_home.MDTD_Home
MDTD_Outlier = mdtd_outlier.MDTD_Outlier
MDTD_Cluster = mdtd_clustering.MDTD_Cluster
#Taxi_Demo = taxi_demo.Taxi_Demo
Gen_Dict = dictionary_generation.GenerateDictionary

if __name__ == '__main__':
    # Fun logo :)
    print(Figlet(font='starwars').renderText('PySpady'))

    while True:
        print("Welcome to PySpady, developed by Michael Paglia, Michael Smith, Joseph Regan, and Proshanto Dabnath.")
        print("Before starting, would you like to see some real-world demonstration data?")
        while (userinput := input("[y]es, [n]o: ")) not in ['y', 'n']:
            pass
        if(userinput == 'y'):
            print("1.) Taxi Demo: This dataset contains taxi pickup and dropoff volume across over 300 locations in NYC in 2017.")
            while(userinput := input("Enter number [1-1]: ")) not in ['1']:
                pass
            if(userinput == '1'):
                print("Enter the month as a numerical value.")
                while True: # So the program doesn't crash on non-numeric input
                    try:
                        month = int(input("Enter a number [1-12]: "))
                        if month in range(1, 13):
                            break
                    except ValueError:
                        pass

                while (tgsd_or_mdtd := input("Enter the method for [T]GSD or [M]DTD: ")) not in ['T', 'M']: pass
                if tgsd_or_mdtd == "T":
                    while (tgsd_model_input := input("Enter the optimizer for TGSD: [A]lternating Direction Method of Multipliers or [G]radient Descent: ")) not in ["A", "G"]: pass
                    tgsd_model = "admm" if tgsd_model_input == "A" else "gd"
                    while (method := input("Enter a method for taxi [p]ickups or [d]ropoffs: ")) not in ['p', 'd']:
                        pass
                    method = 'pickup' if method == 'p' else 'dropoff'

                    print("Enter the perspective for TGSD. This determines how you choose to view outliers, where rows represent physical locations and columns represent time. ")
                    while (perspective := input("Enter a method [p]point, [r]ow, [c]olumn: ")) not in ['p','r','c']:
                        pass
                    perspective = 'point' if perspective == 'p' else ('col' if perspective == 'c' else 'row')

                    print("Please designate whether to run auto search. This will automatically search for an optimal combination of hyperparameters.")
                    while (auto := input("Would you like to run auto search; [y]es, [n]o: ")) not in ['y','n']:
                        pass
                    auto = True if auto == 'y' else False

                    Taxi_Demo = taxi_demo.Taxi_Demo(month, method, perspective, auto, optimizer_method=tgsd_model)
                    Taxi_Demo.clean_and_run()
                else:
                    print("Enter the perspective for MDTD.")
                    while (perspective := input("Enter a method [p]point, [r]ow, [c]olumn: ")) not in ['p','r','c']:
                        pass
                    perspective = 'point' if perspective == 'p' else ('col' if perspective == 'c' else 'row')

                    Taxi_Demo = taxi_demo.Taxi_Demo(month, method="both", perspective=perspective, auto=False, optimizer_method=tgsd_model)
                    Taxi_Demo.clean_and_run()
        else:
            while (tgsd_or_mdtd := input("Enter the method for [T]GSD or [M]DTD: ")) not in ['T', 'M']: pass

            if tgsd_or_mdtd == "T":
                while (tgsd_model_input := input("Enter the optimizer for TGSD: [A]lternating Direction Method of Multipliers or [G]radient Descent: ")) not in ["A", "G"]: pass
                tgsd_model = "admm" if tgsd_model_input == "A" else "gd"

                while (synorno := input("Would you like to use the synthetic data first as an example? [y]es, [n]o: ")) not in ['y', 'n']:
                    pass
                if synorno == "y":
                    while (tgsd_screening_input := input("Would you like to use screening? [y]es, [n]o: ")) not in ['y', 'n']: pass
                    tgsd_screening_input = True if tgsd_screening_input == 'y' else False
                    
                    [x, psi_d, phi_d, mask, k, lam1, lam2, lam3, learning_rate] = tgsd_home.TGSD_Home("tgsd_syn_config.json").config_run(config_path="tgsd_syn_config.json", screening_flag=tgsd_screening_input)
                    Y, W = tgsd_home.TGSD_Home("tgsd_syn_config.json").tgsd(x, psi_d, phi_d, mask, optimizer_method=tgsd_model)

                else:
                    print("Do you have a config file already set up?")
                    while(userinput := input("[y]es, [n]o: ")) not in ['y','n']:
                        pass
                    if(userinput == 'y'):
                        print("Is the path different from config.json?")
                        while(userinput := input("[y]es, [n]o: ")) not in ['y','n']:
                            pass
                        path = input("Enter path: ") if userinput == 'y' else "config.json"
                        [x, psi_d, phi_d, mask, k, lam1, lam2, lam3, learning_rate] = tgsd_home.TGSD_Home(path).config_run(config_path=path)
                        Y, W = tgsd_home.TGSD_Home(path).tgsd(x, psi_d, phi_d, mask, k=k, lambda_1=lam1, lambda_2=lam2, lambda_3=lam3, learning_rate=learning_rate, optimizer_method=tgsd_model)

                    else:
                        print("Would you like to use the autoconfig?")
                        while(userinput := input("[y]es, [n]o: ")) not in ['y','n']:
                            pass
                        if userinput == 'y':
                            residual_percent = input("Residual % [0.0, 0.99]: ")
                            coefficient_percent = input("Coefficient % [0.0, 0.99]: ")
                            tgsd_smartsearch = tgsd_smartsearch.CustomEncoder(config_path="config.json", demo=False,demo_X=None,demo_Phi=None,demo_Psi=None,demo_mask=None, coefficient_threshold=coefficient_percent, residual_threshold=residual_percent)
                            tgsd_smartsearch.run_smart_search()
                            Y, W = tgsd_smartsearch.get_Y_W()
                        else:
                            print("What would you like the created config file's path to be?")
                            config_path = input("Enter a path: ")

                            print("What is the path for the CSV?")
                            csv_path = input("Enter a path: ")

                            print("Enter a dictionary for PSI.")
                            while(psi_d := input("[g]ft, [r]am, [d]ft: ")) not in ['g', 'r', 'd']:
                                pass
                            psi_d = 'gft' if psi_d == 'g' else ('ram' if psi_d == 'r' else 'dft')

                            print("Enter a dictionary for PHI.")
                            while(phi_d := input("[g]ft, [r]am, [d]ft: ")) not in ['g', 'r', 'd']:
                                pass
                            phi_d = 'gft' if phi_d == 'g' else ('ram' if phi_d == 'r' else 'dft')

                            print("Enter a mask mode for TGSD.")
                            while(mask_mode := input("[l]inear, [r]andom, [p]ath: ")) not in ['l', 'r', 'p']:
                                pass
                            mask_mode = 'lin' if mask_mode == 'l' else ('rand' if mask_mode == 'r' else 'path')

                            if mask_mode == 'path':
                                print("Please enter a mask path.")
                                mask_path = input("Enter a path: ")

                            print("Please enter the mask percent.")
                            while True: # So the program doesn't crash on non-numeric input
                                try:
                                    if mask_percent := int(input("Enter a number [1-100]: ")) in range(1, 101):
                                        break
                                except ValueError:
                                    pass

                            print("Please enter the value for K.")
                            while True: # So the program doesn't crash on non-numeric input
                                try:
                                    if k := int(input("Enter a number: ")) in range(1, 1001):
                                        break
                                except ValueError:
                                    pass

                            print("Please enter the value for lambda #1.")
                            while True: # So the program doesn't crash on non-numeric input
                                try:
                                    if lambda_1_value := float(input("Enter a decimal or whole number.")) in range(0, 101):
                                        break
                                except ValueError:
                                    pass

                            print("Please enter the value for lambda #2.")
                            while True: # So the program doesn't crash on non-numeric input
                                try:
                                    if lambda_2_value := float(input("Enter a decimal or whole number.")) in range(0, 101):
                                        break
                                except ValueError:
                                    pass

                            print("Please enter the value for lambda #3.")
                            while True: # So the program doesn't crash on non-numeric input
                                try:
                                    if lambda_3_value := float(input("Enter a decimal or whole number.")) in range(0, 101):
                                        break
                                except ValueError:
                                    pass

                            print("Please enter the learning rate.")
                            while True: # So the program doesn't crash on non-numeric input
                                try:
                                    if learning_rate_value := float(input("Enter a small number: ")) in range(0, 10):
                                        break
                                except ValueError:
                                    pass

                            print("Please enter the dimension of the adjacency list as a numerical value.")
                            while True: # So the program doesn't crash on non-numeric input
                                try:
                                    if adj_square_dimension := int(input("Enter a number: ")):
                                        break
                                except ValueError:
                                    pass

                            print("Please enter the path for the adjacency list.")
                            adj_path = input("Enter a path: ")

                            while (tgsd_screening_input := input("Would you like to use screening? [y]es, [n]o: ")) not in ['y', 'n']: pass
                            tgsd_screening_input = True if tgsd_screening_input == 'y' else False

                            config_json = {
                                'x': csv_path,
                                'adj_path': adj_path,
                                'psi': psi_d,
                                'phi': phi_d,
                                "k": k,
                                "lam_1": lambda_1_value,
                                "lam_2": lambda_2_value,
                                "lam_3": lambda_3_value,
                                "learning_rate": learning_rate_value,
                                'mask_mode': mask_mode,
                                'mask_percent': mask_percent,
                                'adj_square_dimension': adj_square_dimension,
                                "screening": tgsd_screening_input
                            }

                            if mask_mode == 'path':
                                config_json['mask_path'] = mask_path

                            with open(config_path, "w") as outfile:
                                json.dump(config_json, outfile)

                            [x, psi_d, phi_d, mask, k, lam1, lam2, lam3, learning_rate] = tgsd_home.TGSD_Home(config_path).config_run(config_path=config_path)
                            Y, W = tgsd_home.TGSD_Home(config_path).tgsd(x, psi_d, phi_d, mask, k=k, lambda_1=lam1, lambda_2=lam2, lambda_3=lam3, learning_rate=learning_rate, optimizer_method=tgsd_model)

                print(f"Would you like to return {mask.shape[1]} missing (masked) values? ")
                userinput = input("[y]es, [n]o ")
                if userinput == "y":
                    # Returns missing values, downloads new CSV and displays graph of imputed values
                    tgsd_home.TGSD_Home.return_missing_values(mask, psi_d, phi_d, Y, W)
                    print(f".csv of {mask.shape[1]} imputed values downloaded to imputed_values.csv.")
                print("Would you like to perform downstream tasks on your output? ")
                userinput = input("[y]es, [n]o: ")
                if userinput == "y":
                    print("Enter the method for TGSD.")
                    while (method := input("Enter a downstream task [o]utlier detection, [c]ommunity detection. ")) not in ['o','c']:
                        pass
                    if method == "o":
                        print("Enter the perspective for TGSD.")
                        while (perspective := input("Enter a method [p]oint, [r]ow, [c]olumn: ")) not in ['p','r','c']:
                            pass
                        count_outlier = input("How many outliers would you like to plot? The maximum that can be plotted on a graph is 10. ")
                        if perspective == "p":
                            tgsd_outlier.TGSD_Outlier.find_outlier(x, psi_d, Y, W, phi_d, int(count_outlier))
                        elif perspective == "r":
                            tgsd_outlier.TGSD_Outlier.find_row_outlier(x, psi_d, Y, W, phi_d, int(count_outlier))
                        else:
                            tgsd_outlier.TGSD_Outlier.find_col_outlier(x, psi_d, Y, W, phi_d, int(count_outlier))
                    else:
                        print("Performing clustering...")
                        tgsd_clustering.TGSD_Cluster.cluster(psi_d, Y)

            else:
                while (synorno := input("Would you like to use the synthetic data first as an example? \n[y]es, [n]o: ")) not in ['y', 'n']:
                    pass
                if synorno == "n":
                    print("Do you have a config?")
                    while(userinput := input("[y]es, [n]o: ")) not in ['y','n']:
                        pass
                    if(userinput == 'y'):
                        print("Is the path different from mdtd_config.json?")
                        while(userinput := input("[y]es, [n]o: ")) not in ['y','n']:
                            pass
                        path = input("Enter path: ") if userinput == 'y' else "mdtd_config.json"
                        obj2 = MDTD_Driver(config_path=path)
                        tensor, recon_t, phi_y = obj2.mdtd(is_syn=False, X=obj2.X, adj1=obj2.adj_1, adj2=obj2.adj_2, mask=obj2.mask, count_nnz=obj2.count_nnz, num_iters_check=obj2.num_iters_check, lam=obj2.lam, K=obj2.K, epsilon=obj2.epsilon)

                else:
                    obj2 = MDTD_Driver(config_path="mdtd_config.json")
                    tensor, recon_t, phi_y = obj2.mdtd(is_syn=True, X=obj2.X, adj1=obj2.adj_1, adj2=obj2.adj_2, mask=obj2.mask)

                print(f"Would you like to return {len(obj2.mask)} missing (masked) values? ")
                userinput = input("[y]es, [n]o ")
                if userinput == "y":
                    # Returns missing values, downloads new CSV and displays graph of imputed values
                    MDTD_Driver.return_missing_values(obj2.mask, recon_t)
                    print(f".csv of {len(obj2.mask)} imputed values downloaded to tensor_imputed_values.csv.")

                print("Would you like to perform downstream tasks on your output?")
                userinput = input("[y]es, [n]o: ")
                if userinput == "y":
                    print("Enter the method for MDTD.")
                    while (method := input("Enter a downstream task [o]utlier detection, [c]ommunity detection. ")) not in ['o','c']:
                        pass
                    if method == "o":
                        print("Enter the perspective for MDTD.")
                        while (perspective := input("Enter a slice method [x]-axis, [y]-axis, [z]-axis: ")) not in ['x','y','z']:
                            pass
                        count_outlier = input("How many outliers would you like to plot? The maximum that can be plotted on a graph is 10. ")
                        mdtd_outlier.MDTD_Outlier.mdtd_find_outlier(tensor, recon_t, int(count_outlier), perspective)
                    else:
                        num_clusters = input("How many clusters would you like to plot? The maximum is 10. ")
                        mdtd_clustering.MDTD_Cluster.mdtd_clustering(phi_y, int(num_clusters))

            q = input("Would you like to quit?\n[y]es, [n]o ")
            if q == "y":
                print("Quitting PySpady...")
                break
