import dictionary_generation
import tgsd_home, tgsd_outlier, tgsd_clustering
import mdtd_home, mdtd_outlier, mdtd_clustering
import tgsd_smartsearch
import taxi_demo
import json
from pyfiglet import Figlet
TGSD_Driver = tgsd_home.TGSD_Home
TGSD_Outlier = tgsd_outlier.TGSD_Outlier
TGSD_Cluster = tgsd_clustering.TGSD_Cluster
MDTD_Driver = mdtd_home.MDTD_Home
MDTD_Outlier = mdtd_outlier.MDTD_Outlier
MDTD_Cluster = mdtd_clustering.MDTD_Cluster
Taxi_Demo = taxi_demo.Taxi_Demo
Gen_Dict = dictionary_generation.GenerateDictionary

if __name__ == '__main__':
    # Fun logo :)
    print(Figlet(font='starwars').renderText('PySpady'))

    while True:
        print("Welcome to PySpady, developed by Michael Paglia, Michael Smith, Joseph Regan, and Proshanto Dabnath.")
        print("Before starting, would you like to see some demo data?")
        while (userinput := input("[y]es, [n]o: ")) not in ['y', 'n']:
            pass
        if(userinput == 'y'):
            print("1.) Taxi Demo")
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
                    while (method := input("Enter a method [p]ickup or [d]ropoff: ")) not in ['p', 'd']:
                        pass
                    method = 'pickup' if method == 'p' else 'dropoff'

                    print("Enter the perspective for TGSD.")
                    while (perspective := input("Enter a method [p]point, [r]ow, [c]olumn: ")) not in ['p','r','c']:
                        pass
                    perspective = 'point' if perspective == 'p' else ('col' if perspective == 'c' else 'row')

                    print("Please designate whether to run auto search.")
                    while (auto := input("Would you like to run auto search; [y]es, [n]o: ")) not in ['y','n']:
                        pass
                    auto = True if auto == 'y' else False

                    Taxi_Demo = taxi_demo.Taxi_Demo(month, method, perspective, auto)
                    Taxi_Demo.clean_and_run()
                else:
                    print("Enter the perspective for MDTD.")
                    while (perspective := input("Enter a method [p]point, [r]ow, [c]olumn: ")) not in ['p','r','c']:
                        pass
                    perspective = 'point' if perspective == 'p' else ('col' if perspective == 'c' else 'row')

                    Taxi_Demo = taxi_demo.Taxi_Demo(month, method="both", perspective=perspective, auto=False)
                    Taxi_Demo.clean_and_run()
        else:
            while (tgsd_or_mdtd := input("Enter the method for [T]GSD or [M]DTD: ")) not in ['T', 'M']: pass

            if tgsd_or_mdtd == "T":
                print("Do you have a config file already set up?")
                while(userinput := input("[y]es, [n]o: ")) not in ['y','n']:
                    pass
                if(userinput == 'y'):
                    print("Is the path different from config.json?")
                    while(userinput := input("[y]es, [n]o: ")) not in ['y','n']:
                        pass
                    path = input("Enter path: ") if userinput == 'y' else "config.json"
                    [x, psi_d, phi_d, mask] = tgsd_home.TGSD_Home(path).config_run(config_path = path)
                    Y, W = tgsd_home.TGSD_Home(path).tgsd(x, psi_d, phi_d, mask)

                else:
                    print("Would you like to use the autoconfig?")
                    while(userinput := input("[y]es, [n]o: ")) not in ['y','n']:
                        pass
                    if(userinput == 'y'):
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

                        if(mask_mode == 'path'):
                            print("Please enter a mask path.")
                            mask_path = input("Enter a path: ")

                        print("Please enter the the mask percent.")
                        while True: # So the program doesn't crash on non-numeric input
                            try:
                                if mask_percent := int(input("Enter a number [1-100]: ")) in range(1, 101):
                                    break
                            except ValueError:
                                pass

                        print("Please enter the first dimension of the data as a numerical value.")
                        while True: # So the program doesn't crash on non-numeric input
                            try:
                                if first_x_dimension := int(input("Enter a number: ")):
                                    break
                            except ValueError:
                                pass

                        print("Please enter the second dimension of the data as a numerical value.")
                        while True: # So the program doesn't crash on non-numeric input
                            try:
                                if second_x_dimension := int(input("Enter a number: ")):
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

                        config_json = {
                            'x': csv_path,
                            'adj_path': adj_path,
                            'psi': psi_d,
                            'phi': phi_d,
                            'mask_mode': mask_mode,
                            'mask_percent': mask_percent,
                            'first_x_dimension': first_x_dimension,
                            'second_x_dimension': second_x_dimension,
                            'adj_square_dimension': adj_square_dimension
                        }

                        if mask_mode == 'path':
                            config_json['mask_path'] = mask_path

                        with open(config_path, "w") as outfile:
                            json.dump(config_json, outfile)

                        [x, psi_d, phi_d, mask] = tgsd_home.TGSD_Home(config_path).config_run(config_path = config_path)
                        tgsd_home.TGSD_Home(config_path).tgsd(x, psi_d, phi_d, mask)

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
                while (synorno := input("Would you like to use the synthetic data first as an example? [y]es, [n]o: ")) not in ['y', 'n']:
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
                        if perspective == "x":
                            mdtd_outlier.MDTD_Outlier.mdtd_find_outlier(tensor, recon_t, int(count_outlier), x)
                        elif perspective == "y":
                            mdtd_outlier.MDTD_Outlier.mdtd_find_outlier(tensor, recon_t, int(count_outlier), y)
                        else:
                            mdtd_outlier.MDTD_Outlier.mdtd_find_outlier(tensor, recon_t, int(count_outlier), z)
                    else:
                        num_clusters = input("How many clusters would you like to plot? The maximum is 10. ")
                        mdtd_clustering.MDTD_Cluster.mdtd_clustering(phi_y, int(num_clusters))

            q = input("Would you like to quit? [y]es, [n]o")
            if q == "y":
                print("Quitting PySpady...")
                break
